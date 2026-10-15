import os
import torch
import networkx as nx
import json
from openai import Client
from typing import Dict, List, Tuple, Type, Optional
from pydantic import BaseModel, computed_field
from torch import Tensor
from definitions import EXPERIMENTS_DIR, EMS_DIR
from instance import cli
from torch_geometric.data import HeteroData
from src.commands.feature_extraction.main import EdgeType, Graph
from src.commands.feature_extraction.graph_utils import EdgeTypes
from src.commands.graph_learning.main import Trainer, Config as TrainerConfig, BaseTrainingConfig
from src.gnn import HetModelV2 as HetModel, HetModelReturnV2 as HetModelReturn
from src.shared.command_tools import BaseCommand
from src.shared.handlers import ConsoleLogger
from src.shared.env import env
from src import helpers

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def entropy(en : Tensor) -> Tensor:
    return -torch.sum(en * torch.log(en + 10e-5), dim=-1).clip(max=0)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class BEMConfig(BaseModel):

    # ── Graph & Model ──────────────────────────────────────────────────
    expertise_id : str
    model_id : str
    threshold:      float = 0.5   # min probability for a subdomain / lab label to be active
    k : int # Number of triples

    # ── Representative node selection ──────────────────────────────────
    llm : str = "deepseek-chat" # LLM used for llm-based labeling

    # ── Root Node ──────────────────────────────────────────────────────
    name : str
    description : str

    @computed_field
    @property
    def experiment(self) -> str:
        return self.model_id.split('/')[0]
    
    @computed_field
    @property
    def model_name(self) -> str:
        return self.model_id.split('/')[1]
    
    @computed_field
    @property
    def model_path(self) -> str:
        return os.path.join(EXPERIMENTS_DIR, self.experiment, 'models', f'model_{self.model_name}.pt')
    
    @computed_field
    @property
    def expertise_dir(self) -> str:
        return os.path.join(EMS_DIR, self.expertise_id)

class Config(BaseModel):

    expertise_id : str

    @computed_field
    @property
    def conf(self) -> BEMConfig:
        return BEMConfig(**{
            'expertise_id' : self.expertise_id,
            **helpers.load_json_file(os.path.join(EMS_DIR, self.expertise_id, 'config.json'))
        })

class Topic(BaseModel):
    title : str

@cli.register_stage(
    name="build_map",
    description=(
        "Runs the pipeline responsible for building the expertise map from the "
        "knowledge graph and labelling nodes and clusters."
    ),
    add_from_conf=False,
    add_save_to=False,
)
class ExpertiseMapBuilder(BaseCommand):

    def __init__(self, config: BEMConfig, client : Client) -> None:
        super().__init__()
        self.config = config
        self.client = client

    def load_data_and_model(self) -> Tuple[HeteroData, Graph, HetModel, BaseTrainingConfig]:

        self.send_info_event(
            progress=None,
            message="[Load] Starting — loading trainer config and data",
        )

        ### Domains
        trainer_cfg = TrainerConfig(experiment=self.config.experiment)
        trainer     = Trainer(experiment=trainer_cfg.experiment, config=trainer_cfg.training_config)
        configs     = trainer.get_configs()

        if self.config.model_name not in configs:

            raise KeyError(
                f"[Load] Model config '{self.config.model_name}' not found. "
                f"Available: {list(configs.keys())}"
            )

        run_config = configs[self.config.model_name]

        self.send_debug_event(
            progress=None,
            message=f"[Load] Using run config '{self.config.model_name}': {run_config}",
        )

        # Load data through the trainer (handles normalisation)
        data , graph = trainer.load_data(config=run_config,return_raw=True)

        # model
        model = trainer.create_model(config=run_config, data=data)
        state_dict = torch.load(self.config.model_path, weights_only=False, map_location="cpu")
        model.load_state_dict(state_dict)

        self.send_info_event(
            progress=None,
            message=(
                f"[Load] Ready — "
                f"{data['Article'].x.size(0):,} articles, "
                f"{data['Entity'].x.size(0):,} entities"
            ),
        )

        return data, graph, model, run_config
    
    def inference(self, model: HetModel, data: HeteroData) -> HetModelReturn:

        self.send_info_event(progress=None, message="[Infer] Running full-graph forward pass …")

        model = model.eval().cpu()
        data  = data.cpu()
        model.ssl     = False
        model.cluster = False

        with torch.no_grad():
            results = model.forward(data)

        self.send_info_event(
            progress=None,
            message=(
                f"[Infer] Done — "
                f"entity embeddings: {list(results.entities_view.entities_embeddings.shape)}, "
            ),
        )

        return results

    def get_topics_labels(self, model : HetModel, results : HetModelReturn) -> Dict[str, Tensor]:
        return model.predict(results.C)

    def get_entity_topic_scores(self, 
        data : HeteroData, 
        results : HetModelReturn,
        threshold : float
    ) -> Tensor:

        x_e = data['Entity'].x
        x_d = data['Article'].x_passage
        A2T  = results.articles_view.articles_prototypes

        scores = []

        for topic_assignment in A2T.t():
            x_d_subset = x_d[topic_assignment.gt(threshold),:]
            scores.append(0.5 * torch.mean(x_e @ x_d_subset.T, dim=-1) + 0.5)

        scores = torch.stack(scores, dim=0).t()
        scores[scores.isnan()] = 0.0

        return scores

    def validate_response(self, 
        content : str, 
        model : Type[BaseModel]
    ) -> Dict:
        return model(**json.loads(content)).model_dump()

    def query_topic_name(
        self, 
        topic_id : int,
        graph : Graph,
        threshold : float,
        k : int
    ) -> Dict:

        A2T = graph._nodes['Article']._features['topics_assignments']
        articles_indices = set(torch.argwhere(A2T[:,topic_id].gt(0.5)).flatten().tolist())

        S1 = graph._nodes['Entity']._features['topics_scores']
        S2 = graph._nodes['Entity']._features['topics_assignments'].gt(threshold).float()
        S = S1 * S2
        S = S[:,topic_id]

        row, col = graph.pygeo[('Entity', 'HASRELATIONWITH', 'Entity')].edge_index
        S = S[row] * S[col]
        triples_indices = set(S.topk(k=k).indices.tolist())

        articles_titles = '\n'.join([
            a.properties['title']
            for i, a in enumerate(graph._nodes['Article'].nodes_generator)
            if i in articles_indices
        ])

        salient_triples = '\n'.join([
            f"{t.src} => {t.properties['predicate']} => {t.dst}"
            for i, t in enumerate(graph._edges[EdgeTypes.Entity_HASRELATIONWITH_Entity].edges_generator)
            if i in triples_indices
        ])

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert academic taxonomist. Your task is to generate a precise, professional "
                    "label that synthesizes a researcher's expertise based on a provided list articles titles assigned "
                    "to the area of expertise as well ass a list of salient triples."
                    
                    "### Requirements:\n"
                    "1. **Granularity**: Ensure the label is neither too broad nor too specific. It must "
                    "accurately encapsulate the specific intersection of the provided concepts.\n"
                    "2. **Tone**: Use formal, scientific, and academic terminology.\n"
                    "3. **Format**: Provide a concise noun phrase only. Do not use full sentences or "
                    "lengthy descriptions.\n"
                    "4. **Example**: 'AI-driven environmental risk assessment'."
                    
                    "### Output Format:\n"
                    "The response should only contains the title and should be in json format "
                    "{{ \"title\" : \"[GENERATED TITLE]\" }}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"This expertise is described by the following articles : \n"
                    f"{articles_titles}\n\n"
                    f"And the following salient triples : \n"
                    f"{salient_triples}\n\n"
                )
            }
        ]

        response = self.client.chat.completions.create(
            model=self.config.llm,
            messages=messages,
            stream=False,
            temperature=0,  # No randomness
            top_p=1,  # Disable sampling
            seed=42,  # Seed
        )

        content = self.validate_response(response.choices[0].message.content, model=Topic)

        return content

    def get_topic_names(self, graph : Graph, threshold : float, k : int) -> List[Dict]:

        A2T = graph._nodes['Article']._features['topics_assignments']

        communities = []

        for topic_id, community in enumerate(self.track(
            A2T.t(),
            task="representative-node-querying",
            update_every=1
        )):

            num_articles = community.gt(threshold).float().sum()

            if num_articles < 0:
                continue

            properties = self.query_topic_name(topic_id=topic_id, graph=graph, threshold=threshold, k=k)
            properties['id'] = topic_id

            communities.append(properties)

        return communities

    def update_graph(
        self,
        graph : Graph,
        communities : List[Dict],
        threshold : float,
        topics_domains : Optional[Tensor] = None
    ) -> Graph:

        ### Article => Topic
        ### Article => Domain
        ### Entity => Topic
        ### Entity => Domain
        ### Topic => Domain

        Expertise_BELONGSTO_SubDomain = EdgeType("Expertise", "SubDomain", "BELONGSTO")  ### DONE

        Article_BELONGSTO_SubDomain = EdgeType("Article", "SubDomain", "BELONGSTO")
        Article_BELONGSTO_Expertise = EdgeType("Article", "Expertise", "BELONGSTO")
        
        Entity_BELONGSTO_SubDomain = EdgeType("Entity", "SubDomain", "BELONGSTO")
        Entity_BELONGSTO_Expertise = EdgeType("Entity", "Expertise", "BELONGSTO")
        
        Domain_BELONGSTO_ExpertiseMap = EdgeType("Domain", "ExpertiseMap", "BELONGSTO")

        ### Add community node + community
        for community in self.track(
            communities,
            task="update-graph/adding-experties",
            update_every=16
        ):

            topic_id = community['id']

            graph.add_node(
                type='Expertise', 
                id=community['id'],
                properties={
                    k : v 
                    for k,v in community.items() 
                    if k != 'id'
                }
            )

            if topics_domains is not None:

                for subdomain_id, edge_proba in zip(
                    graph._nodes['SubDomain']._id2idx,
                    topics_domains[topic_id,:]
                ):

                    if edge_proba < threshold:
                        continue

                    graph.add_edge(
                        Expertise_BELONGSTO_SubDomain,
                        src=community['id'],
                        dst=subdomain_id,
                        properties={
                            "score" : edge_proba
                        }
                    )

                graph._nodes['Expertise']._features = topics_domains

        A2T = graph._nodes['Article']._features['topics_assignments']
        E2T = graph._nodes['Entity']._features['topics_assignments']

        ### Article => Topic
        for article, topics_assignment in zip(
            graph.get_nodes('Article').nodes_generator, 
            self.track(
                A2T,
                task="update-graph/articles-expertise-links",
                update_every=16
            )
        ):
            
            for topic_id, topic_proba in enumerate(topics_assignment):

                topic_proba = topic_proba.item()

                if topic_proba < threshold:
                    continue

                graph.add_edge(
                    Article_BELONGSTO_Expertise,
                    src=article.id,
                    dst=topic_id,
                    properties={
                        "score" : topic_proba
                    }
                )

        ### Entity => Topic
        for entity, topics_assignment in zip(
            graph.get_nodes('Entity').nodes_generator, 
            self.track(
                E2T,
                task="update-graph/entities-expertise-links",
                update_every=16
            )
        ):
            
            for topic_id, topic_proba in enumerate(topics_assignment):

                topic_proba = topic_proba.item()

                if topic_proba < threshold:
                    continue

                graph.add_edge(
                    Entity_BELONGSTO_Expertise,
                    src=entity.id,
                    dst=topic_id,
                    properties={
                        "score" : topic_proba
                    }
                )

        A2D = graph._nodes['Article']._features.get('domains_assignments')
        E2D = graph._nodes['Entity']._features.get('domains_assignments')

        ### Article => Domain
        if A2D is not None:

            for article, topics_assignment in zip(
                graph.get_nodes('Article').nodes_generator, 
                self.track(
                    A2D,
                    task="update-graph/entities-domain-links",
                    update_every=16
                )
            ):
                
                for topic_id, topic_proba in zip(graph._nodes['SubDomain']._id2idx, topics_assignment):

                    topic_proba = topic_proba.item()

                    if topic_proba < threshold:
                        continue

                    graph.add_edge(
                        Article_BELONGSTO_SubDomain,
                        src=article.id,
                        dst=topic_id,
                        properties={
                            "score" : topic_proba
                        }
                    )

        ### Entity => Domain
        if E2D is not None:

            for entity, topics_assignment in zip(
                graph.get_nodes('Entity').nodes_generator, 
                self.track(
                    E2D,
                    task="update-graph/entities-domain-links",
                    update_every=16
                )
            ):
                
                for topic_id, topic_proba in zip(graph._nodes['SubDomain']._id2idx, topics_assignment):

                    topic_proba = topic_proba.item()

                    if topic_proba < threshold:
                        continue

                    graph.add_edge(
                        Entity_BELONGSTO_SubDomain,
                        src=entity.id,
                        dst=topic_id,
                        properties={
                            "score" : topic_proba
                        }
                    )

        ### Add root node
        graph.add_node(
            type='ExpertiseMap',
            id=self.config.expertise_id,
            properties={
                'name' : self.config.name,
                'description' : self.config.description
            }
        )

        ### Add Domain <-> Root Node relations
        for domain in graph.get_nodes('Domain'):

            graph.add_edge(
                type=Domain_BELONGSTO_ExpertiseMap,
                src=domain.id,
                dst=self.config.expertise_id
            )

        return graph
    
    def __iter__(self):

        ### Load models & data
        data, graph, model, run_config = self.load_data_and_model()

        ### Inference & Update Graph with new properties
        inferece_results = self.inference(model, data)

        if (inferece_results.articles_view.articles_predictions is not None and
            'llm:deepseek-chat' in inferece_results.articles_view.articles_predictions
        ):
            graph._nodes['Article']._features['domains_assignments'] = (
                inferece_results
                    .articles_view
                    .articles_predictions
                    .get('llm:deepseek-chat')
                    .sigmoid()
            )

        if (inferece_results.entities_view.entities_predictions is not None and
            'llm:deepseek-chat' in inferece_results.entities_view.entities_predictions
        ):
            graph._nodes['Entity']._features['domains_assignments'] = (
                inferece_results
                    .entities_view
                    .entities_predictions
                    .get('llm:deepseek-chat')
                    .sigmoid()
            )

        graph._nodes['Article']._features['topics_assignments'] = (
            inferece_results
                .articles_view
                .articles_prototypes
                .sigmoid()
        )

        graph._nodes['Entity']._features['topics_assignments'] = (
            inferece_results
                .entities_view
                .entities_prototypes
                .sigmoid()
        )

        graph._nodes['Entity']._features['topics_scores'] = self.get_entity_topic_scores(
            data=data,
            results=inferece_results,
            threshold=self.config.threshold
        )  

        for key, feat in graph._nodes['Article']._features.items():
            print(f"Article.{key} => {feat.size()}")

        for key, feat in graph._nodes['Entity']._features.items():
            print(f"Entity.{key} => {feat.size()}")

        topics_labels = None

        if len(run_config.labels):
            topics_labels =  self.get_topics_labels(model, inferece_results).get('llm:deepseek-chat')
            print(f"topics_labels.shape = {topics_labels.shape}")

        communities = self.get_topic_names(graph=graph,threshold=self.config.threshold,k=self.config.k)

        print(f"len(communities) = {len(communities)}")
        print(f"communities[:5] = {communities[:5]}")

        ### Save update graph with new node(s) and relationship(s)
        graph = self.update_graph(
            graph=graph,
            communities=communities,
            threshold=self.config.threshold,
            topics_domains=topics_labels
        )

        ### Save Graph to disk
        graph.save(self.config.expertise_dir)

        yield self.make_debug_event(
            progress=None,
            message=f"Expertise map saved to path : {self.config.expertise_dir}"
        )

    @classmethod
    def from_config(cls, config: Config) -> 'ExpertiseMapBuilder':
        builder = ExpertiseMapBuilder(config=config.conf, client=env.llms[config.conf.llm].client)
        builder.add_handler(ConsoleLogger("ExpertiseMapBuilder"))
        return builder