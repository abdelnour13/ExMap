import os
import torch
import networkx as nx
from typing import Dict, List, Literal, Tuple
from pydantic import BaseModel, computed_field
from sknetwork.clustering import Leiden, Louvain
from sknetwork.data import from_edge_list
from torch import Tensor
from definitions import EXPERIMENTS_DIR, EMS_DIR
from instance import cli
from torch_geometric.data import HeteroData
from src.commands.feature_extraction.graph_utils import EdgeType, Graph
from src.commands.graph_learning.main import Config as TrainerConfig
from src.graph_dl.models import Context
from src.graph_dl.models.classifier import ArticleClassifier
from src.shared.command_tools import BaseCommand
from src.shared.file_tools import Batched, Map, Enumerate
from src.shared.handlers import ConsoleLogger
from src.shared.platforms import LLM, RequestReturnType
from src import helpers

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def entropy(en : Tensor) -> Tensor:
    return -torch.sum(en * torch.log(en + 10e-5), dim=-1).clip(max=0)

# ---------------------------------------------------------------------------communities
# Configuration
# ---------------------------------------------------------------------------

MASK = torch.tensor([ 
    True,  True, False, False,  True,  True,  True,  True, False,
    True, False,  True, False, False, False,  True,  True, False,
    False,  True, False, False, False, False,  True,  True,  True,
    False, False,  True,  True,  True, False, False, False, False,
    True, False, False,  True, False,  True, False,  True,  True,
    False,  True,  True, False, False, False, False, False,  True,
    True, False, False, False, False,  True,  True,  True, False,
    False,  True
])

class BEMConfig(BaseModel):

    # ── Graph & Model ──────────────────────────────────────────────────
    expertise_id : str
    model_id : str

    # ── Clustering ─────────────────────────────────────────────────────
    clustering_alg:        str      = "leiden" # Leiden or Louvain
    clustering_resolution: float    = 15.0
    jaccard_mode:          str      = "hard"   # soft or hard
    jaccard_threshold:     float    = 0.5

    # ── Community cleaning ─────────────────────────────────────────────
    max_labels:           int   = 3      # discard entities with > N subdomain labels
    max_degree:           int   = 50     # discard entities with degree > N
    max_generality_score: float = 0.75   # discard entities above this generality
    max_relevancy_score:  float = 0.75   # discard entities above this relevancy
    min_nodes_per_community : int = 10   # The Minimum number of nodes / community

    # ── Representative node selection ──────────────────────────────────
    top_k_candidates:     int   = 50    # max candidates kept per community after ranking
    label_threshold:      float = 0.5   # min probability for a subdomain / lab label to be active

    # ── Representative node selection ──────────────────────────────────
    llm : str = "deepseek-chat" # LLM used for llm-based labeling
    guided : bool = True

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

    def __init__(self, config: BEMConfig, llm : LLM) -> None:
        super().__init__()
        self.config = config
        self.llm = llm

    def load_data_and_model(self) -> Tuple[HeteroData, Graph, ArticleClassifier]:

        self.send_info_event(
            progress=None,
            message="[Load] Starting — loading trainer config and data",
        )

        main_config = TrainerConfig(experiment=self.config.experiment).training_config
        configs = main_config.configs

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
        data , graph = run_config.load_data(return_raw=True)

        # model
        model = run_config.load_model(ctx=Context(data), postfix='final')

        self.send_info_event(
            progress=None,
            message=(
                f"[Load] Ready — "
                f"{data['Article'].x.size(0):,} articles, "
                f"{data['Entity'].x.size(0):,} entities"
            ),
        )

        return data, graph, model
    
    def inference(self, model: ArticleClassifier, data: HeteroData) -> ArticleClassifier.Output:

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
                f"entity embeddings: {list(results.entities_sem_embeddings.shape)}, "
            ),
        )

        return results
    
    @staticmethod
    def jaccard_index(
        edge_index: Tensor,
        probas:     Tensor,
        mode:       Literal["soft", "hard"] = "soft",
        threshold:  float = 0.5,
    ) -> Tensor:
        
        if mode == "hard":
            probas = probas.gt(threshold).float()

        row, col     = edge_index
        a            = torch.stack([probas[row], probas[col]], dim=0)
        intersection = a.min(0).values.sum(-1)
        union        = a.max(0).values.sum(-1)
        jacc         = intersection / union

        return torch.where(jacc.isnan(), torch.zeros_like(jacc), jacc)

    def domains(self, labels : Dict[str, Tensor]) -> Tensor:
        return labels['llm:deepseek-chat']

    def labs(self, labels : Dict[str, Tensor]) -> Tensor:
        return labels['lab']
    
    def cluster(
        self,
        model : ArticleClassifier,
        inference_results : ArticleClassifier.Output,
        edge_index : Tensor,
    ) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:

        self.send_info_event(
            progress=None,
            message=(
                f"[Cluster] Running {self.config.clustering_alg} "
                f"(resolution={self.config.clustering_resolution}) …"
            ),
        )

        entity_label_probas = self.domains(inference_results.entities_labels).sigmoid()

        edge_weight = self.jaccard_index(
            edge_index, entity_label_probas[:,MASK],
            mode=self.config.jaccard_mode,
            threshold=self.config.jaccard_threshold,
        )

        edge_list = list(zip(
            edge_index[0].tolist(),
            edge_index[1].tolist(),
            edge_weight.tolist(),
        ))

        adj = from_edge_list(edge_list)

        alg_cls = Leiden if self.config.clustering_alg == "leiden" else Louvain
        alg  = alg_cls(resolution=self.config.clustering_resolution)
        communities = torch.from_numpy(alg.fit_predict(adj))

        communities_probas = torch.from_numpy(alg.fit_predict_proba(
            from_edge_list(
                list(
                    zip(
                        edge_index[0].tolist(), 
                        edge_index[1].tolist()
                    )
                )
            )
        ))

        if not self.config.guided:
            communities = communities_probas.argmax(dim=-1)

        n_communities = int(communities.max().item() + 1)

        self.send_info_event(
            progress=None,
            message=f"[Cluster] Found {n_communities} communities",
        )

        # community-level domains
        
        with torch.no_grad():

            x_sem = inference_results.entities_sem_embeddings
            x_imp = inference_results.entities_importance_embeddings

            community_embeddings, _, _ = model.reduce.forward(
                x = x_sem,
                edge_index=torch.stack([
                    torch.arange(x_sem.size(0)).to(x_sem.device),
                    communities.to(x_sem.device)
                ]),
                num_nodes=n_communities,
                weights_x=x_imp
            ).astuple()

            communities_labels = model.head.forward(community_embeddings)

        self.send_debug_event(
            progress=None,
            message=(
                f"[Cluster] Community label probas shape: "
                f"{list(self.domains(communities_labels).shape)} {list(self.labs(communities_labels).shape)}"
            ),
        )

        return communities, communities_probas, communities_labels
    
    def clean_community(self, community : Dict) -> Dict:

        community = {**community}

        community['entities'] = [
            entity 
            for entity in community['entities']
            if (
                (entity["degree"]           <= self.config.max_degree)           &
                (entity["generality_scores"] <= self.config.max_generality_score) &
                (entity["relevancy_scores"]  <= self.config.max_relevancy_score)
            )
        ]

        return community

    def process_communities(
        self, 
        graph : Graph,
        data : HeteroData,
        communities : Tensor,
        community_labels : Dict[str, Tensor]
    ) -> List[Dict]:
        
        n_communities = communities.max().item() + 1

        communities_nodelist = [
            { 
                "id" : i, 
                "entities" : [], 
                "labels" : [], 
                "labs" : [] 
            } 
            for i in range(n_communities)
        ]

        for entity, community in zip(graph.get_nodes('Entity'), communities):

            communities_nodelist[community.item()]['entities'].append({
                'id' : entity.id,
                **{
                    k : v
                    for k, v in graph.get_node('Entity', entity.id).properties.items()
                    if not isinstance(v, Tensor)
                }
            })
        
        for community, labels in enumerate(self.domains(community_labels)):

            for label_name, label_proba in zip(data.raw_metadata['labels']['subdomain'], labels):

                if label_proba >= self.config.label_threshold:
                    
                    communities_nodelist[community]['labels'].append((
                        label_name,
                        label_proba.item()
                    ))

        for community, labels in enumerate(community_labels['lab']):

            for label_name, label_proba in zip(data.raw_metadata['labels']['lab'], labels):

                if label_proba >= self.config.label_threshold:
                    
                    communities_nodelist[community]['labs'].append((
                        label_name,
                        label_proba.item()
                    ))
        
        communities_nodelist = [
            self.clean_community(community)
            for community in communities_nodelist
            if len(community['entities']) >= self.config.min_nodes_per_community
        ]

        it = Map(communities_nodelist, self._task_info)
        it = Enumerate(it)
        it = Batched(iterable=it, n=16)
        it = self.track(it=it, task="representative-node-querying")

        for batch in it:

            inputs = [item for _, item in batch]
            indices = [i for i,_ in batch]

            outputs = self.query_representative_node(inputs)

            for i, out in zip(indices, outputs):
                communities_nodelist[i]['llm_label'] = out

        self.send_info_event(
            progress=None,
            message=f"Number of communities after processing : {len(communities_nodelist)} "
        )

        return communities_nodelist
    
    def _select_representative_node(self, community : Dict, attr : str) -> str:
        return sorted(community['entities'], key=lambda e : e[attr], reverse=True)[0]['name']

    def _task_info(self, community : Dict) -> Dict:

        community = {**community}
        
        for attr in ['specificity_weighted_pagerank', 'personalized_pagerank']:
            community[f'{attr}_name'] = self._select_representative_node(community, attr)

        sorted_entities = sorted(community['entities'], key=lambda e : e['personalized_pagerank'], reverse=True)
        sorted_entities = [e['name'] for e in sorted_entities[:min(len(sorted_entities), 50)]]

        return { "concepts" : sorted_entities }

    def query_representative_node(self, inputs : List[Dict]) -> List[str]:

        outputs = self.llm.batch_make_async_request(
            prompt_name='ResearcherExpertiseNaming',
            inputs=inputs,
        )

        return [
            out.unwrap()["label"] 
            for out in outputs
        ]
    
    def update_graph(self,
        graph : Graph,
        data : HeteroData,
        communities_nodelist : List[Dict],
        results : ArticleClassifier.Output
    ) -> Graph:

        Expertise_BELONGSTO_SubDomain = EdgeType("Expertise", "SubDomain", "BELONGSTO")
        Expertise_BELONGSTO_Lab = EdgeType("Expertise", "Lab", "BELONGSTO")

        Entity_BELONGSTO_SubDomain = EdgeType("Entity", "SubDomain", "BELONGSTO")
        Entity_BELONGSTO_Lab = EdgeType("Entity", "Lab", "BELONGSTO")
        Entity_BELONGSTO_Expertise = EdgeType("Entity", "Expertise", "BELONGSTO")

        Domain_BELONGSTO_ExpertiseMap = EdgeType("Domain", "ExpertiseMap", "BELONGSTO")

        for node in graph._nodes['SubDomain']:
            
            graph.add_node(
                type='SubDomain',
                id=node.id, 
                properties={
                    **node.properties,
                }
            )
        
        ### Community Node & Community <-> SubDomain links & Community <-> Entity Links
        for community in self.track(
            communities_nodelist,
            task="update-graph/adding-experties",
            update_every=16
        ):

            graph.add_node(
                type='Expertise', 
                id=community['id'],
                properties={
                    "llm_label" : community.get('llm_label'),
                    **{
                        k : v
                        for k, v in community.items()
                        if k.endswith('_name')
                    }
                }
            )

            for subdomain_id, edge_proba in community['labels']:

                graph.add_edge(
                    Expertise_BELONGSTO_SubDomain,
                    src=community['id'],
                    dst=subdomain_id,
                    properties={
                        "score" : edge_proba
                    }
                )

            for lab_id, edge_proba in community['labs']:

                graph.add_edge(
                    Expertise_BELONGSTO_Lab,
                    src=community['id'],
                    dst=lab_id,
                    properties={
                        "score" : edge_proba
                    }
                )

            for entity in community['entities']:

                graph.add_edge(
                    Entity_BELONGSTO_Expertise,
                    src=entity['id'],
                    dst=community['id']
                )

        ### Add Entity <-> SubDomain links
        for entity, subdomain_probas, lab_probas in zip(
            self.track(
                graph.get_nodes('Entity'), 
                task="update-graph/adding-entities-relations", 
                update_every=512
            ),
            self.domains(results.entities_labels), 
            self.labs(results.entities_labels)
        ):

            for subdomain_id, proba in zip(data.raw_metadata['labels']['subdomain'], subdomain_probas):

                proba = proba.sigmoid().item()

                if proba >= self.config.label_threshold:

                    graph.add_edge(
                        type=Entity_BELONGSTO_SubDomain,
                        src=entity.id,
                        dst=subdomain_id,
                        properties={
                            "score" : proba
                        }
                    )

                for lab_id, proba in zip(data.raw_metadata['labels']['lab'], lab_probas):

                    proba = proba.sigmoid().item()

                    if proba >= self.config.label_threshold:

                        graph.add_edge(
                            type=Entity_BELONGSTO_Lab,
                            src=entity.id,
                            dst=lab_id,
                            properties={
                                "score" : proba
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

        for sub_domain, is_valid in zip(graph.get_nodes('SubDomain'), MASK):
        
            graph.add_node(
                type="SubDomain",
                id=sub_domain.id,
                properties={
                    **sub_domain.properties,
                    "is_valid" : is_valid.item()
                }
            )

        return graph
    
    def _personalized_pagerank(self, graph : Graph) -> Tensor:

        row, col = graph.pygeo['Entity', 'HASRELATIONWITH', 'Entity'].edge_index

        G = nx.DiGraph()

        for src, dst in zip(row, col):
            G.add_edge(src.item(), dst.item())

        G = G.to_undirected()

        scores = torch.pow(
            (1 - graph._nodes['Entity']._features['normalized_entropy']) *
            (1 - graph._nodes['Entity']._features['relevancy_scores']) * 
            (1 - graph._nodes['Entity']._features['generality_scores']),
            1 / 3
        )

        scores = scores / scores.sum()

        pr = nx.pagerank(
            G,
            personalization=dict([
                (i, score.item()) for i, score in enumerate(scores)
            ]),
            max_iter=1000
        )

        return torch.tensor(list(pr.values()))
    
    def __iter__(self):

        ### Load models & data
        data, graph, model = self.load_data_and_model()

        ### Inference & Update Graph with new properties
        inferece_results = self.inference(model, data)

        graph._nodes['Entity']._features = {
            **{
                k : v.sigmoid()
                for k, v in inferece_results.entities_labels.items()
            },
            **graph._nodes['Entity']._features
        }

        ### Community detection & update graph with community / expertise id
        communities, communities_probas, community_labels = self.cluster(
            model,
            inferece_results, 
            data['Entity', 'HASRELATIONWITH', 'Entity'].edge_index
        )

        graph._nodes['Entity']._features['expertise_id'] = communities    
        graph._nodes['Entity']._features['community_proba'] = communities_probas  
        graph._nodes['Entity']._features['entropy'] = entropy(communities_probas) 
        graph._nodes['Entity']._features['normalized_entropy'] = 2 * torch.sigmoid(entropy(communities_probas)) - 1
        graph._nodes['Entity']._features['pagerank_entropy'] = (
            graph._nodes['Entity']._features['normalized_entropy'] * 
            graph._nodes['Entity']._features['pagerank']
        )

        graph._nodes['Entity']._features['personalized_pagerank'] = self._personalized_pagerank(graph)

        ### Clean communities, entities ranking & representative entities selection
        communities_nodelist = self.process_communities(
            graph,
            data,
            communities,
            community_labels
        )

        helpers.save_json_file(os.path.join(self.config.expertise_dir, 'communities.json'), communities_nodelist)

        ### Save update graph with new node(s) and relationship(s)
        graph = self.update_graph(
            graph,
            data,
            communities_nodelist,
            inferece_results
        )

        ### Save Graph to disk
        graph.save(self.config.expertise_dir)

        yield self.make_debug_event(
            progress=None,
            message=f"Expertise map saved to path : {self.config.expertise_dir}"
        )

    @classmethod
    def from_config(cls, config: Config) -> 'ExpertiseMapBuilder':
        builder = ExpertiseMapBuilder(config=config.conf, llm=LLM(config.conf.llm))
        builder.add_handler(ConsoleLogger("ExpertiseMapBuilder"))
        return builder