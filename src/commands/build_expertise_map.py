import os
import torch
import dotenv
from rich.progress import Progress
from src.shared.llms import LLM
from openai import Client
from typing import Any, Dict, List, Literal, Tuple, Optional
from pydantic import BaseModel, computed_field
from sknetwork.clustering import Leiden, Louvain
from sknetwork.data import from_edge_list
from torch import Tensor
from definitions import EXPERIMENTS_DIR, FEATURES_DIR, EMS_DIR, ENV
from instance import cli
from functools import partial
from src.commands.train import Trainer, Config as TrainerConfig
from src.gnn import ArticleAsKG, HetModel, HetModelReturn
from src.shared.command_tools import BaseCommand
from src.shared.handlers import ConsoleLogger
from src import helpers

dotenv.load_dotenv(ENV)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class Config(BaseModel):

    # ── Graph & Model ──────────────────────────────────────────────────
    expertise_id : Optional[str] = None
    graph_id : str = "nantes-uni-2020/all-MiniLM-L6-v2"
    domains_model_id : str = "nantes-uni-2020/Conv_Attn_null_False"
    labs_model_id : str = "nantes-uni-2020_labs/Conv_Attn_null_False"

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
    label_threshold:      float = 0.5   # min probability for a subdomain label to be active

    @computed_field
    @property
    def exprtise_dir(self) -> str:
        return os.path.join(EMS_DIR, self.expertise_id or self.domains_model_id.replace('/','--'))

    @computed_field
    @property
    def graph_dir(self) -> str:
        return os.path.join(FEATURES_DIR, *self.graph_id.split('/'))
    
    @computed_field
    @property
    def features_path(self) -> str:
        return os.path.join(self.graph_dir, 'graph.pt')
    
    @computed_field
    @property
    def metadata_path(self) -> str:
        return os.path.join(self.graph_dir, "metadata.json")
    
    @computed_field
    @property
    def domains_model_path(self) -> str:
        experiment, model_name = self.domains_model_id.split('/')
        return os.path.join(EXPERIMENTS_DIR, experiment, 'models', f'model_{model_name}.pt')
    
    @computed_field
    @property
    def domains_experiment(self) -> str:
        return self.domains_model_id.split('/')[0]
    
    @computed_field
    @property
    def domains_model(self) -> str:
        return self.domains_model_id.split('/')[1]
    
    @computed_field
    @property
    def labs_model_path(self) -> str:
        experiment, model_name = self.labs_model_id.split('/')
        return os.path.join(EXPERIMENTS_DIR, experiment, 'models', f'model_{model_name}.pt')
    
    @computed_field
    @property
    def labs_experiment(self) -> str:
        return self.labs_model_id.split('/')[0]
    
    @computed_field
    @property
    def labs_model(self) -> str:
        return self.labs_model_id.split('/')[1]
    
    llm : str = "deepseek-chat"

    @computed_field
    @property
    def api_key(self) -> str:
        return dotenv.get_key(ENV, f"{self.llm.upper()}_API_KEY")
    

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

    def __init__(self, config: Config, client : Client) -> None:
        super().__init__()
        self.config = config
        self.client = client

    def load_data_and_model(self) -> Tuple[ArticleAsKG, HetModel, HetModel, Dict[str, Dict[str, List[Dict]]]]:

        self.send_info_event(
            progress=0.0,
            message="[Load] Starting — loading trainer config and data",
        )

        ### Domains
        domains_trainer_cfg = TrainerConfig(experiment=self.config.domains_experiment)
        domains_trainer     = Trainer(experiment=domains_trainer_cfg.experiment, config=domains_trainer_cfg.training_config)
        domains_configs     = domains_trainer.get_configs()

        if self.config.domains_model not in domains_configs:

            raise KeyError(
                f"[Load] Model config '{self.config.domains_model}' not found. "
                f"Available: {list(domains_configs.keys())}"
            )

        domains_run_config = domains_configs[self.config.domains_model]

        ### Labs
        labs_trainer_cfg = TrainerConfig(experiment=self.config.labs_experiment)
        labs_trainer     = Trainer(experiment=labs_trainer_cfg.experiment, config=labs_trainer_cfg.training_config)
        labs_configs     = labs_trainer.get_configs()

        if self.config.labs_model not in labs_configs:

            raise KeyError(
                f"[Load] Model config '{self.config.labs_model}' not found. "
                f"Available: {list(labs_configs.keys())}"
            )

        labs_run_config = labs_configs[self.config.labs_model]

        self.send_debug_event(
            progress=0.0,
            message=f"[Load] Using run config '{self.config.labs_model}': {domains_run_config}",
        )

        # Load data through the trainer (handles normalisation)
        domains_data = domains_trainer.load_data()
        labs_data = labs_trainer.load_data()

        # Also keep the raw dict for augmentation later
        self.send_debug_event(
            progress=0.0,
            message=f"[Load] Reading raw graph dict from {self.config.metadata_path}",
        )

        raw_G: Dict[str, Any] = helpers.load_json_file(self.config.metadata_path)

        # Load model
        self.send_debug_event(
            progress=0.0,
            message=f"[Load] Loading model weights from {self.config.domains_model_path}",
        )

        # Domains model
        domains_model = domains_trainer.create_model(config=domains_run_config, data=domains_data)
        state_dict = torch.load(self.config.domains_model_path, weights_only=False, map_location="cpu")
        domains_model.load_state_dict(state_dict)

        # Labs model
        labs_model = labs_trainer.create_model(config=labs_run_config, data=labs_data)
        state_dict = torch.load(self.config.labs_model_path, weights_only=False, map_location="cpu")
        labs_model.load_state_dict(state_dict)

        self.send_info_event(
            progress=0.0,
            message=(
                f"[Load] Ready — "
                f"{domains_data.articles.num_nodes:,} articles, "
                f"{domains_data.kg.num_nodes:,} entities"
            ),
        )

        return domains_data, domains_model, labs_model, raw_G
    
    def inference(self, model: HetModel, data: ArticleAsKG) -> HetModelReturn:

        self.send_info_event(progress=0.0, message="[Infer] Running full-graph forward pass …")

        model = model.eval().cpu()
        data  = data.cpu()
        model.ssl     = False
        model.cluster = False

        with torch.no_grad():
            results = model.forward(data)

        self.send_info_event(
            progress=0.0,
            message=(
                f"[Infer] Done — "
                f"entity embeddings: {list(results.entities_embeddings.shape)}, "
                f"entity predictions: {list(results.entities_predictions.shape)}"
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
    
    def cluster(
        self,
        domains_model : HetModel,
        labs_model : HetModel,
        domains_inference_results : HetModelReturn,
        labs_inference_results : HetModelReturn,
        edge_index : Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:

        self.send_info_event(
            progress=0.0,
            message=(
                f"[Cluster] Running {self.config.clustering_alg} "
                f"(resolution={self.config.clustering_resolution}) …"
            ),
        )

        entity_label_probas = domains_inference_results.entities_predictions.sigmoid()

        edge_weight = self.jaccard_index(
            edge_index, entity_label_probas,
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
        communities = torch.from_numpy(alg.fit_predict(adj)).to(domains_inference_results.entities_embeddings.device)

        n_communities = int(communities.max().item() + 1)

        self.send_info_event(
            progress=0.0,
            message=f"[Cluster] Found {n_communities} communities",
        )

        # community-level domains
        with torch.no_grad():

            community_embeddings = domains_model.readout_fn.forward(
                x = domains_inference_results.entities_embeddings,
                index = communities,
                dim_size = n_communities,
                dim = 0
            )

            communities_domains_probas = domains_model.head(community_embeddings).sigmoid()

        # community-level labs
        with torch.no_grad():

            community_embeddings = labs_model.readout_fn.forward(
                x = labs_inference_results.entities_embeddings,
                index = communities,
                dim_size = n_communities,
                dim = 0
            )

            communities_labs_probas = labs_model.head(community_embeddings).sigmoid()

        self.send_debug_event(
            progress=0.0,
            message=(
                f"[Cluster] Community label probas shape: "
                f"{list(communities_domains_probas.shape)} {communities_labs_probas.shape}"
            ),
        )

        return communities, communities_domains_probas, communities_labs_probas
    
    def clean_community(self, community : Dict, id2Entity : Dict[str, Dict]) -> Dict:

        community = {**community}

        community['entities'] = [
            e_id 
            for e_id in community['entities']
            if (
                (id2Entity[e_id]["degree"]           <= self.config.max_degree)           &
                (id2Entity[e_id]["generality_score"] <= self.config.max_generality_score) &
                (id2Entity[e_id]["relevancy_score"]  <= self.config.max_relevancy_score)
            )
        ]

        return community

    def process_communities(
        self, 
        id2Entity : Dict[str, Dict],
        data : ArticleAsKG,
        communities : Tensor,
        communities_domains : Tensor,
        communities_labs : Tensor
    ) -> Dict:
        
        n_communities = communities.max().item() + 1
        communities_nodelist = [{ "id" : i, "entities" : [], "labels" : [], "labs" : [] } for i in range(n_communities)]

        for entity_id, community in zip(data.kg.entity_names, communities):
            communities_nodelist[community.item()]['entities'].append(entity_id)
        
        for community, labels in enumerate(communities_domains):

            for label_name, label_proba in zip(data.metadata['labels']['domains'], labels):

                if label_proba >= self.config.label_threshold:
                    
                    communities_nodelist[community]['labels'].append((
                        label_name,
                        label_proba.item()
                    ))

        for community, labels in enumerate(communities_labs):

            for label_name, label_proba in zip(data.metadata['labels']['labs'], labels):

                if label_proba >= self.config.label_threshold:
                    
                    communities_nodelist[community]['labs'].append((
                        label_name,
                        label_proba.item()
                    ))
        
        communities_nodelist = [
            self.clean_community(community, id2Entity)
            for community in communities_nodelist
            if len(community['entities']) >= self.config.min_nodes_per_community
        ]

        pbar = Progress()
        tid = pbar.add_task("Communities Processing", total=len(communities_nodelist))
        pbar.start()

        for i,community in enumerate(communities_nodelist):
            communities_nodelist[i] = self.query_representative_node(community, id2Entity)
            pbar.update(tid, advance=1)

        self.send_info_event(
            progress=0.0,
            message=f"Number of communities after processing : {len(communities_nodelist)} ",
        )

        return communities_nodelist

    def query_representative_node(self, community : Dict, id2Entity : Dict[str, Dict]) -> Dict:

        community = {**community}

        sorted_entities = sorted(community['entities'], key=lambda eid : id2Entity[eid]['specificity_weighted_pagerank'], reverse=True)
        community['name'] = id2Entity[sorted_entities[0]]['name']

        sorted_entities = [id2Entity[eid]['name'] for eid in sorted_entities[:min(len(sorted_entities), 20)]]

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert academic taxonomist. Your task is to generate a precise, professional "
                    "label that synthesizes a researcher's expertise based on a provided list of up to 20 concepts. "
                    
                    "### Requirements:\n"
                    "1. **Granularity**: Ensure the label is neither too broad nor too specific. It must "
                    "accurately encapsulate the specific intersection of the provided concepts.\n"
                    "2. **Tone**: Use formal, scientific, and academic terminology.\n"
                    "3. **Format**: Provide a concise noun phrase only. Do not use full sentences or "
                    "lengthy descriptions.\n"
                    "4. **Example**: 'AI-driven environmental risk assessment'."
                    
                    "### Output Format:\n"
                    "The response should only contain the number that coresponds to the level with no additional explanation."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"This expertise is described by the following entities : \n"
                    "\n".join(sorted_entities)
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

        content = response.choices[0].message.content
        community['llm_label'] = content.strip()

        return community
    
    def update_graph(self,
        raw_G : Dict[str, Dict[str, List[Dict]]],
        data : ArticleAsKG,
        communities_nodelist : List[Dict],
        domains_results : HetModelReturn,
        labs_results : HetModelReturn
    ) -> Dict[str, Dict[str, List[Dict]]]:
        
        raw_G['nodes']['Expertise'] = []
        raw_G['edges']['Expertise_BelongsTo_SubDomain'] = []
        raw_G['edges']['Expertise_BelongsTo_Lab'] = []
        raw_G['edges']['Entity_BelongsTo_Expertise'] = []
        raw_G['edges']['Entity_BelongsTo_SubDomain'] = []
        raw_G['edges']['Entity_BelongsTo_Lab'] = []
        
        ### Community Node & Community <-> SubDomain links & Community <-> Entity Links
        for community in communities_nodelist:

            raw_G['nodes']['Expertise'].append({
                "id" : community['id'],
                "name" : community['name'],
                "llm_label" : community['llm_label'],
                "type" : "Expertise"
            })

            for subdomain_id, edge_proba in community['labels']:

                raw_G['edges']['Expertise_BelongsTo_SubDomain'].append({
                    "id": f"{community['id']}-{subdomain_id}",
                    "src": community['id'],
                    "dst": subdomain_id,
                    "score": edge_proba
                })

            for lab_id, edge_proba in community['labs']:

                raw_G['edges']['Expertise_BelongsTo_Lab'].append({
                    "id": f"{community['id']}-{lab_id}",
                    "src": community['id'],
                    "dst": lab_id,
                    "score": edge_proba
                })

            for entity in community['entities']:

                raw_G['edges']['Entity_BelongsTo_Expertise'].append({
                    "id": f"{entity}-{community['id']}",
                    "src": entity,
                    "dst": community['id'],
                })

        ### Add Entity <-> SubDomain links
        for entity, probas in zip(data.kg.entity_names, domains_results.entities_predictions):

            for subdomain_id, proba in zip(data.metadata['labels']['domains'], probas):

                proba = proba.sigmoid().item()

                if proba >= self.config.label_threshold:

                    raw_G['edges']['Entity_BelongsTo_SubDomain'].append({
                        "id": f"{entity}-{subdomain_id}",
                        "src": entity,
                        "dst": subdomain_id,
                        "score" : proba
                    })

        ### Add Entity <-> Lab links
        for entity, probas in zip(data.kg.entity_names, labs_results.entities_predictions):

            for lab_id, proba in zip(data.metadata['labels']['labs'], probas):

                proba = proba.sigmoid().item()

                if proba >= self.config.label_threshold:

                    raw_G['edges']['Entity_BelongsTo_Lab'].append({
                        "id": f"{entity}-{lab_id}",
                        "src": entity,
                        "dst": lab_id,
                        "score" : proba
                    })

        return raw_G
    
    def __iter__(self):

        ### Load models & data
        data, domains_model, labs_model, raw_G = self.load_data_and_model()

        ### Inference
        domains_inference_results  = self.inference(domains_model, data)
        labs_inference_results  = self.inference(labs_model, data)

        ### Community detection
        communities, communities_domains, communities_labs = self.cluster(
            domains_model,
            labs_model, 
            domains_inference_results, 
            labs_inference_results,
            data.kg.edge_index
        )

        id2Entity = dict([(e['id'], e) for e in raw_G['nodes']['Entity']])
        

        ### Clean communities, entities ranking & representative entities selection
        communities_nodelist = self.process_communities(
            id2Entity,
            data,
            communities,
            communities_domains,
            communities_labs
        )

        raw_G = self.update_graph(
            raw_G,
            data,
            communities_nodelist,
            domains_inference_results,
            labs_inference_results
        )

        os.makedirs(self.config.exprtise_dir, exist_ok=True)
        data_path = os.path.join(self.config.exprtise_dir, 'dump.json')
        helpers.save_json_file(data_path, raw_G)

        yield self.make_debug_event(
            progress=None,
            message=f"Expertise map saved to path : {data_path}"
        )

    @classmethod
    def from_config(cls, config: Config) -> 'ExpertiseMapBuilder':

        llm = LLM.from_name(config.llm)

        client = Client(
            api_key=config.api_key,
            base_url=llm.url
        )

        builder = cls(config=config, client=client)
        builder.add_handler(ConsoleLogger("ExpertiseMapBuilder"))
        return builder