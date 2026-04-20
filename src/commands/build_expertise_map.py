import os
import torch
from typing import Any, Dict, List, Literal, Tuple, Optional
from pydantic import BaseModel, computed_field
from sknetwork.clustering import Leiden, Louvain
from sknetwork.data import from_edge_list
from torch import Tensor
from definitions import EXPERIMENTS_DIR, FEATURES_DIR, EMS_DIR
from instance import cli
from functools import partial
from src.commands.train import Trainer, Config as TrainerConfig
from src.gnn import ArticleAsKG, HetModel, HetModelReturn
from src.shared.command_tools import BaseCommand
from src.shared.handlers import ConsoleLogger
from src import helpers

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class Config(BaseModel):

    # ── Graph & Model ──────────────────────────────────────────────────
    expertise_id : Optional[str] = None
    graph_id : str = "nantes-uni-2020/all-MiniLM-L6-v2"
    model_id : str = "nantes-uni-2020/Conv_Attn_null_False"

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
        return os.path.join(EMS_DIR, self.expertise_id or self.model_id.replace('/','--'))

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
    def model_path(self) -> str:
        experiment, model_name = self.model_id.split('/')
        return os.path.join(EXPERIMENTS_DIR, experiment, 'models', f'model_{model_name}.pt')
    
    @computed_field
    @property
    def experiment(self) -> str:
        return self.model_id.split('/')[0]
    
    @computed_field
    @property
    def model(self) -> str:
        return self.model_id.split('/')[1]
    

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

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config

    def load_data_and_model(self) -> Tuple[ArticleAsKG, HetModel, Dict[str, Dict[str, List[Dict]]]]:

        self.send_info_event(
            progress=0.0,
            message="[Load] Starting — loading trainer config and data",
        )

        trainer_cfg = TrainerConfig(experiment=self.config.experiment)
        trainer     = Trainer(experiment=trainer_cfg.experiment, config=trainer_cfg.training_config)
        configs     = trainer.get_configs()

        if self.config.model not in configs:

            raise KeyError(
                f"[Load] Model config '{self.config.model}' not found. "
                f"Available: {list(configs.keys())}"
            )

        run_config = configs[self.config.model]

        self.send_debug_event(
            progress=0.0,
            message=f"[Load] Using run config '{self.config.model}': {run_config}",
        )

        # Load data through the trainer (handles normalisation)
        data = trainer.load_data()

        # Also keep the raw dict for augmentation later
        self.send_debug_event(
            progress=0.0,
            message=f"[Load] Reading raw graph dict from {self.config.metadata_path}",
        )

        raw_G: Dict[str, Any] = helpers.load_json_file(self.config.metadata_path)

        # Load model
        self.send_debug_event(
            progress=0.0,
            message=f"[Load] Loading model weights from {self.config.model_path}",
        )

        model = trainer.create_model(config=run_config, data=data)
        state_dict = torch.load(self.config.model_path, weights_only=False, map_location="cpu")
        model.load_state_dict(state_dict)

        self.send_info_event(
            progress=0.0,
            message=(
                f"[Load] Ready — "
                f"{data.articles.num_nodes:,} articles, "
                f"{data.kg.num_nodes:,} entities"
            ),
        )

        return data, model, raw_G
    
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
        model : HetModel,
        results : HetModelReturn,
        edge_index : Tensor,
    ) -> Tuple[Tensor, Tensor]:

        self.send_info_event(
            progress=0.0,
            message=(
                f"[Cluster] Running {self.config.clustering_alg} "
                f"(resolution={self.config.clustering_resolution}) …"
            ),
        )

        entity_label_probas = results.entities_predictions.sigmoid()

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
        communities = torch.from_numpy(alg.fit_predict(adj)).to(results.entities_embeddings.device)

        n_communities = int(communities.max().item() + 1)

        self.send_info_event(
            progress=0.0,
            message=f"[Cluster] Found {n_communities} communities",
        )

        # Pool entity embeddings per community → community-level label probas
        with torch.no_grad():

            community_embeddings = model.readout_fn.forward(
                x = results.entities_embeddings,
                index = communities,
                dim_size = n_communities,
                dim = 0
            )

            communities_labels_probas = model.head(community_embeddings).sigmoid()

        self.send_debug_event(
            progress=0.0,
            message=(
                f"[Cluster] Community label probas shape: "
                f"{list(communities_labels_probas.shape)}"
            ),
        )

        return communities, communities_labels_probas
    
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
        communities_labels : Tensor
    ) -> Dict:
        
        n_communities = communities.max().item() + 1
        communities_nodelist = [{ "id" : i, "entities" : [], "labels" : [] } for i in range(n_communities)]

        for entity_id, community in zip(data.kg.entity_names, communities):
            communities_nodelist[community.item()]['entities'].append(entity_id)
        
        for community, labels in enumerate(communities_labels):

            for label_name, label_proba in zip(data.metadata['labels'], labels):

                if label_proba >= self.config.label_threshold:
                    
                    communities_nodelist[community]['labels'].append((
                        label_name,
                        label_proba.item()
                    ))

        communities_nodelist = list(
            map(
                partial(self.query_representative_node,id2Entity=id2Entity),
                filter(
                    lambda x : len(x['entities']) >= self.config.min_nodes_per_community,
                    map(
                        partial(self.clean_community,id2Entity=id2Entity),
                        communities_nodelist
                    )
                )
            )
        )

        self.send_info_event(
            progress=0.0,
            message=f"Number of communities after processing : {len(communities_nodelist)} ",
        )

        return communities_nodelist

    def query_representative_node(self, community : Dict, id2Entity : Dict[str, Dict]) -> Dict:

        community = {**community}

        current_max = 0.0
        current_entity = None

        for e_id in community['entities']:

            e = id2Entity[e_id]

            if e['specificity_weighted_pagerank'] > current_max:
                current_max = e['specificity_weighted_pagerank']
                current_entity = e['name']

        community['name'] = current_entity

        return community
    
    def update_graph(self,
        raw_G : Dict[str, Dict[str, List[Dict]]],
        data : ArticleAsKG,
        communities_nodelist : List[Dict],
        results : HetModelReturn,
    ) -> Dict[str, Dict[str, List[Dict]]]:
        
        raw_G['nodes']['Expertise'] = []
        raw_G['edges']['Expertise_BelongsTo_SubDomain'] = []
        raw_G['edges']['Entity_BelongsTo_Expertise'] = []
        raw_G['edges']['Entity_BelongsTo_SubDomain'] = []
        
        ### Community Node & Community <-> SubDomain links & Community <-> Entity Links
        for community in communities_nodelist:

            raw_G['nodes']['Expertise'].append({
                "id" : community['id'],
                "name" : community['name'],
                "type" : "Expertise"
            })

            for subdomain_id, edge_proba in community['labels']:

                raw_G['edges']['Expertise_BelongsTo_SubDomain'].append({
                    "id": f"{community['id']}-{subdomain_id}",
                    "src": community['id'],
                    "dst": subdomain_id,
                    "score": edge_proba
                })

            for entity in community['entities']:

                raw_G['edges']['Entity_BelongsTo_Expertise'].append({
                    "id": f"{entity}-{community['id']}",
                    "src": entity,
                    "dst": community['id'],
                })

        ### Add Entity <-> SubDomain links
        for entity, probas in zip(data.kg.entity_names, results.entities_predictions):

            for subdomain_id, proba in zip(data.metadata['labels'], probas):

                proba = proba.sigmoid().item()

                if proba >= self.config.label_threshold:

                    raw_G['edges']['Entity_BelongsTo_SubDomain'].append({
                        "id": f"{entity}-{subdomain_id}",
                        "src": entity,
                        "dst": subdomain_id,
                        "score" : proba
                    })

        return raw_G
    
    def __iter__(self):

        data, model, raw_G = self.load_data_and_model()
        inference_results  = self.inference(model, data)
        communities, communities_labels = self.cluster(model, inference_results, data.kg.edge_index)
        id2Entity = dict([(e['id'], e) for e in raw_G['nodes']['Entity']])

        communities_nodelist = self.process_communities(
            id2Entity,
            data,
            communities,
            communities_labels
        )

        raw_G = self.update_graph(
            raw_G,
            data,
            communities_nodelist,
            inference_results
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
        builder = cls(config=config)
        builder.add_handler(ConsoleLogger("ExpertiseMapBuilder"))
        return builder