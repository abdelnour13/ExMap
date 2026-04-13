import torch
import os
import pandas as pd
import numpy as np
import networkx as nx
from torch_scatter import scatter
from rich.progress import Progress
from pydantic import BaseModel, computed_field
from typing import Self, Dict, List, Tuple, Optional, Literal, Any
from definitions import EXPERIMENTS_DIR, DATA_DIR
from torch_geometric import utils as gnn_utils
from torch import Tensor
from src.shared.command_tools import BaseCommand
from src.gnn import ArticleAsKG, HetModel, HetModelReturn, KnowledgeGraph
from src.shared.handlers import ConsoleLogger
from src.commands.train import Trainer, Config as TrainerConfig
from src.shared.file_tools import FileWriter
from src.commands.build_kg.graph_construction.neo4j import Neo4J, Config as Neo4JConfig
from instance import cli
from sknetwork.clustering import Leiden, Louvain
from sknetwork.data import from_edge_list
from rich.progress import Progress

class Config(BaseModel):

    corpus : str = 'nantes-uni-2020'
    experiment : str = 'default'
    model : str = 'default'

    def model_post_init(self, context : Any) -> None:
        os.makedirs(self.map_dir, exist_ok=True)

    @computed_field
    @property
    def map_dir(self) -> str:
        return os.path.join(DATA_DIR, self.corpus, 'expertise_map')

    @computed_field
    @property
    def map_filename(self) -> str:
        return os.path.join(self.map_dir, 'expertise_map.jsonl')
    
    @computed_field
    @property
    def nodes_metadata_filename(self) -> str:
        return os.path.join(self.map_dir, 'nodes_metadata.csv')
    
    @computed_field
    @property
    def community_metadata_filename(self) -> str:
        return os.path.join(self.map_dir, 'community_metadata.csv')

def jaccard_index(
    edge_index : Tensor, 
    probas : Tensor, 
    mode : Literal['soft','hard'] = "soft",
    threshold : float = 0.5
) -> Tensor:
    
    if mode == 'hard':
        probas = probas.gt(threshold).float()

    row, col = edge_index
    a = torch.stack([probas[row], probas[col]], dim = 0)
    intersection = a.min(0).values.sum(-1)
    union = a.max(0).values.sum(-1)
    jacc_idx = intersection / union
    jacc_idx = torch.where(jacc_idx.isnan(), 0.0, jacc_idx)
    return jacc_idx

@cli.register_stage(
    name="build_map",
    description=(
        "Runs the pipeline responsible for building the expertise map from the knowledge graph and labeling nodes and  "
        "the clusters."
    ),
    add_from_conf=False,
    add_save_to=False
)
class ExpertiseMapBuilder(BaseCommand):

    def __init__(self, config : Config):
        super().__init__()
        self.config = config

    def load_data_and_model(self) -> Tuple[ArticleAsKG, HetModel]:

        self.send_debug_event(
            progress=0.0,
            message="Loading config file..."
        )

        config = TrainerConfig(experiment=self.config.experiment)
        trainer = Trainer(experiment=config.experiment, config=config.training_config)
        configs = trainer.get_configs()
        config = configs[self.config.model]

        self.send_debug_event(
            progress=0.0,
            message="Loading data..."
        )

        data = trainer.load_data()
        data = trainer.filter_data(data)
        data = trainer.keep_largest_cc(data) if config.keep_largest_cc else data
        data.split = trainer.create_or_load_split(data)
        data.articles.split = data.split

        self.send_debug_event(
            progress=0.0,
            message=f"Loading model {self.config.model}..."
        )

        model = trainer.create_model(config = config, data = data)

        state_dict = torch.load(
            f = os.path.join(
                EXPERIMENTS_DIR,
                self.config.experiment, 
                "models",
                f'model_{self.config.model}_0.pt',
            ), 
            weights_only = False, 
            map_location = 'cpu'
        )

        model.load_state_dict(state_dict)

        return data, model
    
    def get_metadata_df(self,
        data : ArticleAsKG,              
        results : HetModelReturn,
        threshold : float      
    ) -> pd.DataFrame:
        
        self.send_info_event(
            progress=0.0,
            message="Constructing entities metadata..."
        )
        
        kg = data.kg
        
        ### Labels
        probas = results.entities_predictions.sigmoid()
        labels = probas.gt(threshold)

        metadata = pd.DataFrame(
            data = labels.numpy(),
            index=np.array(kg.entity_names),
            columns=data.metadata['labels']
        )

        ### Calculate PageRank Score
        G = nx.from_edgelist(kg.edge_index.t().tolist())
        pagerank_scores = pd.Series(nx.pagerank(G))
        pagerank_scores = pagerank_scores[np.arange(kg.num_nodes)].to_numpy()

        ### Degrees
        out_degrees = gnn_utils.degree(index = kg.edge_index[0], num_nodes = kg.num_nodes).numpy()
        in_degrees = gnn_utils.degree(index = kg.edge_index[1], num_nodes = kg.num_nodes).numpy()
        degrees = out_degrees + in_degrees

        ### Specifity Weighted Page Rank Score
        specifity_weighted_page_rank = pagerank_scores * np.log((kg.num_nodes + 1) / (degrees + 1))

        metadata['pagerank'] = pagerank_scores
        metadata['specifity_weighted_page_rank'] = specifity_weighted_page_rank
        metadata['degree'] = degrees

        ### Calculate Number of Labels in the neighborhood
        neighborhood_labels = scatter(
            src = labels[torch.cat([kg.edge_index[0], kg.edge_index[1]], dim = -1)],
            index = torch.cat([kg.edge_index[1], kg.edge_index[0]], dim = -1),
            dim = 0,
            reduce="sum"
        )

        neighborhood_labels = neighborhood_labels.max(-1).values / (neighborhood_labels.sum(-1) + 1e-3)

        metadata['neighborhood_labels'] = neighborhood_labels.numpy()
        metadata['num_labels'] = metadata[data.metadata['labels']].sum(1)

        return metadata
    
    def inference(self,
        model : HetModel,
        data : ArticleAsKG
    ) -> HetModelReturn:
        
        self.send_info_event(
            progress=0.0,
            message="Inference..."
        )
        
        model = model.eval().cpu()
        data = data.cpu()

        model.ssl = False
        model.cluster = False

        with torch.no_grad():
            return model.forward(data)
        
    def cluster(self,
        model : HetModel,
        entities_embeddings : Tensor,
        edge_index : Tensor,
        alg : Literal["leiden", "louvain"] = "leiden",
        alg_args : Optional[Dict] = None,
        jaccrd_args : Optional[Dict] = None
    ) -> Tuple[Tensor, Tensor]:
        
        self.send_info_event(
            progress=0.0,
            message="Clustering..."
        )
        
        ### Arguments
        alg_args = alg_args or {}

        if "resolution" not in alg_args:
            alg_args["resolution"] = 15

        ### Cluster Labels
        with torch.no_grad():
            entities_labels_probas = model.head(entities_embeddings).sigmoid()

        ### Create new graph based on jaccard similarity
        jaccard_args = jaccrd_args or { "mode" : "hard", "threshold" : 0.5 }

        edge_weight = self.jaccard_index(edge_index, entities_labels_probas, **jaccard_args)

        edge_list = list(zip(
            edge_index[0].tolist(),
            edge_index[1].tolist(),
            edge_weight.tolist()
        ))

        adj = from_edge_list(edge_list)

        ### Perform Community Detection
        lv = Leiden(**alg_args) if alg == "leiden" else Louvain(**alg_args)
        communities = lv.fit_predict(adj)
        communities = torch.from_numpy(communities).to(entities_labels_probas.device)

        ### Graph Coarsining

        # Clusters-level Representations
        N = communities.max().item() + 1

        with torch.no_grad():

            clusters_representations = model.readout_fn.forward(
                x = entities_embeddings,
                index = communities,
                dim_size = N,
                dim = 0
            )

            communities_labels_probas = model.head(clusters_representations).sigmoid()

        self.send_debug_event(
            progress=0.0,
            message=f"Found total of {communities.max().item()} communities..."
        )

        return communities, communities_labels_probas
    
    def query_representative_nodes(self,
        nodes_metadata : pd.DataFrame,
        domains : List[str],
        communities : Tensor,
        communities_labels_probas : Tensor,
        max_labels : int = 3,
        max_degree : int = 50,
        threshold : float = 0.5,
    ) -> pd.DataFrame:
        
        self.send_info_event(
            progress=0.0,
            message="Querying representative nodes..."
        )
        
        domains = np.array(domains)

        nodes_metadata['community'] = communities
        community_labels = communities_labels_probas.gt(threshold)

        centroids_df = []

        pbar = Progress()
        all_comunities = np.unique(communities.numpy()).tolist()
        task = pbar.add_task("Getting Community Centroids", total=len(all_comunities))
        pbar.start()

        for community in all_comunities:

            condidates = nodes_metadata \
                .query(f"community == {community}") \
                .query(f"num_labels <= {max_labels}") \
                .query(f"degree <= {max_degree}") \
                .sort_values(by = "specifity_weighted_page_rank", ascending = False)

            community_domains = domains[community_labels[community,:]].tolist()

            for domain in community_domains:
                condidates = condidates[condidates[domain]]

            if condidates.shape[0] > 0:

                centroids_df.append({
                    "id" : community,
                    "name" : condidates.head(1).index.tolist()[0],
                    "domains" : community_domains,
                })
                
            pbar.update(task, advance=1, refresh=True)

        self.send_debug_event(
            progress=0.0,
            message=f"Number of communities with valid representative nodes {len(centroids_df)}..."
        )

        df = pd.DataFrame(centroids_df)
        df['size'] = torch.unique(communities, return_counts=True)[1].numpy()[df['id'].to_numpy()]

        return df
    
    def build_graph(self,
        kg : KnowledgeGraph,
        predicates : List[str],
        nodes_metadata : pd.DataFrame,
        communities_metadata : pd.DataFrame,
        labels : list[str],
        mapping : Dict[str, str],
    ) -> None:
        
        self.send_info_event(
            progress=0.0,
            message="Starting Graph Construction..."
        )
        
        nodes = {}
        relations = []
        domains = np.array(labels)

        pbar = Progress()

        task1 = pbar.add_task("Linking Between Domains and Sub-Domains", total=len(mapping))
        task2 = pbar.add_task("Linking Expertise to their domains", total=len(communities_metadata))
        task3 = pbar.add_task("Adding Graph Nodes", total=len(nodes_metadata))
        task4 = pbar.add_task("Adding Knowledeg Graph Nodes", total=kg.edge_index.shape[1])

        pbar.start()
        
        ### Link SubDomains to their domains
        for sub_domain, domain in mapping.items():

            nodes[f"SubDomain_{sub_domain}"] = {
                "type" : "Domain",
                "proprties" : {
                    "id" : f"SubDomain_{sub_domain}",
                    "name" : sub_domain,
                    "labels" : [domain, sub_domain],
                    "score" : 1.0
                }
            }

            nodes[f"Domain_{domain}"] = {
                "type" : "Domain",
                "proprties" : {
                    "id" : f"Domain_{domain}",
                    "name" : domain,
                    "labels" : [domain, "univ-nantes-2020"],
                    "score" : 1.0
                }
            }

            relations.append({
                "id" : f"SubDomain_{sub_domain}--Domain_{domain}",
                "relation" : {
                    "type" : "isSubdomainOf",
                    "properties" : {}
                },
                "n1" : nodes[f"SubDomain_{sub_domain}"],
                "n2" : nodes[f"Domain_{domain}"]
            })

            pbar.update(task1, advance=1, refresh=True)

        ### Link Expertise to their domains
        for idx, row in communities_metadata.iterrows():

            labels = row['domains']
            sub_labels = set([mapping[label] for label in labels])

            nodes[f"Expertise_{idx}"] = {
                "type" : "Expertise",
                "proprties" : {
                    "id" : f"Expertise_{idx}",
                    "name" : row['name'],
                    "labels" : [*labels,*sub_labels],
                    "score" : 1.0 # Replace it with the probability
                },
            }

            for sub_domain in row['domains']:
                
                relations.append({
                "id" : f"Expertise_{idx}--SubDomain_{sub_domain}",
                "relation" : {
                    "type" : "hasSubdomain",
                    "properties" : {}
                },
                "n1" : nodes[f"Expertise_{idx}"],
                "n2" : nodes[f"SubDomain_{sub_domain}"],
            })
                
            pbar.update(task2, advance=1, refresh=True)

        ### Link Nodes In Knowledge Graph to their Communities
        for i,(idx, node) in enumerate(nodes_metadata.iterrows()):

            labels = set(domains[nodes_metadata[domains].loc[idx].values])
            sub_labels = set([mapping[label] for label in labels])

            nodes[f"Node_{i}"] = {
                "type" : "Node",
                "proprties" : {
                    "id" : f"Node_{i}",
                    "name" : idx,
                    "labels" : [*labels,*sub_labels],
                    "score" : node["specifity_weighted_page_rank"]
                },
            }

            if node['community'] in communities_metadata['id']:

                relations.append({
                    "id" : f"Node_{i}--Expertise_{node['community']}",
                    "relation" : {
                        "type" : "hasSubdomain",
                        "properties" : {}
                    },
                    "n1" : nodes[f"Node_{i}"],
                    "n2" : nodes[f"Expertise_{node['community']}"]
                })

            pbar.update(task3, advance=1, refresh=True)

        ### Link Nodes With Other Nodes
        for (src, dst), edge_type in zip(kg.edge_index.t(), kg.edge_type):
            
            relations.append({
                "id" : f"Node_{src.item()}--Node_{dst.item()}",
                "relation" : {
                    "type" : "isRelatedTo",
                    "properties" : {
                        "predicate" : predicates[edge_type.item()]
                    }
                },
                "n1" : nodes[f"Node_{src.item()}"],
                "n2" : nodes[f"Node_{dst.item()}"]
            })

            pbar.update(task4, advance=1, refresh=True)

        return relations
            
    def __iter__(self):
        
        graph, model = self.load_data_and_model()
        results = self.inference(model, graph)
        nodes_metadata = self.get_metadata_df(graph, results, 0.5)

        communities, communities_labels_probas = self.cluster(
            model = model,
            entities_embeddings = results.entities_embeddings,
            edge_index = graph.kg.edge_index,
            alg = "leiden",
            alg_args = { "resolution" : 15 },
            jaccrd_args = { "mode" : "hard", "threshold" : 0.5 }
        )

        communities_metadata = self.query_representative_nodes(
            nodes_metadata=nodes_metadata,
            domains=graph.metadata['labels'],
            communities=communities,
            communities_labels_probas=communities_labels_probas,
            max_labels = 3,
            max_degree = 50,
            threshold = 0.5,
        )

        relations = self.build_graph(
            kg=graph.kg,
            predicates=graph.metadata['predicates'],
            nodes_metadata=nodes_metadata,
            communities_metadata=communities_metadata,
            labels=graph.metadata['labels'],
            mapping={
                "Robotics" : "Advanced Production Technologies",
                "Processes and Assembly" : "Advanced Production Technologies",
                "Additive Manufacturing" : "Advanced Production Technologies",
                "Monitoring and Analysis" : "Advanced Production Technologies",
                "Eco-design" : "Sustainable Industry",
                "Reduction and Treatment of Effluents and Waste" : "Sustainable Industry",
                "Recycling / Reuse" : "Sustainable Industry",
                "Eco-responsible Processes" : "Sustainable Industry",
                "Decarbonization of Mobility" : "Sustainable Industry",
                "Decarbonization of Energy" : "Sustainable Industry",
                "Composites and New Materials" : "Sustainable Industry",
                "Industrial Digitalization" : "Industrial Digitalization",
                "Digital Engineering of Products and Processes" : "Industrial Digitalization",
                "Industrial Organization Performance" : "Industrial Digitalization",
                "Preparation for New Jobs in the Future Industry" : "Industry 5.0",
                "Change Management" : "Industry 5.0",
                "User-Centered Design" : "Industry 5.0",
                "Augmentation of Human Capabilities in Manufacturing" : "Industry 5.0"
            },
        )

        nodes_metadata.to_csv(self.config.nodes_metadata_filename,index=True)
        communities_metadata.to_csv(self.config.community_metadata_filename,index=False)
        
        writer = FileWriter.from_filename(self.config.map_filename).buffered(128)

        for rel in relations:
            writer.write([rel])

        writer.flush()

        neo4j = Neo4J.from_config(Neo4JConfig(
            input_filename=self.config.map_filename,
            output_dir=os.path.join(self.config.map_dir, "neo4j")
        ))

        neo4j.run()

        yield from []

    @classmethod
    def from_config(cls, config : Config) -> Self:

        builder = ExpertiseMapBuilder(config=config)
        
        builder \
            .add_handler(ConsoleLogger("ExpertiseMapBuilder"))

        return builder