import torch
import networkx as nx
from itertools import batched
from torch_geometric import utils as gnn_utils
from torch import Tensor
from src.shared.command_tools import BaseCommand
from src.commands.feature_extraction.graph_utils import Graph, Node, EdgeType, EdgeTypes
from sentence_transformers import SentenceTransformer
from typing import Dict, TypedDict, List
from collections import defaultdict
from definitions import hf_model_dir

def _minmax(t: Tensor) -> Tensor:
    mn, mx = t.min(), t.max()
    return (t - mn) / (mx - mn + 1e-8)

class Features(TypedDict):
    nodes : Dict[str, Dict[str, Tensor]]
    edges : Dict[str, Dict[str, Tensor]]

class Embedder:

    def __init__(self, 
        model : SentenceTransformer,
    ) -> None:
        self.model = model

    def _query(self, sentences : List[str], **kwargs) -> Tensor:
        
        return self.model.encode(
            sentences=sentences,
            **kwargs
        )

    def _passage(self, sentences : List[str], **kwargs) -> Tensor:
        
        return self.model.encode(
            sentences=sentences,
            **kwargs
        )
    
    def extract(
        self,
        sentences : List[str],
        query : bool,
        **kwargs
    ) -> Tensor:
        
        if query:
            return self._query(sentences, **kwargs)
        
        return self._passage(sentences, **kwargs)
    
    def is_basic(self) -> bool:
        return self.__class__ == Embedder
    
    @staticmethod
    def from_name(model_name : str, device : str) -> 'Embedder':

        model = SentenceTransformer(model_name,device=device)

        if 'e5' in model_name.lower():
            return E5Embedder(model)
        
        if 'qwen' in model_name.lower():
            return QwenEmbedder(model)
        
        return Embedder(model)

class QwenEmbedder(Embedder):

    def _query(self, sentences : List[str], **kwargs) -> Tensor:

        return self.model.encode(
            sentences=sentences,
            prompt_name="query",
            **kwargs
        )
    
    def _passage(self, sentences : List[str], **kwargs) -> Tensor:
        
        return self.model.encode(
            sentences=sentences,
            **kwargs
        )
    
class E5Embedder(Embedder):

    def _query(self, sentences : List[str], **kwargs) -> Tensor:

        return self.model.encode(
            sentences=[f"query: {sent}" for sent in sentences],
            **kwargs
        )
    
    def _passage(self, sentences : List[str], **kwargs) -> Tensor:
        
        return self.model.encode(
            sentences=[f"passage: {sent}" for sent in sentences],
            **kwargs
        )

class GraphEmbedder:

    def __init__(self,
        _host : BaseCommand,
        model_name : str,
        batch_size : int,
        device : str             
    ) -> None:
        self._host = _host
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
    
    def _extract_for_node(self, 
        n_type : str,
        attr_name : str,
        graph : Graph,
        model : Embedder, 
        query : bool
    ) -> Tensor:

        progress = 0.0

        self._host.send_info_event(
            progress=progress,
            task=f"feature-extraction/{n_type}",
            message=f"[STAGE 3] Starting Feature Extraction for node {n_type}..."
        )

        F = []

        def _map(node : Node) -> str:
            return node.properties[attr_name]

        for batch in batched(map(_map, graph.get_nodes(n_type).nodes_generator), self.batch_size):
        
            features = model.extract(
                sentences=list(batch),
                query=query,
                batch_size=self.batch_size,
                device=self.device,
                show_progress_bar=False,
                convert_to_tensor=True,
                normalize_embeddings=True,
            ).cpu()

            F.append(features)

            progress += len(batch) / len(graph.get_nodes(n_type))

            self._host.send_info_event(
                progress=progress,
                task=f"feature-extraction/{n_type}",
                message=f"[STAGE 3] {len(batch)} embeddings computed..."
            )

        F = torch.cat(F, dim=0)

        self._host.send_debug_event(
            progress=1.0,
            task=f"feature-extraction/{n_type}",
            message=f"[STAGE 3] Feature Matrix Shape for node {n_type} is {'x'.join(map(str,F.shape))}..."
        )

        return F

    def _extract_for_edge(self, 
        e_type : EdgeType,
        attr_name : str,
        graph : Graph,
        model : Embedder, 
        query : bool
    ) -> Tensor:

        progress = 0.0

        self._host.send_info_event(
            progress=progress,
            task=f"feature-extraction/{e_type.src}_{e_type.name}_{e_type.dst}",
            message=f"[STAGE 3] Starting Feature Extraction for edge {e_type.src}_{e_type.name}_{e_type.dst}..."
        )

        F = []

        def _map(node : Node) -> str:
            return node.properties[attr_name]

        for batch in batched(map(_map, graph.get_edges(e_type).edges_generator), self.batch_size):
        
            features = model.extract(
                sentences=list(batch),
                query=query,
                batch_size=self.batch_size,
                device=self.device,
                show_progress_bar=False,
                convert_to_tensor=True,
                normalize_embeddings=True,
            ).cpu()

            F.append(features)

            progress += len(batch) / len(graph.get_edges(e_type))

            self._host.send_info_event(
                progress=progress,
                task=f"feature-extraction/{e_type.src}_{e_type.name}_{e_type.dst}",
                message=f"[STAGE 3] {len(batch)} embeddings computed..."
            )

        F = torch.cat(F, dim=0)

        self._host.send_debug_event(
            progress=1.0,
            task=f"feature-extraction/{e_type}",
            message=f"[STAGE 3] Feature Matrix Shape for edge {e_type} is {'x'.join(map(str,F.shape))}..."
        )

        return F

    def _compute_scores(self, 
        graph : Graph,
        features : Features
    ) -> Features:
        
        r = EdgeTypes.Entity_APPEARNSIN_Article

        generality_scores = []
        relevancy_scores = []

        self._host.send_info_event(
            progress=0.0,
            task="feature-extraction/scores",
            message="[Stage 3] computing generality and relevancy scores..."
        )

        for e_idx in range(features['nodes']['Entity']['x'].size(0)):

            entities, articles = graph.pygeo[r.src, r.name, r.dst].edge_index
            art_indices = torch.unique(articles[entities == e_idx])

            a_emb = features['nodes']['Article']['x'][art_indices]
            generality_scores.append(1.0 - (a_emb @ a_emb.T).mean().item())

            e_vec = features['nodes']['Entity']['x'][e_idx].unsqueeze(0)
            p_emb = features['nodes']['Article']['x_passage'][art_indices]
            relevancy_scores.append(1.0 - (e_vec @ p_emb.T).mean().item())

            if (e_idx % 4096 == 0) or (e_idx+1) == len(features['nodes']['Entity']['x']):

                self._host.send_info_event(
                    progress=((e_idx+1)/len(features['nodes']['Entity']['x'])),
                    task="feature-extraction/scores",
                    message="[Stage 3] computing generality and relevancy scores..."
                )

        relevancy_scores = torch.tensor(relevancy_scores)
        generality_scores = torch.tensor(generality_scores)

        row, col = graph.pygeo['Entity', 'HASRELATIONWITH', 'Entity'].edge_index
        num_entities = graph.pygeo['Entity'].num_nodes

        features['nodes']['Entity']['relevancy_scores'] = relevancy_scores
        features['nodes']['Entity']['generality_scores'] = generality_scores
        features['nodes']['Entity']['out_degree'] = gnn_utils.degree(row, num_entities)
        features['nodes']['Entity']['in_degree'] = gnn_utils.degree(col, num_entities)
        features['nodes']['Entity']['degree'] = features['nodes']['Entity']['in_degree'] + features['nodes']['Entity']['out_degree']

        ### PageRank Score
        G = nx.DiGraph()

        for e in graph.get_edges(EdgeTypes.Entity_HASRELATIONWITH_Entity).edges_generator:
            G.add_edge(e.src, e.dst)

        G = G.to_undirected()

        scores = nx.pagerank(G)
        scores = torch.tensor([scores[e] for e in graph.get_nodes('Entity').ids()])

        features['nodes']['Entity']['pagerank'] = scores
        features['nodes']['Entity']['specificity_weighted_pagerank'] = scores * (
            torch.log(torch.scalar_tensor(num_entities).float() + 1.0) / 
            torch.log(features['nodes']['Entity']['degree'] + 1.0)
        )

        for key in ['relevancy_scores', 'generality_scores', 'pagerank', 'specificity_weighted_pagerank']:
            features['nodes']['Entity'][key] = _minmax(features['nodes']['Entity'][key])

        return features

    def extract_features(self, graph : Graph) -> Graph:

        spec = {
            "nodes" : {
                "Article" : "abstract",
                "Entity"  : "name",
                "SubDomain" : "name",
                "AltLabel" : "name"
            },
            "edges" : {
                EdgeTypes.Entity_HASRELATIONWITH_Entity : "predicate"
            }
        }

        features : Features = {
            "nodes" : defaultdict(dict),
            "edges" : defaultdict(dict)
        }

        total = len(spec['nodes']) + len(spec['edges'])

        self._host.send_info_event(
            progress=0.0,
            task="feature-extraction",
            message="[STAGE 3] Feature Extraction..."
        )

        model = Embedder.from_name(hf_model_dir(self.model_name), device=self.device)
        
        ### Nodes
        for i, (n_type, attr_name) in enumerate(spec['nodes'].items()):
            
            features['nodes'][n_type]["x"] = self._extract_for_node(n_type, attr_name, graph, model, query=True)

            if n_type == 'Article':

                if model.is_basic():
                    features['nodes'][n_type]["x_passage"] = features['nodes'][n_type]["x"]
                else:
                    features['nodes'][n_type]["x_passage"] = self._extract_for_node(n_type, attr_name, graph, model, query=False)
            

            self._host.send_info_event(
                progress=(i+1)/total,
                task="feature-extraction",
                message=f"[STAGE 3] Feature Extraction for node {n_type} completed..."
            )

        ### Edges
        for i, (e_type, attr_name) in enumerate(spec['edges'].items()):

            features['edges'][e_type]["edge_attr"] = self._extract_for_edge(e_type, attr_name, graph, model, query=True)

            self._host.send_info_event(
                progress=(i+1+len(spec['nodes'].items()))/total,
                task="feature-extraction",
                message=f"[STAGE 3] Feature Extraction for edge {e_type} completed..."
            )

        ### Scores
        features = self._compute_scores(graph, features)

        for n_type, features_dict in features['nodes'].items():
            graph._nodes[n_type]._features = {
                **graph._nodes[n_type]._features,
                **features_dict
            }

        for e_type, features_dict in features['edges'].items():

            graph._edges[e_type]._features = {
                **graph._edges[e_type]._features,
                **features_dict
            }

        return graph