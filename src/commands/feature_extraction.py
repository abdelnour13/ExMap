"""
Knowledge Graph Feature Extraction Pipeline
============================================
Processes JSONL triples into an embedded, annotated graph ready for GNN training.

Stages:
  1. Parse  – Build graph structure from raw JSONL records
  2. Filter – Prune by connectivity, coverage, and label criteria
  3. Embed  – Encode nodes/edges with SentenceTransformer
  4. Score  – Add generality, relevancy, PageRank, degree metadata
  5. Save   – Write training tensors + Neo4j-ready metadata (link preserved)

Entry-point:
    python feature_extraction.py                          # defaults
    python feature_extraction.py --corpus my-corpus
    python feature_extraction.py --min-entity-articles 3
"""

import os
import torch
import networkx as nx
from torch import Tensor
from itertools import batched
from src.shared.file_tools import FileReader
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Dict, Any, Optional, List, TypedDict, Tuple, Generator
from collections import defaultdict, Counter
from src import helpers
from src.shared.command_tools import BaseCommand
from src.shared.handlers import Event, ConsoleLogger
from sentence_transformers import SentenceTransformer
from functools import cached_property
from torch_geometric import utils as gnn_utils
from torch_geometric.data import HeteroData
from torch_geometric.transforms import RemoveIsolatedNodes
from definitions import KGS_DIR, FEATURES_DIR, hf_model_dir
from src.env import env
from pydantic import BaseModel, computed_field
from instance import cli

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slugify(s : str) -> str:
    return s.lower().replace(' ','_')

def _as_raw(o : object, exclude : Optional[List[str]] = None) -> object:

    exclude = exclude or []

    if isinstance(o, (list, tuple, set)):
        return [_as_raw(item, exclude) for item in o]
    
    if isinstance(o, (dict, defaultdict)):
        return {str(key) : _as_raw(item, exclude) for key, item in o.items() if key not in exclude}
    
    if is_dataclass(o):

        return {
            field.name : _as_raw(getattr(o, field.name), exclude)
            for field in fields(o)
            if field.name not in exclude
        }

    return o

def _pairs(items : list):

    v = []

    for item1, item2 in zip(items[:-1], items[1:]):
        yield '.'.join(v+[item1]), '.'.join(v+[item1,item2])
        v.append(item1)

def _minmax(t: Tensor) -> Tensor:
    mn, mx = t.min(), t.max()
    return (t - mn) / (mx - mn + 1e-8)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class FilterConfig(BaseModel):
    require_subdomain : bool = True
    min_entities_per_article : int = 1
    min_articles_per_entity : int = 1
    largest_connected_component : bool = True

class FeatureExtractionConfig(BaseModel):
    kg_id : str
    model_name: str
    batch_size: int
    device: str
    filter : FilterConfig

    @computed_field
    @property
    def raw_path(self) -> str:
        return os.path.join(KGS_DIR, self.kg_id, 'entities_mapping', 'data.jsonl')
    
class Config(BaseModel):
    features_id : str

    @computed_field
    @property
    def features_dir(self) -> str:
        return os.path.join(FEATURES_DIR, self.features_id)
    
    @computed_field
    @property
    def config_path(self) -> str:
        return os.path.join(self.features_dir, 'config.json')
    
    @computed_field
    @property
    def conf(self) -> FeatureExtractionConfig:
        return FeatureExtractionConfig(**helpers.load_json_file(self.config_path))
    
# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Node:
    id : str
    properties : Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash(self.id)

@dataclass
class Edge:
    src : str
    dst : str
    id : Optional[str] = None
    properties : Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):

        if self.id is None:
            self.id = f"{self.src}-{self.dst}"

    def __hash__(self):
        return hash(self.id)


@dataclass
class NodeStore:
    _nodes : List[Node] = field(default_factory=list,init=False)
    _id2idx : Dict[str, int] = field(default_factory=dict,init=False)
    _features : Dict[str, Tensor] = field(default_factory=dict,init=False)

    @property
    def nodes(self) -> List[Node]:

        return [
            Node(
                id=node.id,
                properties={
                    **node.properties,
                    **{
                        key : value[idx].item() if len(value.size()) == 1 else value[idx]
                        for key, value in self._features.items()
                    }
                }
            )
            for idx, node in enumerate(self._nodes)
        ]

    def add_node(self, id : str, properties : Optional[Dict[str, Any]] = None) -> bool:

        properties = properties or {}
        
        ### Retrieve node
        node = self.get_node(id)

        ### Case 01 : Node exists
        if node is not None:
            
            ### Update properties
            for key, value in properties.items():

                if key not in node.properties:
                    node.properties[key] = value
                elif isinstance(value, set) and isinstance(node.properties[key], set):
                    node.properties[key] = set.update(node.properties[key], value)

            return False
        
        ### Case 02 : Node doesn't exist
        self._id2idx[id] = len(self._nodes)
        self._nodes.append(Node(id=id,properties=properties))

        return True
    
    def get_node(self, id : str) -> Optional[Node]:

        if id not in self._id2idx:
            return None
        
        idx = self._id2idx[id]
        node = self._nodes[idx]

        properties = {
            key : value[idx].item() if len(value.size()) == 1 else value[idx]
            for key, value in self._features.items()
            if len(value.size()) == 1
        }

        return Node(
            id=node.id,
            properties={
                **node.properties,
                **properties
            }
        )
    
@dataclass
class EdgeStore:
    _edges : List[Edge] = field(default_factory=list,init=False)
    _id2idx : Dict[str, int] = field(default_factory=dict,init=False)
    _features : Dict[str, Tensor] = field(default_factory=dict,init=False)

    @cached_property
    def source_edges(self) -> Dict[str, Edge]:

        r = defaultdict(list)

        for edge in self._edges:
            r[edge.src].append(edge)

        return r
        
    
    @cached_property
    def target_edges(self) -> Dict[str, Edge]:
        
        r = defaultdict(list)

        for edge in self._edges:
            r[edge.dst].append(edge)

        return r
    
    @cached_property
    def edges(self) -> Dict[Tuple[str, str], Edge]:

        r = defaultdict(list)

        for edge in self._edges:
            r[(edge.src,edge.dst)].append(edge)

        return r

    def add_edge(self, 
        src : str, 
        dst : str, 
        id : Optional[str] = None, 
        properties : Optional[Dict[str, Any]] = None
    ) -> bool:
        
        properties = properties or {}
        
        ### Retrieve node
        id = id or Edge(src,dst).id
        edge = self.get_edge(id)

        ### Case 01 : Edge exists
        if edge is not None:
            
            ### Update properties
            for key, value in properties.items():

                if key not in edge.properties:
                    edge.properties[key] = value
                elif isinstance(value, set) and isinstance(edge.properties[key], set):
                    edge.properties[key] = set.update(edge.properties[key], value)

            return False
        
        ### Case 02 : Edge doesn't exist
        e = Edge(id=id,src=src,dst=dst,properties=properties)
        self._id2idx[e.id] = len(self._edges)
        self._edges.append(e)

        return True
    
    def get_edge(self, id : str) -> Optional[Edge]:

        if id not in self._id2idx:
            return None
        
        idx = self._id2idx[id]
        edge = self._edges[idx]

        properties = {
            key : value[idx].item() if len(value.size()) == 1 else value[idx]
            for key, value in self._features.items()
        }

        return Edge(
            id=edge.id,
            src=edge.src,
            dst=edge.dst,
            properties={
                **edge.properties,
                **properties
            }
        )

@dataclass(frozen=True)
class EdgeType:
    src : str
    dst : str
    name : str

    def __repr__(self):
        return f'{self.src}_{self.dst}_{self.name}'

    def __str__(self):
        return self.__repr__() 

class RawNodeStore(TypedDict):
    nodes : List[Dict]
    id2idx : Dict[str, int]

class RawEdgeStore(TypedDict):
    edges : List[Dict]
    id2idx : Dict[str, int]

class RawGraph(TypedDict):
    nodes : Dict[str, RawNodeStore]
    edges : Dict[str, RawEdgeStore]

@dataclass
class Graph:
    _nodes : Dict[str, NodeStore] = field(default_factory=lambda : defaultdict(NodeStore),init=False)
    _edges : Dict[EdgeType, EdgeStore] = field(default_factory=lambda : defaultdict(EdgeStore),init=False)

    @property
    def node_types(self) -> List[str]:
        return list(self._nodes.keys())
    
    @property
    def edge_types(self) -> List[EdgeType]:
        return list(self._edges.keys())
    
    @property
    def node_counts(self) -> Dict[str, int]:
        
        return {
            key : len(value._nodes)
            for key, value in self._nodes.items()
        }
    
    @property
    def edge_counts(self) -> Dict[EdgeType, int]:
        return {
            key : len(value._edges)
            for key, value in self._edges.items()
        }
    
    @cached_property
    def raw_data(self) -> RawGraph:
        return _as_raw(self, exclude=['_features'])
    
    @cached_property
    def degrees(self) -> Dict[EdgeType, Tuple[Counter, Counter]]:

        result = {}

        for edge_type, edge_store in self._edges.items():

            src_counter = Counter()
            dst_counter = Counter()

            for edge in edge_store._edges:
                src_counter[edge.src] += 1
                dst_counter[edge.dst] += 1

            result[edge_type] = (src_counter,dst_counter)

        return result
    
    @cached_property
    def pygeo(self) -> HeteroData:
        
        data = HeteroData()

        for node_type, node_store in self._nodes.items():

            data[node_type].node_id = torch.arange(len(node_store._nodes))

            for feature_name, feature_tensor in node_store._features.items():
                data[node_type][feature_name] = feature_tensor

        for edge_type, edge_store in self._edges.items():

            edge_index = [
                [
                    self._nodes[edge_type.src]._id2idx[e.src],
                    self._nodes[edge_type.dst]._id2idx[e.dst]
                ] 
                for e in edge_store._edges
            ]

            data[edge_type.src, edge_type.name, edge_type.dst].edge_index = torch.tensor(edge_index).t()
            data[edge_type.src, edge_type.name, edge_type.dst].edge_id = torch.arange(len(edge_store._edges))

            if edge_type == FeatureExtractor.Entity_HASRELATIONWITH_Entity:

                predicates = set([e.properties['predicate'] for e in edge_store._edges])
                predicates2id = { pred : i for i, pred in enumerate(predicates) }
                e_type = torch.tensor([predicates2id[e.properties['predicate']] for e in edge_store._edges])

                data[edge_type.src, edge_type.name, edge_type.dst].edge_type = e_type
                data.predicates2id = predicates2id

            for feature_name, feature_tensor in edge_store._features.items():
                data[edge_type.src, edge_type.name, edge_type.dst][feature_name] = feature_tensor

        return data

    def get_nodes(self, type : str) -> List[Node]:
        return self._nodes[type]._nodes
    
    def get_edges(self, type : EdgeType) -> List[Edge]:
        return self._edges[type]._edges

    def add_node(self, 
        type : str,
        id : str, 
        properties : Optional[Dict[str, Any]] = None
    ) -> bool:
        return self._nodes[type].add_node(id, properties)
    
    def get_node(self, type : str, id : str) -> Optional[Node]:
        return self._nodes[type].get_node(id)

    def add_edge(self, 
        type : EdgeType,
        src : str, 
        dst : str, 
        id : Optional[str] = None, 
        properties : Optional[Dict[str, Any]] = None
    ) -> bool:
        return self._edges[type].add_edge(src,dst,id,properties)
    
    def get_edge(self, type : EdgeType, id : str) -> Optional[Edge]:
        return self._edges[type].get_edge(id)
    
    def save(self, dir : str) -> None:

        helpers.save_json_file(os.path.join(dir, 'raw.json'), self.raw_data)

        features = {
            "nodes" : defaultdict(dict),
            "edges" : defaultdict(dict)
        }

        for node_type, node_store in self._nodes.items():
            features['nodes'][node_type] = node_store._features

        for edge_type, edge_store in self._edges.items():
            features['edges'][str(edge_type)] = edge_store._features

        torch.save(features, os.path.join(dir, 'features.pt'))
    
    @staticmethod
    def load(dir : str) -> 'Graph':
        
        graph = Graph()

        raw_data = helpers.load_json_file(os.path.join(dir, 'raw.json'))

        for node_type, node_store in raw_data['_nodes'].items():
            for node in node_store['_nodes']:
                graph.add_node(node_type, node['id'], node['properties'])

        for edge_type, edge_store in raw_data['_edges'].items():
            
            for edge in edge_store['_edges']:

                src, dst, name = edge_type.split('_')

                graph.add_edge(
                    EdgeType(src,dst,name), 
                    edge['src'], 
                    edge['dst'], 
                    edge['id'], 
                    edge['properties']
                )

        features = torch.load(os.path.join(dir, 'features.pt'), weights_only=False)

        for node_type, node_features in features['nodes'].items():
            graph._nodes[node_type]._features = node_features

        for edge_type, edge_features in features['edges'].items():
            src, dst, name = edge_type.split('_')
            graph._edges[EdgeType(src,dst,name)]._features = edge_features

        return graph


# ---------------------------------------------------------------------------
# Stage 01 : Graph Construction
# ---------------------------------------------------------------------------

class GraphBuilder:

    def __init__(self, 
        _host : BaseCommand,
        reader : FileReader
    ) -> None:
        self._host = _host
        self.reader = reader

    def build(self) -> Graph:

        self._host.send_info_event(
            progress=0.0,
            task="graph-construction",
            message="[STAGE 1] Graph construction..."
        )

        graph = Graph()

        for records, progress in self.reader.batched(4096):

            for record in records:
                self.update(record, graph)

            self._host.send_info_event(
                progress=progress,
                task="graph-construction",
                message=f"Processed {len(records)} records."
            )

        for domain in env.cs_domains:

            domain_id = _slugify(domain.domain)
            subdomain_id = _slugify(domain.sub_domain)

            graph.add_node(
                type='Domain',
                id=domain_id,
                properties={
                    "name" : domain.domain
                }
            )

            graph.add_edge(
                type=FeatureExtractor.SubDomain_ISSUBDOMAINOF_Domain,
                src=subdomain_id,
                dst=domain_id
            )

        return graph

    def update(self, doc: dict, graph: Graph) -> None:

        article = doc['sentence']['document'] if doc['sentence'] is not None else None

        if article is not None:

            # Article
            graph.add_node(
                type="Article",
                id=article['id'],
                properties={
                    "hal_id" : article['id'],
                    "title" : article['title'],
                    "abstract" : article['abstract'],
                    "date" : article['date']
                }
            )

            # Entity
            for entity in [doc['head'], doc['tail']]:

                entity_id = _slugify(entity)

                graph.add_node(
                    type='Entity',
                    id=entity_id,
                    properties={
                        "name" : entity
                    }
                )

                graph.add_edge(
                    type=FeatureExtractor.Entity_APPEARNSIN_Article,
                    src=entity_id,
                    dst=article['id']
                )

            head_id = _slugify(doc['head'])
            tail_id = _slugify(doc['tail'])

            graph.add_edge(
                type=FeatureExtractor.Entity_HASRELATIONWITH_Entity,
                src=head_id,
                dst=tail_id,
                id=f"{head_id}-{doc['predicate']}-{tail_id}",
                properties={
                    "predicate" : doc['predicate'],
                    "articles" : set([article['id']]) if article is not None else set(),
                    "source" : set(["llm"]) if article is not None else set()
                }
            )

        if article is not None:

            # Lab
            for lab in article['teams']:

                lab_id = lab

                graph.add_node(
                    type='Lab',
                    id=lab_id,
                    properties={
                        "name" : lab
                    }
                )

                graph.add_edge(
                    type=FeatureExtractor.Article_BELONGSTO_Lab,
                    src=article['id'],
                    dst=lab_id
                )

            # Venue
            for venue in article['venues']:

                graph.add_node(
                    type='Venue',
                    id=venue,
                    properties={
                        "name" : venue
                    }
                )

                graph.add_edge(
                    type=FeatureExtractor.Article_BELONGSTO_Venue,
                    src=article['id'],
                    dst=venue
                )

                for v1, v2 in _pairs(venue.split('.')):

                    graph.add_node(
                        type='Venue',
                        id=v1,
                        properties={
                            "name" : v1
                        }
                    )

                    graph.add_node(
                        type='Venue',
                        id=v2,
                        properties={
                            "name" : v2
                        }
                    )

                    graph.add_edge(
                        type=FeatureExtractor.Venue_ISSUBVENUEOF_Venue,
                        src=v2,
                        dst=v1
                    )

            # Author
            for author in article['authors']:

                author_id = _slugify(author)

                graph.add_node(
                    type='Author',
                    id=author_id,
                    properties={
                        "name" : author
                    }
                )

                graph.add_edge(
                    type=FeatureExtractor.Article_WRITTENBY_Author,
                    src=article['id'],
                    dst=author_id
                )

            # Keyword
            for keyword in article['keywords']:

                keyword_id = _slugify(keyword)

                graph.add_node(
                    type='Keyword',
                    id=keyword_id,
                    properties={
                        "name" : keyword
                    }
                )

                graph.add_edge(
                    type=FeatureExtractor.Article_HASKEYWORD_Keyword,
                    src=article['id'],
                    dst=keyword_id
                )

            # SubDomains
            for domain in article['labels']:

                domain_id = _slugify(domain)

                graph.add_node(
                    type='SubDomain',
                    id=domain_id,
                    properties={
                        "name" : domain
                    }
                )

                graph.add_edge(
                    type=FeatureExtractor.Article_BELONGSTO_SubDomain,
                    src=article['id'],
                    dst=domain_id
                )

# ---------------------------------------------------------------------------
# Stage 02 : Filtering
# ---------------------------------------------------------------------------

class GraphFilter:

    def __init__(self, 
        _host : BaseCommand,
        filter : FilterConfig
    ) -> None:
        self._host = _host
        self.filter = filter

    def filter_graph(self, graph : Graph) -> Graph:

        self._host.send_info_event(
            progress=None,
            task="graph-filtering",
            message="[STAGE 2] Graph Filtering..."
        )

        stop = False
        attempt = 0

        node_degrees = {
            type_ : graph.pygeo[type_].num_nodes
            for type_ in graph.pygeo.node_types
        }

        graph_geo = graph.pygeo
        
        while not stop:

            self._host.send_info_event(
                progress=None,
                task="graph-filtering",
                message=f"[STAGE 2] Graph Filtering attmpt={attempt}..."
            )

            ### Articles connected to at least one label & connected to at least one node
            numentities_per_article = gnn_utils.degree(graph_geo['Entity', 'APPEARIN', 'Article'].edge_index[1])
            article_mask = (numentities_per_article >= self.filter.min_entities_per_article)

            if self.filter.require_subdomain:
                numsubdomains_per_article = gnn_utils.degree(graph_geo['Article', 'BELONGSTO', 'SubDomain'].edge_index[0])
                article_mask = article_mask & (numsubdomains_per_article != 0)

            ### Entities : Largest connected component & connected to at least one article
            numarticles_per_entity = gnn_utils.degree(graph_geo['Entity', 'APPEARIN', 'Article'].edge_index[0])
            entity_mask = numarticles_per_entity >= self.filter.min_articles_per_entity

            if self.filter.largest_connected_component:

                component = helpers.torch_connected_components(
                    graph_geo['Entity', 'HASRELATIONWITH', 'Entity'].edge_index,
                    graph_geo['Entity'].num_nodes
                )

                unique_components, counts = torch.unique(component, return_counts=True)
                lcc = unique_components[counts == counts.max()][0].item()

                entity_mask = entity_mask & (component == lcc)

            graph_geo = graph_geo.subgraph({ 'Article' : article_mask, 'Entity' : entity_mask })

            ### Isolated Nodes
            rin = RemoveIsolatedNodes()
            graph_geo = rin.forward(graph_geo)

            ### Stopping critera
            new_node_degrees = {
                type_ : graph_geo[type_].num_nodes
                for type_ in graph_geo.node_types
            }

            stop = all([value == new_node_degrees[key] for key, value in node_degrees.items()])
            node_degrees = new_node_degrees
            attempt += 1

        ### Constructing new graph
        self._host.send_info_event(
            progress=None,
            task="graph-filtering",
            message=f"[STAGE 2] Constructing new graph..."
        )

        subset = Graph()

        for node_type in graph_geo.node_types:

            node_ids = graph_geo[node_type].node_id

            for node_id in node_ids.tolist():

                node = graph._nodes[node_type]._nodes[node_id]

                subset.add_node(
                    type=node_type,
                    id=node.id,
                    properties=node.properties
                )

        for edge_type in graph_geo.edge_types:

            edge_ids = graph_geo[edge_type].edge_id
            src, type_, dst = edge_type
            edge_type = EdgeType(src, dst, type_)

            for edge_id in edge_ids.tolist():

                edge = graph._edges[edge_type]._edges[edge_id]

                subset.add_edge(
                    type=edge_type,
                    src=edge.src,
                    dst=edge.dst,
                    id=edge.id,
                    properties=edge.properties
                )

        for node_type, node_store in subset._nodes.items():

            self._host.send_info_event(
                progress=None,
                task="graph-filtering",
                message=f"[STAGE 2] count({node_type})={len(node_store._nodes)}..."
            )

        for edge_type, edge_store in subset._edges.items():

            self._host.send_info_event(
                progress=None,
                task="graph-filtering",
                message=f"[STAGE 2] count({edge_type.src}_{edge_type.name}_{edge_type.dst})={len(edge_store._edges)}..."
            )

        return subset
    
# ---------------------------------------------------------------------------
# Stage 03 : Feature Extraction
# ---------------------------------------------------------------------------

class Features(TypedDict):
    nodes : Dict[str, Dict[str, Tensor]]
    edges : Dict[str, Dict[str, Tensor]]

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

    def _is_e5(self) -> bool:
        return "e5" in self.model_name.lower()

    def _e5(self, text: str, prefix: str) -> str:

        if self._is_e5():
            return f"{prefix}: {text}"
        
        return text
    
    def _extract_for_node(self, 
        n_type : str,
        attr_name : str,
        graph : Graph,
        model : SentenceTransformer, 
        emb_type : str = "query"
    ) -> Tensor:

        progress = 0.0

        self._host.send_info_event(
            progress=progress,
            task=f"feature-extraction/{n_type}",
            message=f"[STAGE 3] Starting Feature Extraction for node {n_type}..."
        )

        F = []

        def _map(node : Node) -> str:
            return self._e5(node.properties[attr_name], emb_type)

        for batch in batched(map(_map, graph.get_nodes(n_type)), self.batch_size):
        
            features = model.encode(
                sentences=list(batch),
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
        model : SentenceTransformer, 
        emb_type : str = "query"
    ) -> Tensor:

        progress = 0.0

        self._host.send_info_event(
            progress=progress,
            task=f"feature-extraction/{e_type.src}_{e_type.name}_{e_type.dst}",
            message=f"[STAGE 3] Starting Feature Extraction for edge {e_type.src}_{e_type.name}_{e_type.dst}..."
        )

        F = []

        def _map(node : Node) -> str:
            return self._e5(node.properties[attr_name], emb_type)

        for batch in batched(map(_map, graph.get_edges(e_type)), self.batch_size):
        
            features = model.encode(
                sentences=list(batch),
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
        
        r = FeatureExtractor.Entity_APPEARNSIN_Article

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

        for e in graph.get_edges(FeatureExtractor.Entity_HASRELATIONWITH_Entity):
            G.add_edge(e.src, e.dst)

        G = G.to_undirected()

        scores = nx.pagerank(G)
        scores = torch.tensor([scores[e.id] for e in graph.get_nodes('Entity')])

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
                "Entity"  : "name"
            },
            "edges" : {
                FeatureExtractor.Entity_HASRELATIONWITH_Entity : "predicate"
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

        model = SentenceTransformer(hf_model_dir(self.model_name), device=self.device)
        
        ### Nodes
        for i, (n_type, attr_name) in enumerate(spec['nodes'].items()):
            
            features['nodes'][n_type]["x"] = self._extract_for_node(n_type, attr_name, graph, model, "query")

            if n_type == 'Article':

                if not self._is_e5():
                    features['nodes'][n_type]["x_passage"] = features['nodes'][n_type]["x"]
                else:
                    features['nodes'][n_type]["x_passage"] = self._extract_for_node(n_type, attr_name, graph, model, "passage")
            

            self._host.send_info_event(
                progress=(i+1)/total,
                task="feature-extraction",
                message=f"[STAGE 3] Feature Extraction for node {n_type} completed..."
            )

        ### Edges
        for i, (e_type, attr_name) in enumerate(spec['edges'].items()):

            features['edges'][e_type]["edge_attr"] = self._extract_for_edge(e_type, attr_name, graph, model, "query")

            self._host.send_info_event(
                progress=(i+1+len(spec['nodes'].items()))/total,
                task="feature-extraction",
                message=f"[STAGE 3] Feature Extraction for edge {e_type} completed..."
            )

        ### Scores
        features = self._compute_scores(graph, features)

        for n_type, features_dict in features['nodes'].items():
            graph._nodes[n_type]._features = features_dict

        for e_type, features_dict in features['edges'].items():
            graph._edges[e_type]._features = features_dict

        return graph

# ---------------------------------------------------------------------------
# Main Command
# ---------------------------------------------------------------------------

@cli.register_stage(
    name="feature_extraction_2",
    description=(
        "Transforms raw JSONL triples into an embedded, annotated knowledge graph "
        "ready for GNN training. Includes parsing, filtering, sentence-bert embedding, "
        "PageRank scoring, and dual-format output (graph.pt + metadata.json)."
    ),
    add_from_conf=False,
    add_save_to=False,
)
# @add_handlers
class FeatureExtractor(BaseCommand):

    Entity_APPEARNSIN_Article = EdgeType("Entity", "Article", "APPEARIN")
    Entity_HASRELATIONWITH_Entity = EdgeType("Entity", "Entity", "HASRELATIONWITH")
    Article_BELONGSTO_Lab = EdgeType("Article", "Lab", "BELONGSTO")
    Article_BELONGSTO_Venue = EdgeType("Article", "Venue", "BELONGSTO")
    Article_WRITTENBY_Author = EdgeType("Article", "Author", "WRITTENBY")
    Article_HASKEYWORD_Keyword = EdgeType("Article", "Keyword", "HASKEYWORD")
    Article_BELONGSTO_SubDomain = EdgeType("Article", "SubDomain", "BELONGSTO")
    Venue_ISSUBVENUEOF_Venue = EdgeType("Venue", "Venue", "ISSUBVENUEOF")
    SubDomain_ISSUBDOMAINOF_Domain = EdgeType("SubDomain", "Domain", "ISSUBDOMAINOF")

    def __init__(self, 
        reader : FileReader, 
        filter : FilterConfig,
        model_name : str,
        batch_size : int,
        device : str,
        graph_dir : str
    ) -> None:
        super().__init__()
        self.reader = reader
        self.filter = filter
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.graph_dir = graph_dir

    def __iter__(self) -> Generator[Event, None, None]:

        ### Stage 01 : Graph Building
        graph = GraphBuilder(self, self.reader).build()

        ### Stage 02 : Filtering
        graph = GraphFilter(self, self.filter).filter_graph(graph)

        ### Stage 03 : Feature Extraction
        graph = GraphEmbedder(self, self.model_name, self.batch_size, self.device).extract_features(graph)

        ### Stage 04 : Save graph
        graph.save(self.graph_dir)

        # Load & Test
        g = Graph.load(self.graph_dir)   

        print(g.get_node('Entity', 'model'))     

        yield from []

    
    @classmethod
    def from_config(cls, config: Config) -> "FeatureExtractor":

        fe = FeatureExtractor(
            reader=FileReader.from_filename(filename=config.conf.raw_path),
            filter=config.conf.filter,
            model_name=config.conf.model_name,
            batch_size=config.conf.batch_size,
            device=config.conf.device,
            graph_dir=config.features_dir
        )

        fe.add_handler(ConsoleLogger("FeatureExtraction"))

        return fe