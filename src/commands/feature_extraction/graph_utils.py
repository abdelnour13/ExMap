import torch
import os
from src import helpers
from torch_geometric.data import HeteroData
from torch import Tensor
from collections import defaultdict, Counter
from functools import cached_property
from dataclasses import dataclass, field, is_dataclass, fields
from typing import Dict, Any, Optional, List, Tuple, Generator, TypedDict

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
    def nodes_generator(self) -> Generator[Node, None, None]:

        for idx, node in enumerate(self._nodes):

            yield Node(
                id=node.id,
                properties={
                    **node.properties,
                    **{
                        key : value[idx].item() 
                        for key, value in self._features.items()
                        if len(value.size()) == 1
                    }
                }
            )

    @cached_property
    def nodes(self) -> List[Node]:
        return list(self.nodes_generator)
    
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
    
    def add_node(self, 
        id : str, 
        properties : Optional[Dict[str, Any]] = None
    ) -> bool:

        properties = properties or {}
        
        ### Retrieve node
        node = self.get_node(id)

        ### Case 01 : Node exists
        if node is not None:
            
            ### Update properties
            for key, value in properties.items():

                if key not in node.properties:
                    node.properties[key] = value
                elif isinstance(node.properties[key], set):
                    value = set([value]) if not isinstance(value, set) else value
                    node.properties[key] = set.update(node.properties[key], value)

            return False
        
        ### Case 02 : Node doesn't exist
        self._id2idx[id] = len(self._nodes)
        self._nodes.append(Node(id=id,properties=properties))

        return True
    
    def __len__(self):
        return len(self._nodes)
    
    def ids(self) -> List[str]:
        return list(self._id2idx.keys())
    
@dataclass
class EdgeStore:
    _edges : List[Edge] = field(default_factory=list,init=False)
    _id2idx : Dict[str, int] = field(default_factory=dict,init=False)
    _features : Dict[str, Tensor] = field(default_factory=dict,init=False)

    @cached_property
    def source_edges(self) -> Dict[str, List[Edge]]:

        r = defaultdict(list)

        for edge in self._edges:
            r[edge.src].append(edge)

        return r
        
    @cached_property
    def target_edges(self) -> Dict[str, List[Edge]]:
        
        r = defaultdict(list)

        for edge in self._edges:
            r[edge.dst].append(edge)

        return r
    
    @cached_property
    def edges(self) -> Dict[Tuple[str, str], List[Edge]]:

        r = defaultdict(list)

        for edge in self._edges:
            r[(edge.src,edge.dst)].append(edge)

        return r
    
    @property
    def edges_generator(self) -> Generator[Edge, None, None]:

        for idx, edge in enumerate(self._edges):

            yield Edge(
                id=edge.id,
                src=edge.src,
                dst=edge.dst,
                properties={
                    **edge.properties,
                    **{
                        key : value[idx].item() if len(value.size()) == 1 else value[idx]
                        for key, value in self._features.items()
                        if len(value.size()) == 1
                    }
                }
            )


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
                elif isinstance(edge.properties[key], set):
                    value = value if isinstance(value, set) else set([value])
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
    
    def __len__(self):
        return len(self._edges)
    
    def ids(self) -> List[str]:
        return list(self._id2idx.keys())
    
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

class NodeTypes:
    Article = "Article"
    Entity = "Entity"
    Lab = "Lab"
    Venue = "Venue"
    Author = "Author"
    Keyword = "Keyword"
    SubDomain = "SubDomain"
    Domain = "Domain"
    AltLabel = "AltLabel"

class EdgeTypes:
    Entity_APPEARNSIN_Article = EdgeType("Entity", "Article", "APPEARIN")
    Entity_HASRELATIONWITH_Entity = EdgeType("Entity", "Entity", "HASRELATIONWITH")
    Article_BELONGSTO_Lab = EdgeType("Article", "Lab", "BELONGSTO")
    Article_BELONGSTO_Venue = EdgeType("Article", "Venue", "BELONGSTO")
    Article_WRITTENBY_Author = EdgeType("Article", "Author", "WRITTENBY")
    Article_HASKEYWORD_Keyword = EdgeType("Article", "Keyword", "HASKEYWORD")
    Article_BELONGSTO_SubDomain = EdgeType("Article", "SubDomain", "BELONGSTO")
    Venue_ISSUBVENUEOF_Venue = EdgeType("Venue", "Venue", "ISSUBVENUEOF")
    SubDomain_ISSUBDOMAINOF_Domain = EdgeType("SubDomain", "Domain", "ISSUBDOMAINOF")
    SubDomain_HASALTLABEL_Domain = EdgeType("SubDomain", "AltLabel", "HASALTLABEL")

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
    def degrees(self) -> Dict[EdgeType, Tuple[Counter, Counter, Counter]]:

        result = {}

        for edge_type, edge_store in self._edges.items():

            src_counter = Counter()
            dst_counter = Counter()
            global_counter = Counter()

            for edge in edge_store._edges:
                src_counter[edge.src] += 1
                dst_counter[edge.dst] += 1
                global_counter[edge.src] += 1
                global_counter[edge.dst] += 1

            result[edge_type] = (src_counter,dst_counter,global_counter)

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

            if edge_type == EdgeTypes.Entity_HASRELATIONWITH_Entity:

                predicates = set([e.properties['predicate'] for e in edge_store._edges])
                predicates2id = { pred : i for i, pred in enumerate(predicates) }
                e_type = torch.tensor([predicates2id[e.properties['predicate']] for e in edge_store._edges])

                data[edge_type.src, edge_type.name, edge_type.dst].edge_type = e_type
                data.predicates2id = predicates2id

            for feature_name, feature_tensor in edge_store._features.items():
                data[edge_type.src, edge_type.name, edge_type.dst][feature_name] = feature_tensor

        return data

    def get_nodes(self, type : str) -> NodeStore:
        return self._nodes[type]
    
    def get_edges(self, type : EdgeType) ->  EdgeStore:
        return self._edges[type]

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