import torch
from torch import Tensor
from src import helpers
from torch_geometric import utils as gnn_utils
from torch_geometric.transforms import RemoveIsolatedNodes
from src.shared.command_tools import BaseCommand
from pydantic import BaseModel
from typing import List, Dict
from src.commands.feature_extraction.graph_utils import Graph, EdgeType

class FilterConfig(BaseModel):
    min_entities_per_article : int = 1
    min_articles_per_entity : int = 1
    largest_connected_component : bool = True

def _exclude_keys(d : Dict, keys : List[str]) -> Dict:

    return {
        k : v
        for k, v in d.items()
        if k not in keys
    }

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

            ### Articles connected to at least one node
            numentities_per_article = gnn_utils.degree(
                graph_geo['Entity', 'APPEARIN', 'Article'].edge_index[1],
                num_nodes=len(graph_geo['Article'].node_id)
            )

            article_mask = (numentities_per_article >= self.filter.min_entities_per_article)

            ### Articles connected to at least one label
            # if self.filter.require_subdomain:
            #     numsubdomains_per_article = gnn_utils.degree(graph_geo['Article', 'BELONGSTO', 'SubDomain'].edge_index[0])
            #     article_mask = article_mask & (numsubdomains_per_article != 0)

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

            graph_geo = graph_geo.subgraph({ 
                'Article' : article_mask, 
                'Entity' : entity_mask 
            })

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

            subset._nodes[node_type]._features = _exclude_keys(graph_geo[node_type], ['node_id'])

            for k, v in subset._nodes[node_type]._features.items():
                print(f"{node_type}.{k} {v.shape}")

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

            subset._edges[edge_type]._features = _exclude_keys(
                graph_geo[edge_type.src, edge_type.name, edge_type.dst], 
                ['edge_id', 'edge_index']
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