import torch
from torch import Tensor
from src import helpers
from torch_geometric import utils as gnn_utils
from torch_geometric.data import HeteroData
from torch_geometric.transforms import RemoveIsolatedNodes
from src.shared.command_tools import BaseCommand
from pydantic import BaseModel
from typing import List, Dict
from torch_scatter import scatter
from src.commands.feature_extraction.graph_utils import Graph, EdgeType, EdgeTypes, NodeTypes

class FilterConfig(BaseModel):
    require_subdomain : bool = False
    min_entities_per_article : int = 1
    min_articles_per_entity : int = 1
    largest_connected_component : bool = False

def _exclude_keys(d : Dict, keys : List[str]) -> Dict:

    return {
        k : v
        for k, v in d.items()
        if k not in keys and isinstance(v, Tensor)
    }

class GraphFilter:

    def __init__(self, 
        _host : BaseCommand,
        filter : FilterConfig,
        label_name : str
    ) -> None:
        self._host = _host
        self.filter = filter
        self.label_name = label_name

    def require_subdomain(self, graph : HeteroData) -> Tensor:
    
        y = graph['Article'][self.label_name]
        y = y / y.max()
        y = y.gt(0.5)

        article_mask = (y.sum(dim=1) > 0)

        return article_mask

    def min_entities_per_article(self, graph : HeteroData, n : int) -> Tensor:

        num_entities_per_article = gnn_utils.degree(
            graph['Entity', 'APPEARIN', 'Article'].edge_index[1],
            num_nodes=len(graph['Article'].node_id)
        )

        article_mask = (num_entities_per_article >= n)

        return article_mask

    def min_articles_per_article(self, graph : HeteroData, n : int) -> Tensor:
    
        num_articles_per_entity = gnn_utils.degree(
            graph['Entity', 'APPEARIN', 'Article'].edge_index[0],
            num_nodes=len(graph['Entity'].node_id)
        )

        entity_mask = (num_articles_per_entity >= n)

        return entity_mask

    def lcc(self, graph : HeteroData) -> Tensor:

        component = helpers.torch_connected_components(
            graph['Entity', 'HASRELATIONWITH', 'Entity'].edge_index,
            len(graph['Entity'].node_id)
        )

        unique_components, counts = torch.unique(component, return_counts=True)
        lcc = unique_components[counts == counts.max()][0].item()

        entity_mask = (component == lcc)

        return entity_mask

    def remove_isolated_nodes(self, graph: HeteroData) -> HeteroData:

        T = NodeTypes.Triple
        E = NodeTypes.Entity

        for n_type in graph.node_types:
            graph[n_type].num_nodes = len(graph[n_type].node_id)

        ### Every node starts alive and can only die (masks only shrink),
        ### so the loop always terminates and comparing counts is a valid stop test.
        mask = {
            n_type: torch.ones_like(graph[n_type].node_id, dtype=torch.bool)
            for n_type in graph.node_types
        }

        counts = {
            n_type: int(mask[n_type].sum())
            for n_type in graph.node_types
        }

        ### Node types linked to Triples: must keep >= 1 valid Triple
        linked_to_trp = {
            n
            for src, _, dst in graph.edge_types
            if T in (src, dst)
            for n in (src, dst)
            if n != T
        }

        ### Node types with at least one non-Triple edge type:
        ### must keep >= 1 alive non-Triple neighbour
        linked_to_other = {
            n
            for src, _, dst in graph.edge_types
            if T not in (src, dst)
            for n in (src, dst)
        }

        trp1_id, h_id = graph[EdgeTypes.Triple_HASHEAD_Entity.as_tuple()].edge_index
        trp2_id, t_id = graph[EdgeTypes.Triple_HASTAIL_Entity.as_tuple()].edge_index

        stop = False
        while not stop:

            ### Recomputed every pass from the current alive nodes
            local_mask = {  # >= 1 alive non-Triple neighbour
                n_type: torch.zeros_like(graph[n_type].node_id, dtype=torch.bool)
                for n_type in graph.node_types
            }
            t_mask = {  # >= 1 alive Triple (OR over all Triple edge types)
                n_type: torch.zeros_like(graph[n_type].node_id, dtype=torch.bool)
                for n_type in linked_to_trp
            }

            ### Iterate over edges
            for (src, rel, dst) in graph.edge_types:

                row, col = graph[(src, rel, dst)].edge_index

                if src != T:
                    src_degree = scatter(
                        src=mask[dst][col].long(),
                        index=row,
                        dim=0,
                        dim_size=graph[src].num_nodes,
                        reduce="sum",
                    )
                    if dst != T:
                        local_mask[src] |= src_degree.gt(0)
                    else:
                        t_mask[src] |= src_degree.gt(0)

                if dst != T:
                    dst_degree = scatter(
                        src=mask[src][row].long(),
                        index=col,
                        dim=0,
                        dim_size=graph[dst].num_nodes,
                        reduce="sum",
                    )
                    if src != T:
                        local_mask[dst] |= dst_degree.gt(0)
                    else:
                        t_mask[dst] |= dst_degree.gt(0)

            ### Non-Triple nodes: apply only the conditions that exist for their type
            new_mask = {}

            for n_type in graph.node_types:

                if n_type == T:
                    continue

                m = mask[n_type]

                if n_type in linked_to_other:
                    m = m & local_mask[n_type]

                if n_type in linked_to_trp:
                    m = m & t_mask[n_type]

                new_mask[n_type] = m

            ### Triples: exactly one head and one tail, both must still be alive
            M1 = torch.zeros_like(mask[T], dtype=torch.bool)
            M1[trp1_id] = new_mask[E][h_id]

            M2 = torch.zeros_like(mask[T], dtype=torch.bool)
            M2[trp2_id] = new_mask[E][t_id]

            new_mask[T] = mask[T] & M1 & M2

            mask = new_mask

            ### Stopping criterion
            new_counts = {
                n_type: int(m.sum())
                for n_type, m in mask.items()
            }

            stop = all(
                new_counts[n_type] == counts[n_type]
                for n_type in graph.node_types
            )

            counts = new_counts

        return graph.subgraph(mask)

    def subset(self, graph_geo : HeteroData, graph : Graph) -> Graph:

        subset = Graph()

        ### Copy Nodes
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

        ### Copy Edges
        for edge_type in graph_geo.edge_types:

            edge_ids = graph_geo[edge_type].edge_id
            src, type_, dst = edge_type
            edge_type = EdgeType(src, dst, type_)

            if edge_type not in graph.edge_types:
                continue

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

        ### Log New Stats
        for node_type, node_store in subset._nodes.items():

            self._host.send_info_event(
                progress=0.0,
                task="graph-filtering",
                message=f"[STAGE 2] count({node_type})={len(node_store._nodes)}..."
            )

        for edge_type, edge_store in subset._edges.items():

            self._host.send_info_event(
                progress=0.0,
                task="graph-filtering",
                message=f"[STAGE 2] count({edge_type.src}_{edge_type.name}_{edge_type.dst})={len(edge_store._edges)}..."
            )

        self._host.send_info_event(
            progress=1.0,
            task="graph-filtering",
            message=f"[STAGE 2] completed."
        )

        return subset

    def filter_graph(self, graph : Graph) -> Graph:

        stop = False
        attempt = 0

        node_degrees = {
            type_ : len(graph.pygeo[type_].node_id)
            for type_ in graph.pygeo.node_types
        }

        graph_geo = graph.pygeo
        
        while not stop:

            self._host.send_info_event(
                progress=0.0,
                task="graph-filtering",
                message=f"[STAGE 2] Graph Filtering attmpt={attempt}..."
            )


            article_mask = self.min_entities_per_article(
                graph_geo, 
                self.filter.min_entities_per_article
            )

            if self.filter.require_subdomain:
                article_mask = article_mask & self.require_subdomain(graph_geo)

            entity_mask = self.min_articles_per_article(
                graph_geo,
                self.filter.min_articles_per_entity
            )

            entity_mask = entity_mask & self.lcc(graph_geo)

            graph_geo = graph_geo.subgraph({ 
                'Article' : article_mask, 
                'Entity' : entity_mask 
            })

            graph_geo = self.remove_isolated_nodes(graph_geo)

            ### Stopping critera
            new_node_degrees = {
                type_ : len(graph_geo[type_].node_id)
                for type_ in graph_geo.node_types
            }

            stop = all([value == new_node_degrees[key] for key, value in node_degrees.items()])
            node_degrees = new_node_degrees
            attempt += 1

        ### Constructing new graph
        self._host.send_info_event(
            progress=0.0,
            task="graph-filtering",
            message=f"[STAGE 2] Constructing new graph..."
        )

        subset = self.subset(graph_geo, graph)

        self._host.send_info_event(
            progress=1.0,
            task="graph-filtering",
            message=f"[STAGE 2] completed."
        )

        return subset