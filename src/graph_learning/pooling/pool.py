import torch
from torch import nn, Tensor
from typing import Tuple, Optional
from torch_scatter import scatter
from torch_geometric import utils as gnn_utils

def _tuple(ee : Tensor | Tuple[Tensor, Tensor]) -> Tuple[Tensor, Optional[Tensor]]:

    if isinstance(ee, Tensor):
        return ee, None

    assert isinstance(ee, tuple)
    return ee

def _get_mapping(indices : Tensor) -> Tensor:

    uniques : Tensor = torch.unique(indices)
    N : int = uniques.max().item() + 1
    mapping = torch.empty(N).fill_(-1).long()
    mapping[uniques] = torch.arange(len(uniques))
    mapping = mapping.to(indices.device)

    return mapping

def _reindex(indices : Tensor, mapping : Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:

    if mapping is None:
        mapping = _get_mapping(indices)

    return mapping[indices], mapping

def ee_edge_pool(
    edge_index : Tensor,
    e_group : Tensor,
    num_nodes : int,
    edge_attr  : Optional[Tensor] = None,
    mapping : Optional[Tensor] = None,
    reduce : str = 'mean'
) -> Tuple[Tensor, Optional[Tensor]]:

    src, dst = edge_index
    e_group = _reindex(e_group , mapping)

    src = e_group[src]
    dst = e_group[dst]
    edge_index = torch.stack([src, dst], dim=0).to(edge_index.device)

    edge_index, edge_attr = gnn_utils.coalesce(
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=num_nodes,
        reduce=reduce
    )

    edge_index, edge_attr = gnn_utils.remove_self_loops(
        edge_index=edge_index, 
        edge_attr=edge_attr
    )

    return edge_index, edge_attr

def ea_edge_pool(
    edge_index : Tensor,
    e_group : Tensor,
    mapping : Optional[Tensor] = None,
) -> Tensor:

    src, dst = edge_index
    e_group = _reindex(e_group , mapping)

    src = e_group[src]
    edge_index = torch.stack([src, dst], dim=0)

    return edge_index

def pool(
    x : Tensor,
    ee : Tensor | Tuple[Tensor, Tensor],
    ea : Optional[Tensor],
    e_group : Tensor,
    reduce : str
) -> Tuple[Tensor, Tuple[Tensor, Optional[Tensor]], Tensor]:
 
    mapping = _get_mapping(e_group)
    ee_index, ee_attr = _tuple(ee)

    ee_index, ee_attr = ee_edge_pool(
        edge_index=ee_index,
        e_group=e_group,
        num_nodes=x.size(0),
        edge_attr=ee_attr,
        mapping=mapping,
        reduce=reduce
    )

    if ea is not None:

        ea = ea_edge_pool(
            edge_index=ea,
            e_group=e_group,
            mapping=mapping
        )

    x = scatter(
        src=x,
        index=e_group,
        dim=0,
        dim_size=mapping[e_group],
        reduce=reduce
    )

    return x, (ee_index, ee_attr), ea