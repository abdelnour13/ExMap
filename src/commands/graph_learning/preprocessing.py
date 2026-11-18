import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric import utils as gnn_utils
from torch_sparse import SparseTensor
from typing import List, Optional, Literal, Tuple
from pydantic import BaseModel, Field

def _remove_self_loops(batch : HeteroData) -> HeteroData:

    edge_index, edge_type = gnn_utils.remove_self_loops(
        batch['Article', 'relate-to', 'Article'].edge_index,
        batch['Article', 'relate-to', 'Article'].edge_type
    )

    batch['Article', 'relate-to', 'Article'].edge_index = edge_index
    batch['Article', 'relate-to', 'Article'].edge_type = edge_type

    return batch

class NodeAugmentationArgs(BaseModel):
    node_type : str
    p : float = 0.2
    mode : Literal['row', 'col', 'all']

class EdgeAugmentationArgs(BaseModel):
    edge_type : Tuple[str, str, str]
    p : float = 0.2

class AugmentationConfig(BaseModel):
    nodes : List[NodeAugmentationArgs] = Field(default_factory=list)
    edges : List[EdgeAugmentationArgs] = Field(default_factory=list)

def _augment(
    batch : HeteroData, 
    config : AugmentationConfig,
    training : bool = True, 
) -> HeteroData:

    if not training:
        return batch

    for conf in config.nodes:

        batch[conf.node_type].x, _ = gnn_utils.mask_feature(
            batch[conf.node_type].x,
            p=conf.p,
            mode=conf.mode
        )

    for conf in config.edges:

        edge_index = batch[conf.edge_type].edge_index
        edge_attr = None

        if 'edge_attr' in batch[conf.edge_type]:
            edge_attr = batch[conf.edge_type].edge_attr

        edge_index, edge_attr = gnn_utils.dropout_adj(
            edge_index=edge_index,
            edge_attr=edge_attr,
            p=conf.p,
            num_nodes=batch[conf.edge_type[0]].node_id.size(0)
        )

        batch[conf.edge_type].edge_index = edge_index
        batch[conf.edge_type].edge_attr  = edge_attr

    return batch

def _subgraph_by_articles(
    data : HeteroData, 
    article_ids : Tensor, 
    save_old_idx : bool = False
) -> HeteroData:

    entities, articles = data['Entity', 'APPEARIN', 'Article'].edge_index

    entity_ids = torch.unique(entities[torch.isin(articles, article_ids)])

    subset = data.subgraph({
        "Article" : article_ids,
        "Entity" : entity_ids
    })

    for n_type in subset.node_types:

        if 'og_node_id' not in subset[n_type]:
            subset[n_type].og_node_id = subset[n_type].node_id

        subset[n_type].node_id = torch.arange(len(subset[n_type].x))

    for e_type in subset.edge_types:

        if 'og_edge_id' not in subset[e_type]:
            subset[e_type].og_edge_id = subset[e_type].edge_id

        subset[e_type].edge_id = torch.arange(len(subset[e_type].edge_index[0]))

    return subset

def _select_nodes(data: HeteroData, n_types: List[str]) -> HeteroData:

    G = HeteroData()

    for n_type in n_types:
        for key, value in data[n_type].items():
            G[n_type][key] = value.clone() if torch.is_tensor(value) else value

    for e_type in data.edge_types:
        if e_type[0] in n_types and e_type[2] in n_types:
            for key, value in data[e_type].items():
                G[e_type][key] = value.clone() if torch.is_tensor(value) else value

    # G.predicates2id = data.predicates2id

    return G

def _add_labels(data : HeteroData, n_types : List[str]) -> HeteroData:

    for type_ in n_types:

        e_type = [e for e in data.edge_types if e[0] == 'Article' and e[2] == type_][0]

        row, col = data[e_type].edge_index
        num_cols = data['Article'].num_nodes
        num_rows = data[type_].num_nodes

        A = SparseTensor(row=row,col=col,sparse_sizes=(num_cols,num_rows))

        data['Article'][f'{type_.lower()}'] = A.to_dense().clone()

    return data

def _add_metapaths(data : HeteroData, types : Optional[List[str]] = None) -> HeteroData:

    types = ['Author'] if types is None else types

    edge_index = []
    edge_type  = []

    for i, type_ in enumerate(types):

        e_type = [
            e 
            for e in data.edge_types if (e[0] == 'Article' and e[2] == type_) or (e[0] == type_ and e[2] == 'Article')
        ][0]

        row, col = data[e_type].edge_index
        n1, _, n2 = e_type
        num_cols = data[n1].num_nodes
        num_rows = data[n2].num_nodes

        A = SparseTensor(row=row,col=col,sparse_sizes=(num_cols,num_rows))
        A = A @ A.t() if n1 == 'Article' else A.t() @ A
        row, col, _ = A.coo()

        edge_index.append(torch.stack([row, col], dim=0))
        edge_type.append(torch.zeros_like(row).fill_(i))

    edge_index = torch.cat(edge_index, dim=-1)
    edge_type = torch.cat(edge_type, dim=-1)

    data['Article', 'relate-to', 'Article'].edge_index = edge_index
    data['Article', 'relate-to', 'Article'].edge_type = edge_type
    data['Article', 'relate-to', 'Article'].edge_id = torch.arange(len(row))

    return data

def _norm_features(data : HeteroData) -> HeteroData:

    for n_type in data.node_types:
        t = data[n_type].x
        mean = t.mean(0, keepdim=True)
        std  = t.std(0,  keepdim=True)
        data[n_type].x = (t - mean) / (std + 1e-3)

    for e_type in data.edge_types:
        if 'x' in data[e_type]:
            t = data[e_type].x
            mean = t.mean(0, keepdim=True)
            std  = t.std(0,  keepdim=True)
            data[e_type].x = (t - mean) / (std + 1e-3)

    return data

def _merge_labels(
    data : HeteroData,
    label : str,
    merges : List[List[str]]
) -> HeteroData:

    label_names : List[str] = data.raw_metadata["labels"][label]
    labels_matrix : Tensor  = data['Article'][label]

    merged_label_names : List[str] = []
    merged_label_matrix : List[Tensor] = []

    for merge in merges:

        label_col = []

        for l in merge:
            idx = label_names.index(l)
            label_col.append(labels_matrix[:,idx])

        label_col = torch.stack(label_col, dim=0).max(dim=0).values
        merged_label_matrix.append(label_col)
        merged_label_names.append('/'.join(merge))

    M = torch.stack(merged_label_matrix, dim=1)
    data.raw_metadata["labels"][label] = data.raw_metadata["labels"][label] + merged_label_names
    data['Article'][label] = torch.cat([labels_matrix, M], dim=-1)

    return data