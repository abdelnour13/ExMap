import torch
import random
import numpy as np
from typing import Any, Union, Tuple
from scipy.sparse import csgraph, csr_matrix
from torch import Tensor
from torch.nn import Module
from torch_sparse import SparseTensor
from torch_geometric.data import HeteroData, Data

def seed_everything(seed : int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_similarity_graph( 
    x : Tensor, 
    th : float,
    batch_size : int = 1024
) -> Tuple[Tensor, Tensor]:

    edge_index = []
    edge_weight = []

    for i, batch_i in enumerate(x.split(batch_size)):

        for j, batch_j in enumerate(x.split(batch_size)):

            if i < j:
                continue

            S = batch_i @ batch_j.T

            row, col = torch.argwhere(S >= th).t()

            weights = S[row, col]
            row = row + i * batch_size
            col = col + j * batch_size

            edge_index.append(torch.stack([row, col], dim=0))
            edge_weight.append(weights)

    edge_index = torch.cat(edge_index, dim=-1)
    edge_weight = torch.cat(edge_weight, dim=-1)

    return edge_index, edge_weight

def torch_connected_components(edge_index : Tensor, num_nodes : int) -> Tensor:
    
    row, col = edge_index.cpu().numpy()
    data = np.ones_like(row)

    adj = csr_matrix(
        (data, (row, col)),
        shape=(num_nodes, num_nodes),
        dtype=int
    )

    _, labels = csgraph.connected_components(adj, directed=False)

    return torch.tensor(labels)

def model_size_mb(model : Module) -> float:

    param_size = 0

    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0

    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2

    return size_all_mb

def move_data_to_device(
    data : Any,
    device : Union[str,torch.device]                 
) -> Any:
        
    if isinstance(data, (Tensor,SparseTensor,Data,HeteroData)):
        return data.to(device)
    
    if isinstance(data, list):
        return [move_data_to_device(item, device) for item in data]
    
    if isinstance(data, tuple):
        return tuple([move_data_to_device(item, device) for item in data])
    
    if isinstance(data, dict):

        return {
            key : move_data_to_device(item, device) 
            for key,item in data.items()
        }

    raise Exception(f"{type(data)} is not supported.")

def create_toy_graph(
    num_nodes : int,
    num_edges : int,
    num_types : int,
    node_features : int,
    edge_features : int,
    type_features : int
) -> dict:
    
    x = torch.randn(num_nodes, node_features)
    e = torch.randn(num_edges, edge_features)
    r = torch.randn(num_types, type_features)

    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_type = torch.randint(0, num_types, (num_edges,))

    return {
        "x" : x,
        "e" : e,
        "r" : r,
        "edge_index" : edge_index,
        "edge_type" : edge_type
    }