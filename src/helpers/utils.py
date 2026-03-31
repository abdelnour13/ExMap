import json
import torch
import asyncio
import numpy as np
from typing import List, Dict, Any, TypeVar, Union, Optional
from openai import AsyncOpenAI, RateLimitError, APIError
from returns.result import Result
from dataclasses import fields
from torch import Tensor
from scipy.sparse import csgraph, csr_matrix
from torch_sparse import SparseTensor
from torch_geometric.data import Data
from src.gnn import ArticleAsKG

DS = TypeVar('DS')

def load_json_file(path : str) -> List[Dict] | Dict[str, Any]:

    with open(path, "r") as f:
        return json.load(f)
    
async def make_openai_request(
    client : AsyncOpenAI,
    model : str,
    messages : List[Dict],
    max_attempts : int = 3
):

    for attempt in range(max_attempts):

        try:
            
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,  # No randomness
                top_p=1,  # Disable sampling
                seed=42,  # Seed
            )

            return Result.from_value(response)
        
        except (RateLimitError, APIError) as e:

            if attempt == max_attempts - 1:
                return Result.from_failure(e)

            wait = 2 ** (0.8 * attempt)
            await asyncio.sleep(wait)

def asdict_shallow(obj : DS) -> Dict:

    return {
        field.name : getattr(obj, field.name)
        for field in fields(obj)
    }

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

def override_dict(arg1 : dict, arg2 : dict) -> dict:

    result = {}

    for key, value in arg1.items():

        if key in arg2:

            if not isinstance(value, dict):
                result[key] = arg2[key]
            elif not isinstance(arg2[key], dict):
                result[key] = arg2[key]
            else:
                result[key] = override_dict(value, arg2[key])

        else:

            result[key] = value

    for key, value in arg2.items():

        if key not in arg1:
            result[key] = value

    return result

def torch_connected_components(edge_index : Tensor, num_nodes : int) -> Tensor:
    
    row, col = edge_index.numpy()
    data = np.ones_like(row)

    adj = csr_matrix(
        (data, (row, col)),
        shape=(num_nodes, num_nodes),
        dtype=int
    )

    _, labels = csgraph.connected_components(adj, directed=False)

    return torch.tensor(labels)

def model_size_mb(model : torch.nn.Module) -> float:

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
        
    if isinstance(data, (Tensor,SparseTensor,Data,ArticleAsKG)):
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