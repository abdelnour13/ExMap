import json
import torch
import asyncio
from typing import List, Dict, Any, TypeVar
from openai import AsyncOpenAI, RateLimitError, APIError
from returns.result import Result
from dataclasses import fields

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