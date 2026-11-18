import torch
from torch import Tensor
from src.helpers.torch_utils import TensorDict

def _score(subset : TensorDict) -> Tensor:
    return subset['scores']

def _is_valid(s : Tensor, th : float) -> bool:

    # if torch.sum(s <= th) <= 1:
    #     return False
    
    return True

def _eliminate(
    s : Tensor,
    th : float | None,
    k : int | None
) -> Tensor:

    if k is None and th is None:
        return torch.zeros_like(s, dtype=torch.bool)    # eliminate nothing

    mask = torch.ones_like(s, dtype=torch.bool)         # True = eliminate

    if k is not None:
        topk_mask = torch.zeros_like(mask)
        topk_mask[s.topk(k=min(k, s.numel())).indices] = True
        mask &= topk_mask

    if th is not None:
        mask &= (s >= th)

    return mask

def select_representative_nodes(
    cluster : TensorDict,
    max_num_representatives : int,
    k : int | None,
    th_eliminate : int | None,
    th_merge : int | None
) -> Tensor:

    indices = torch.arange(cluster['x'].size(0), device=cluster['x'].device)
    unvisited = torch.ones_like(indices, dtype=torch.bool)
    selected = torch.zeros_like(indices, dtype=torch.bool)

    while (unvisited.sum().item() > 0) and (selected.sum().item() < max_num_representatives):

        scores = _score(cluster)
        idx = scores.argmax().item()

        x = cluster['x']
        s = x[[idx],:] @ x.t()
        s = torch.flatten(s)

        if not _is_valid(s, th_eliminate):
            unvisited[indices[idx]] = False 
        else:
            k = k+1 if k is not None else None
            mask = _eliminate(s, th_merge, k)
            unvisited[indices[idx]] = False
            unvisited[indices[mask]] = False
            selected[indices[idx]] = True

        cluster = cluster[unvisited[indices]]
        indices = indices[unvisited[indices]]

    return selected