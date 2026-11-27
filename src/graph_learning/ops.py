import torch
from torch import Tensor
from typing import Tuple, Optional, Callable
from torch.nn import functional as F
from torch_scatter import scatter
from torch_geometric import utils as gnn_utils
from torch_sparse import SparseTensor

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

def to_sparse_matrix(S : Tensor, mask : Tensor) -> SparseTensor:

    row, col = mask.argwhere().t()
    value = S[row, col]

    return SparseTensor(
        row=row,
        col=col,
        value=value
    )

def entropy(
    x : Tensor, 
    dim : int = -1,
    binary :  bool = False,
    eps : float = 1e-6
) -> Tensor:

    if binary:
        return -(x * (x + eps).log() + (1 - x) * (1 - x + eps).log())

    return -torch.sum(x * (x + eps).log(), dim=dim)

def dist(x: Tensor, y: Tensor) -> Tensor:

    norm_x = torch.sum(x ** 2, dim=-1)
    norm_y = torch.sum(y ** 2, dim=-1)

    dot = x @ y.T

    D = (norm_x[:, None] + norm_y[None, :]) - 2 * dot
    D = D.clamp(min=1e-16).sqrt()

    return D

def min_max_norm(x : Tensor, dim : Optional[int] = None, eps : float = 1e-8) -> Tensor:

    if dim is None:
        return (x - x.min()) / (x.max() - x.min() + eps)

    _min = x.min(dim=dim, keepdim=True).values
    _max = x.max(dim=dim, keepdim=True).values

    return (x - _min) / (_max - _min + eps)

def cos_sim(x : Tensor, y : Tensor, norm : bool = True) -> Tensor:

    if norm:
        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)

    return x @ y.T

def STE(
    codebook : Tensor,
    x : Tensor,
    assignments : Tensor,
    distance : Callable[[Tensor, Tensor], Tensor] = F.mse_loss
) -> Tuple[Tensor, Tensor, Tensor]:

    indices = assignments.argmax(dim=-1)
    q = codebook[indices]

    x_new = x + (q - x).detach()

    codebook_loss = distance(q, x.detach())
    commitement_loss = distance(q.detach(), x)

    return x_new, codebook_loss, commitement_loss

def softSTE(
    codebook : Tensor,
    x : Tensor,
    temp : Tensor,
    assignments : Tensor,
    distance = F.mse_loss
) -> Tuple[Tensor, Tensor, Tensor]:

    assignments = (assignments / temp).softmax(dim=-1) # (N, K)
    q = assignments @ codebook # (N, K) x (K, D) => (N, D)

    codebook_loss = distance(q, x.detach())
    commitement_loss = distance(q.detach(), x)

    x_new = x + (q - x).detach()

    return x_new, codebook_loss, commitement_loss

def softresSTE(
    codebooks : Tensor,
    x : Tensor,
    assignments : Tensor,
    distance = F.mse_loss,
) -> Tuple[Tensor, Tensor, Tensor]:

    h = x
    q = torch.zeros_like(x)

    for codebook, logits in zip(codebooks, assignments):

        p = logits.softmax(dim=-1)
        h_quan = p @ codebook
        
        h = h - h_quan
        q = q + h_quan

    codebook_loss = distance(q, x.detach())
    commitement_loss = distance(q.detach(), x)

    x_new = x + (q - x).detach()

    return x_new, codebook_loss, commitement_loss


def topkSTE(
    codebook : Tensor,
    assignments : Tensor,
    x : Tensor,
    k : int = 5,
    weighted : bool = False,
    distance = F.mse_loss
) -> Tuple[Tensor, Tensor, Tensor]:

    topk = assignments.topk(k=k, dim=-1, largest=False, sorted=True)
    values, indices = topk.values, topk.indices

    q = codebook[indices,:] # N x K x D

    if weighted:
        q = q * values[:,:,None].softmax(dim=1)
        q = q.sum(dim=1) # N x D
    else:
        q = q.mean(dim=1) # N x D

    x_new = x + (q - x).detach()
        
    codebook_loss = distance(q, x.detach())
    commitement_loss = distance(q.detach(), x)

    return x_new, codebook_loss, commitement_loss

def mSTE(
    codebook : Tensor,
    assignments : Tensor,
    x : Tensor,
    threshold : float = 0.0, # \sigmoid(0.0) = 0.5
    weighted : bool = False
) -> Tuple[Tensor, Tensor, Tensor]:

    datapoints, topics = assignments.gt(threshold).argwhere().t()

    not_included = assignments \
        .gt(threshold).float() \
        .sum(dim=-1).eq(0)     \
        .argwhere().flatten().to(x.device)

    if len(not_included) > 0:
        datapoints = torch.cat([datapoints, not_included])
        topics = torch.cat([topics, assignments[not_included,:].argmax(dim=-1)])

    weights = assignments[datapoints, topics]

    if weighted:

        weights = gnn_utils.softmax(
            src=weights[:,None],
            index=datapoints,
            num_nodes=x.size(0),
            dim=0
        )

        q = scatter(
            src=codebook[topics,:] * weights,
            index=datapoints,
            dim=0,
            dim_size=x.size(0),
            reduce='sum'
        )

    else:

        q = scatter(
            src=codebook[topics,:],
            index=datapoints,
            dim=0,
            dim_size=x.size(0),
            reduce='mean'
        )

    x_new = x + (q - x).detach()
    
    codebook_loss = F.mse_loss(q, x.detach())
    commitement_loss = F.mse_loss(q.detach(), x)

    return x_new, codebook_loss, commitement_loss