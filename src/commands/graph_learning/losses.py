import torch
from torch import Tensor
from torch.nn import functional as F
from torch_geometric import utils as gnn_utils
from torch_sparse import SparseTensor
from torch_scatter import scatter
from typing import Tuple, Literal, Optional

### Utils ###

def _select_batch_size(x : Tensor, batch_size : Optional[int]) -> Tensor:

    if batch_size is None:
        return x
    
    return x[:,None]

### Features Reconstruction Losses ###

def mse_rec_loss(
    a : Tensor,
    b : Tensor,
    batch_size : Optional[int]
) -> Tensor:
    
    a = _select_batch_size(a, batch_size)
    b = _select_batch_size(b, batch_size)

    return F.mse_loss(a, b)

def cosine_similarity_rec_loss(
    a: Tensor,
    b: Tensor,
    batch_size : Optional[int]
) -> Tensor:
    
    a = _select_batch_size(a, batch_size)
    b = _select_batch_size(b, batch_size)

    a = F.normalize(a, dim=1)
    b = F.normalize(b, dim=1)

    A = a @ a.T
    B = b @ b.T

    return F.mse_loss(A, B)

### Structure Reconstruction Losses ###

def bce_structure_rec_loss(
    a: Tensor,
    edge_index: Tensor,
) -> Tensor:
    
    N = a.shape[0]
    
    row, col = edge_index

    A = a @ a.T
    B = SparseTensor(row=row, col=col).to_dense().to(A.device)

    L = F.binary_cross_entropy_with_logits(A, B, reduction="none")

    l_pos = L[row, col].mean()
    L[row, col] = 0.0

    l_neg = L.sum() / (N*N - len(row))

    return (l_pos + l_neg) / 2

def margin_structure_loss(
    x : Tensor,
    edge_index : Tensor,
    num_neg : int,
    margin : float
) -> Tensor:
    row, col = edge_index

    x = F.normalize(x, dim=1)

    # positive scores
    pos_sim = (x[row] * x[col]).sum(dim=-1)

    pos_sim = scatter(
        pos_sim,
        index=row,
        dim=0,
        dim_size=x.size(0),
        reduce='min'
    )

    n_row, n_col = gnn_utils.negative_sampling(
        edge_index=edge_index,
        num_nodes=x.size(0),
        num_neg_samples=len(pos_sim) * num_neg
    )

    neg_sim = (x[n_row] * x[n_col]).sum(dim=-1)

    L = F.relu(margin - pos_sim[n_row] + neg_sim)

    return L.mean()

### Constrastive Losses ###

def infonce_agreement_loss(za : Tensor, ze : Tensor, temperature : float = 0.1):

    # normalize embeddings
    za = F.normalize(za, dim=1)
    ze = F.normalize(ze, dim=1)

    # similarity matrix
    logits = za @ ze.T

    # temperature scaling
    logits = logits / temperature

    # positives are diagonal entries
    labels = torch.arange(za.size(0), device=za.device)

    # symmetric InfoNCE
    loss_a2e = F.cross_entropy(logits, labels)
    loss_e2a = F.cross_entropy(logits.T, labels)

    loss = (loss_a2e + loss_e2a) / 2

    return loss

def bc(a : Tensor, b : Tensor) -> Tensor:

    # a = a.sigmoid()
    # b = b.sigmoid()

    eps = 1e-6

    L1 = (a + eps).log() @ b.T
    L2 = (1 - a + eps).log() @ (1 - b).T

    L = - (L1 + L2)

    # print(a)
    # print(b)
    # print(L)

    l_pos = torch.diag(L).mean()
    l_neg = L.fill_diagonal_(0.0).mean()

    return l_pos - l_neg

def bce_agreement_loss(
    a : Tensor,
    b : Tensor
) -> Tensor:
    return (bc(a, b) + bc(b, a)) / 2

### Sharpness Loss ###

def entropy(p : Tensor) -> Tensor:
    return - torch.sum(p * torch.log(p + 1e-6), dim=-1)

def sharpness_loss(h : Tensor) -> Tensor:

    h_prob = h
    h_entropy = -(h_prob * h_prob.log() + (1 - h_prob) * (1 - h_prob).log())

    return h_entropy.mean()

def entropy(p : Tensor) -> Tensor:
    return -torch.sum(p * torch.log(p + 1e-6), dim=-1)

def e_loss(p : Tensor) -> Tensor:

    q = p / (p.sum() + 1e-6)

    e_local = entropy(q).mean()
    e_global = entropy(q.mean(dim=0))

    return e_local - e_global

### Sparsity Loss ###

def l1_sparisty_loss(h : Tensor) -> Tensor:
    return h.mean()

def soft_topk_sparsity_loss(
    h : Tensor, 
    rate : float
) -> Tensor:
    return F.softplus(h.mean(dim=-1) - rate).mean()

def top_k_operator(h : Tensor, rate : float) -> Tuple[Tensor, Tensor]:

    k = int(rate * h.shape[0])

    indices = h.topk(k=k,dim=-1).indices
    mask = torch.zeros_like(h)
    row = torch.arange(h.size(0)).repeat(h.size(1))
    col = indices.T.reshape(-1)
    mask[row, col] = 1.0

    return h * mask, mask

### Losses Objectives ###

def features_rec_loss(
    a1 : Tensor,
    a2 : Tensor,
    batch_size : Optional[int],
    variant : Literal['mse', 'cosine']
) -> Tensor:
    
    match variant:

        case 'mse':
            return mse_rec_loss(a1, a2, batch_size=batch_size)
        case 'cosine':
            return cosine_similarity_rec_loss(a1, a2, batch_size=batch_size)
        case _:
            raise Exception(f"Unknown loss '{variant}'")
        
def structure_rec_loss(
    a : Tensor,
    edge_index : Tensor,
    num_neg : Optional[int],
    margin : Optional[float],
    variant : Literal['bce', 'margin']
) -> Tensor:
    
    match variant:

        case 'bce':
            return bce_structure_rec_loss(a, edge_index=edge_index)
        case 'cosine':
            return margin_structure_loss(a, edge_index, num_neg=num_neg, margin=margin)
        case _:
            raise Exception(f"Unknown loss '{variant}'")

def agreement_loss(
    a1 : Tensor, 
    a2 : Tensor,
    variant : Literal['bce', 'infonce'],
    apply_topk_operator : bool,
    topk_rate : Optional[float]
) -> Tensor:
        
    if apply_topk_operator:
        a1 = top_k_operator(a1, topk_rate)
        a2 = top_k_operator(a1, topk_rate)

    match variant:
        case 'bce':
            loss = bce_agreement_loss(a1, a2)
        case 'infonce':
            loss = infonce_agreement_loss(a1, a2)
        case _:
            raise Exception(f"Unknown loss '{variant}'")
        
    return loss

def sparisty_loss(
    h : Tensor, 
    topk_rate : Optional[float],
    variant : Literal['l1', 'soft_topk']
) -> Tensor:
    
    match variant:

        case 'l1':
            loss = l1_sparisty_loss(h)
        case 'soft_topk':
            loss = soft_topk_sparsity_loss(h, topk_rate)
        case _:
            raise Exception(f"Unknown loss '{variant}'")
        
    return loss

def diversity_loss(h : Tensor) -> Tensor:

    h = F.normalize(h, p=2, dim=1)
    S = h @ h.t()

    return S.fill_diagonal_(0.0).mean()