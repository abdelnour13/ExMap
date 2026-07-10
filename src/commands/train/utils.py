import torch
from torch_geometric import utils as gnn_utils
from torch import Tensor
from torch.nn import functional as F
from torch_sparse import SparseTensor
from torch_scatter import scatter
from typing import Tuple, Literal

def cosine_similarity_rec_loss(
    a: Tensor,
    b: Tensor,
    batch_size : int
) -> Tensor:
    
    a = a[:batch_size]
    b = b[:batch_size]

    a = F.normalize(a, dim=1)
    b = F.normalize(b, dim=1)

    A = a @ a.T
    B = b @ b.T

    return F.mse_loss(A, B)

def structure_rec_loss(
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

def infonce_agreement_loss(za : Tensor, ze : Tensor, temperature : float):

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

    l1 = a.log() @ b.T
    l2 = (1 - a).log() @ (1 - b).T

    return - (l1 + l2)

def bce_agreement_loss(
    a : Tensor,
    b : Tensor
) -> Tensor:
    
    a_prob = a.sigmoid()
    b_prob = b.sigmoid()

    return (bc(a_prob, b_prob) + bc(b_prob, a_prob)) / 2

def entropy(p : Tensor) -> Tensor:
    return - torch.sum(p * torch.log(p + 1e-6), dim=-1)

def l1_sparisty_loss(h : Tensor) -> Tensor:
    return h.sigmoid().sum()

def sharpness_loss(h : Tensor) -> Tensor:

    h_prob = h.sigmoid()
    h_entropy = -(h_prob * h_prob.log() + (1 - h_prob) * (1 - h_prob).log())

    return h_entropy.mean()

def top_k_operator(h : Tensor, rate : float = 0.1) -> Tuple[Tensor, Tensor]:

    k = int(rate * h.shape[0])

    indices = h.topk(k=k,dim=-1).indices
    mask = torch.zeros_like(h)
    row = torch.arange(h.size(0)).repeat(4)
    col = indices.T.reshape(-1)
    mask[row, col] = 1.0

    return h * mask, mask


_REGISTRY = {
    'cosine_similarity_rec_loss' : cosine_similarity_rec_loss,
    'structure_rec_loss' : structure_rec_loss,
    'margin_structure_loss' : margin_structure_loss,
    'infonce_agreement_loss' : infonce_agreement_loss,
    'bce_agreement_loss' : bce_agreement_loss,
    'l1_sparisty_loss' : l1_sparisty_loss,
    'sharpness_loss' : sharpness_loss,
    'top_k_operator' : top_k_operator
}

FuncType = Literal[
    'cosine_similarity_rec_loss',
    'structure_rec_loss',
    'margin_structure_loss',
    'infonce_agreement_loss',
    'bce_agreement_loss',
    'l1_sparisty_loss',
    'sharpness_loss',
    'top_k_operator'
]

def call_with_args(func_type : FuncType, **kwargs) -> Tensor:

    func = _REGISTRY.get(func_type)

    if func is None:
        raise Exception(f"Unknown function '{func_type}'.")
    
    return func(**kwargs)