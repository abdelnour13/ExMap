import torch
import math
from torch import nn as tnn
from torch.nn import functional as F
from src.graph_dl import ops
from src.graph_dl.layers.reduction import reduce
from src.graph_dl.utils import AdjMatrix, EdgeIndex
from torch import nn, Tensor
from torch_scatter import scatter
from torch_geometric import utils as gnn_utils
from typing import Tuple, Optional, Literal, Dict, Callable, List
from dataclasses import dataclass, field
from src.graph_dl.layers.codebook import MultiLevelCodebook, CodebookOutput

POOLING_KIND = Literal['dmon', 'diffpool', 'mincut', 'enad', 'vq']
AUX_LOSS_KIND = Literal['dmon', 'diffpool', 'mincut', 'enad']
UNPOOL_KIND = Literal['restore', 'weighted_restore', 'matmul']

@dataclass
class Input:
    EA : AdjMatrix
    EE : AdjMatrix

@dataclass
class ClusterPoolOutput:
    S : Tensor
    x : Tensor
    EE : AdjMatrix
    EA : Optional[AdjMatrix]
    aux_loss : Optional[Tensor]
    input : Input
    metadata : dict = field(default_factory=dict)

class Pool(nn.Module):

    def pool(
        self,
        x : Tensor,
        EE : AdjMatrix,
        EA : AdjMatrix,
    ) -> ClusterPoolOutput:
        pass

    def unpool(self, 
        x : Tensor,
        EE : AdjMatrix,
        EA : AdjMatrix,
        out : ClusterPoolOutput           
    ) -> ClusterPoolOutput:
        pass


class ClusterPool(nn.Module):

    def __init__(self, 
        nn : nn.Module, 
        in_channels : int,
        kind : AUX_LOSS_KIND,
        unpool_kind : UNPOOL_KIND,
        norm : bool = False
    ) -> None:
        super().__init__()

        self.nn = nn
        self.in_channels = in_channels
        self.kind : AUX_LOSS_KIND = kind
        self.norm = norm
        self.unpool_kind : UNPOOL_KIND = unpool_kind

        _LOSSES : Dict[AUX_LOSS_KIND, Callable[['ClusterPool', Tensor, Tensor], Tensor]] = {
            'dmon' : self.dmon_aux_loss,
            'diffpool' : self.diffpool_aux_loss,
            'mincut' : self.mincut_aux_loss,
            'enad' : self.enad_aux_loss
        }

        self._aux_loss = _LOSSES[kind]

    def aux_loss(self, EE : Tensor, S : Tensor) -> Dict[str, Tensor]:
        return self._aux_loss(EE, S)

    def assign(self, x : Tensor, edge_index : Tensor) -> Tensor:

        S_raw = self.nn(x=x, edge_index=edge_index, edge_attr=None)
        S = torch.softmax(S_raw, dim=-1)

        return S

    def one_hot(self, S : Tensor) -> Tensor:
    
        idx = S.argmax(dim=-1)
        I = torch.eye(n=S.size(-1), device=idx.device, dtype=torch.float)
        S_h = I[idx,:]

        S = S + (S_h - S).detach()

        return S

    def enad_aux_loss(self, EE : Tensor, S : Tensor) -> Dict[str, Tensor]:
        return {}

    def diffpool_aux_loss(self, EE : Tensor, S : Tensor) -> Dict[str, Tensor]:

        lp_loss = torch.norm(EE.detach() - S @ S.t(), p='fro')        # Link Prediction Loss
        entropy_loss = ops.entropy(x=S, dim=-1, binary=False).mean()  # Entropy loss

        return {
            "lp" : lp_loss,
            "entropy" : entropy_loss,
            "aux_loss" : lp_loss - entropy_loss
        }

    def mincut_aux_loss(self, EE : Tensor, S : Tensor) -> Dict[str, Tensor]:

        D = torch.diag(EE.detach().sum(dim=-1))

        a = torch.trace(S.t() @ EE.detach() @ S)
        b = torch.trace(S.t() @ D @ S)

        l1 = - a / b   # Community Detection Term

        CC = S.t() @ S
        CC = CC / torch.norm(CC, p='fro')

        I = torch.eye(S.size(-1)).to(S.device)
        I = I / math.sqrt(S.size(-1))

        l2 = torch.norm(CC - I)  # Collapse Prevention term

        return {
            "community_detection" : l1,
            "collapse_term" : l2,
            "aux_loss" : l1 + l2
        }

    def dmon_aux_loss(self, EE : Tensor, S : Tensor) -> Tensor:

        d = EE.detach().sum(dim=-1, keepdim=True)
        two_m = d.sum().item()
        d = d / d.norm(p=2, dim=0)

        A = S.t() @ EE @ S
        B = d @ d.t()
        B = S.t() @ B @ S

        l1 = torch.trace(A - B) / two_m

        lambda_ = math.sqrt(S.size(-1)) / S.size(0)
        l2 = lambda_* torch.norm(S.sum(dim=0), p='fro') - 1

        return {
            "community_detection" : l1,
            "collapse_term" : l2,
            "aux_loss" : l1 + l2
        }
    
    def node_pool(self, x : Tensor, S : Tensor) -> Tensor:

        eps = 1e-6

        if self.norm:
            S = S / (S.sum(dim=0, keepdim=True) + eps)

        return S.t() @ x

    def edge_pool(self, EE : Tensor, EA : Tensor, S : Tensor) -> Tuple[Tensor, Tensor]:

        EE_pooled = S.t() @ EE @ S # (N'_e, N_e) x (N_e, N_e) x (N_e, N'_e) = (N'_e, N'_e)
        EA_pooled = S.t() @ EA     # (N'_e, N_e) x (N_e, N_a)  = (N'_e, N_a)

        return EE_pooled, EA_pooled

    def pool(
        self,
        x : Tensor,
        EE : AdjMatrix,
        EA : AdjMatrix,
    ) -> ClusterPoolOutput:

        S = self.assign(x, edge_index=EE.edge_index.edge_index) # (N_e, N'_e)

        if self.kind == 'enad':
            S = self.one_hot(S)

        aux_loss = self.aux_loss(EE=EE.dense, S=S)

        x_pooled  = self.node_pool(x=x, S=S)

        EE_pooled, EA_pooled = self.edge_pool(
            EE=EE.dense,
            EA=EA.dense,
            S=S
        )

        return ClusterPoolOutput(
            S=S,
            x=x_pooled,
            EE=AdjMatrix(_dense=EE_pooled),
            EA=AdjMatrix(_dense=EA_pooled),
            aux_loss=aux_loss.get('aux_loss'),
            input=Input(EE=EE, EA=EA),
            metadata=aux_loss 
        )

    def edge_unpool(self, output : ClusterPoolOutput) -> Tuple[AdjMatrix, AdjMatrix]:

        if self.unpool_kind == 'restore':
            EE, EA = output.input.EE, output.input.EA
        elif self.unpool_kind == 'weighted_restore':

            row, col = output.input.EE.edge_index.edge_index
            W = output.S @ output.S.t()
            w = W[row, col]

            EE = AdjMatrix(_edge_index=EdgeIndex(
                row=row,
                col=col,
                num_nodes=output.input.EE.edge_index.num_nodes,
                value=w
            ))

            EA = output.input.EA

        else:

            EE = AdjMatrix(_dense=output.S @ output.EE.dense @ output.S.t())
            EA = output.input.EA

        return EE, EA

    def unpool(self, 
        x : Tensor,
        EE : AdjMatrix,
        EA : AdjMatrix,
        out : ClusterPoolOutput           
    ) -> ClusterPoolOutput:

        S = out.S.t()
        x_unpooled  = self.node_pool(x=x, S=S)

        EE_unpooled, EA_unpooled = self.edge_unpool(output=out)

        return ClusterPoolOutput(
            S=S,
            x=x_unpooled,
            EE=EE_unpooled,
            EA=EA_unpooled,
            aux_loss=None,
            input=Input(EE=EE, EA=EA)
        )

class ENADPool(nn.Module):

    def __init__(self, 
        nn : nn.Module,
        in_channels : int,
        norm : bool = False
    ) -> None:
        super().__init__()
        
        self.nn = nn
        self.in_channels = in_channels
        self.norm = norm

        self.w_m = tnn.Sequential(
            tnn.Linear(in_features=in_channels, out_features=in_channels),
            tnn.LeakyReLU(negative_slope=0.2),
            tnn.Linear(in_features=in_channels, out_features=1),
        )

        self.w_e1 = tnn.Sequential(
            tnn.Linear(in_features=in_channels, out_features=in_channels // 2),
            tnn.LeakyReLU(negative_slope=0.2),
            tnn.Linear(in_features=in_channels // 2, out_features=1),
        )

        self.w_e2 = tnn.Sequential(
            tnn.Linear(in_features=in_channels, out_features=in_channels // 2),
            tnn.LeakyReLU(negative_slope=0.2),
            tnn.Linear(in_features=in_channels // 2, out_features=1),
        )

    def one_hot(self, S : Tensor) -> Tensor:

        idx = S.argmax(dim=-1)
        I = torch.eye(n=S.size(-1), device=idx.device, dtype=torch.float)
        S_h = I[idx,:]

        S = S + (S_h - S).detach()

        return S

    def assign(self, x : Tensor, edge_index : Tensor) -> Tuple[Tensor, Tensor]: 

        if x.isnan().sum() > 0:
            raise Exception("Assign input is nan")

        if not x.isfinite().all():
            raise Exception("Assign input is inf")

        S_raw = self.nn(x, edge_index=edge_index, edge_attr=None)

        if S_raw.isnan().sum() > 0:
            raise Exception("Assign input is nan")

        if not S_raw.isfinite().all():
            raise Exception("Assign input is inf")

        S = torch.softmax(S_raw, dim=-1)

        return S, S_raw

    def attn_node_pool(self, 
        x : Tensor,
        S_h : Tensor
    ) -> Tensor:
        
        attn_weights = self.w_m(x)
        node2cluster = S_h.argwhere().t()

        x_pooled, _, _ = reduce(
            x=x,
            edge_index=node2cluster,
            num_nodes=S_h.size(-1),
            weights=attn_weights,
            kind='attn'
        ).astuple()

        return x_pooled

    def attn_edge_pool(self, 
        x : Tensor,
        S_h : Tensor,
        EE : Tensor,
        EA : Tensor
    ) -> Tensor:

        if S_h.isnan().sum() > 0:
            raise Exception("S_h has nan.")

        if x.isnan().sum() > 0:
            raise Exception("x has nan.")

        if not S_h.isfinite().all():
            raise Exception("S_h has inf.")

        if not x.isfinite().all():
             raise Exception("x has inf.")

        if not EE.isfinite().all():
            raise Exception(f"EE contains inf : {id(EE)}")

        x_l = self.w_e1(x)
        x_r = self.w_e2(x)
        S_raw = x_l + x_r.t()
        S = torch.exp(S_raw - S_raw.max()) # NxN

        c = S_h.argmax(dim=-1)

        eps = 1e-5

        A = scatter(src=S, index=c,dim=0,dim_size=S_h.size(-1), reduce="sum") # c has cluster of each node
        B = scatter(src=A, index=c,dim=1,dim_size=S_h.size(-1), reduce="sum") # CxC
        E = S / (B[c, c] + eps)

        if not EE.isfinite().all():
            raise Exception(f"EE contains inf : {id(EE)}")

        if not S.isfinite().all():
            print(x_l.min(), x_r.max())
            print(S_raw.min(), S_raw.max())
            raise Exception("S contains inf")

        if not A.isfinite().all():
            raise Exception("A contains inf")

        if not B.isfinite().all():
            raise Exception("B contains inf")

        if not E.isfinite().all():
            raise Exception(f"E contains inf {torch.sum(B == 0)}")

        if S_h.isnan().sum() > 0:
            raise Exception("NaN in S_h")

        if A.isnan().sum() > 0:
            raise Exception("NaN in A")

        if B.isnan().sum() > 0:
            raise Exception("NaN in B")

        if S.isnan().sum() > 0:
            raise Exception("NaN in S")

        if E.isnan().sum() > 0:
            raise Exception("NaN in E")

        EE_pooled = S_h.t() @ (EE * E) @ S_h
        EA_pooled = S_h.t() @ EA

        if EE_pooled.isnan().sum() > 0:
            raise Exception("NaN in EE_pooled")

        return EE_pooled, EA_pooled
        
    def pool(
        self,
        x : Tensor,
        EE : AdjMatrix,
        EA : AdjMatrix,
    ) -> ClusterPoolOutput:

        S, S_raw = self.assign(x, edge_index=EE.edge_index.edge_index)
        S_h = self.one_hot(S)

        x_pooled = self.node_pool(x, S_h)

        dense = EE.dense

        if not dense.isfinite().all():
            raise Exception("EE contains inf (wtf)")

        # EE_pooled, EA_pooled = self.attn_edge_pool(
        #     x=x, 
        #     S_h=S_h, 
        #     EE=dense, 
        #     EA=EA.dense
        # )

        EE_pooled, EA_pooled = self.edge_pool(
            S=S_h, 
            EE=dense, 
            EA=EA.dense
        )

        return ClusterPoolOutput(
            S=S,
            x=x_pooled,
            EE=AdjMatrix(_dense=EE_pooled),
            EA=AdjMatrix(_dense=EA_pooled),
            aux_loss=torch.scalar_tensor(0.0).to(x.device),
            input=Input(EE=EE, EA=EA),
            metadata={ "S_h" : S_h }
        )

    def node_pool(self, x : Tensor, S : Tensor) -> Tensor:
    
        eps = 1e-6

        if self.norm:
            S = S / (S.sum(dim=0, keepdim=True) + eps)

        return S.t() @ x

    def edge_pool(self, EE : Tensor, EA : Tensor, S : Tensor) -> Tuple[Tensor, Tensor]:

        EE_pooled = S.t() @ EE @ S # (N'_e, N_e) x (N_e, N_e) x (N_e, N'_e) = (N'_e, N'_e)
        EA_pooled = S.t() @ EA     # (N'_e, N_e) x (N_e, N_a)  = (N'_e, N_a)

        return EE_pooled, EA_pooled

    def unpool(self, 
        x : Tensor,
        EE : AdjMatrix,
        EA : AdjMatrix,
        out : ClusterPoolOutput           
    ) -> ClusterPoolOutput:

        S = out.metadata['S_h'].t()
        x_unpooled = self.node_pool(x=x, S=S)

        EE_unpooled, EA_unpooled = self.edge_pool(
            EE=EE.dense,
            EA=EA.dense,
            S=S
        )

        return ClusterPoolOutput(
            S=S,
            x=x_unpooled,
            EE=AdjMatrix(_dense=EE_unpooled),
            EA=AdjMatrix(_dense=EA_unpooled),
            aux_loss=torch.scalar_tensor(0.0).to(x.device),
            input=Input(EE=EE, EA=EA)
        )

class VQPool(nn.Module):

    def __init__(self, codebook : MultiLevelCodebook) -> None:

        super().__init__()

        self.codebook = codebook
        self.n_prototypes = self.codebook.levels[-1]

    def node_pool(self, outputs : List[CodebookOutput]) -> Tuple[Tensor, Tensor, Tensor]:

        q = torch.cat([out.q for out in outputs], dim=-1)
        clusters = outputs[-1].S.argmax(dim=-1)
        codebook_loss = sum(out.codebook_loss for out in outputs) / len(outputs)
        commitement_loss = sum(out.commitement_loss for out in outputs) / len(outputs)

        x_new = scatter(
            src=q,
            index=clusters,
            dim=0,
            dim_size=outputs[-1].S.size(-1),
            reduce='mean'
        )

        return x_new, codebook_loss, commitement_loss

    def edge_pool(self, 
        S : Tensor,
        edge_index : Tensor, 
        entity_article : Optional[Tensor] = None,
        edge_attr : Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:

        clusters = S.argmax(dim=-1)
        src, dst = edge_index
        src, dst = clusters[src], clusters[dst]

        ### Entity <-> Entity pooling
        edge_index = torch.stack([src, dst], dim=0)

        edge_index, edge_attr = gnn_utils.coalesce(
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=S.size(0),
            reduce='mean'
        )

        edge_index, edge_attr = gnn_utils.remove_self_loops(
            edge_index=edge_index,
            edge_attr=edge_attr
        )

        ### Entity<->Article Pooling
        if entity_article is not None:

            src, dst = entity_article
            src = clusters[src]

            entity_article, _ = gnn_utils.coalesce(
                edge_index=entity_article,
                num_nodes=S.size(0),
                reduce='mean'
            )

            entity_article, _ = gnn_utils.remove_self_loops(edge_index=entity_article)

        return edge_index, entity_article, edge_attr

    def unpool(
        self,
        x : Tensor,
        EE : AdjMatrix,
        EA : AdjMatrix,
        out : ClusterPoolOutput
    ) -> ClusterPoolOutput:

        c = out.S.argmax(dim=-1)
        x_unpooled = x[c,:]

        return ClusterPoolOutput(
            S=out.S,
            x=x_unpooled,
            EE=out.input.EE,
            EA=out.input.EA,
            aux_loss=None,
            input=Input(EA=EA, EE=EE)
        )        

    def pool(
        self,
        x : Tensor,
        EE : AdjMatrix,
        EA : Optional[AdjMatrix],
    ) -> ClusterPoolOutput:

        outputs = self.codebook.forward(x)
        x_pooled, codebook_loss, commitement_loss = self.node_pool(outputs)

        EE_pooled, EA_pooled, _ = self.edge_pool(
            S=outputs[-1].S,
            edge_index=torch.stack([EE.edge_index.row, EE.edge_index.col], dim=0),
            entity_article=torch.stack([EA.edge_index.row, EA.edge_index.col], dim=0),
            edge_attr=None
        )

        return ClusterPoolOutput(
            S=outputs[-1].S,
            x=x_pooled,
            EE=AdjMatrix(_edge_index=EdgeIndex(
                row=EE_pooled[0], 
                col=EE_pooled[1],
                num_nodes=(self.n_prototypes, self.n_prototypes)
            )),
            EA=AdjMatrix(_edge_index=EdgeIndex(
                row=EA_pooled[0],
                col=EA_pooled[1],
                num_nodes=(self.n_prototypes, EA.edge_index.num_nodes[1])
            )),
            aux_loss=codebook_loss + commitement_loss,
            input=Input(EA=EA, EE=EE),
            metadata={
                "codebook_loss" : codebook_loss,
                "commitement_loss" : commitement_loss
            }
        )


def create_pooling_layer(
    kind : POOLING_KIND,
    in_channels : int,
    nn : Optional[nn.Module] = None,
    num_prototypes : Optional[int] = None,
    norm : bool = False,
    unpool_kind : UNPOOL_KIND = 'matmul'
) -> Pool:

    # if kind == 'enad':
    #     return ENADPool(nn=nn, in_channels=in_channels, norm=norm)
    if kind == 'vq':

        return VQPool(
            codebook=MultiLevelCodebook(
                channels=in_channels, 
                levels=[num_prototypes]
            )
        )
    
    else:

        return ClusterPool(
            nn=nn, 
            in_channels=in_channels, 
            kind=kind, 
            norm=norm, 
            unpool_kind=unpool_kind
        )
    

if __name__ == '__main__':

    D = 64
    N_e, N_a = 128, 10
    N_ee, N_ea = 328, 150

    x = torch.randn(size=(N_e, D))

    EE = AdjMatrix(
        _edge_index=EdgeIndex(
            row=torch.randint(low=0, high=N_e, size=(N_ee,)),
            col=torch.randint(low=0, high=N_e, size=(N_ee,)),
            num_nodes=(N_e, N_e)
        )
    )

    EA = AdjMatrix(
        _edge_index=EdgeIndex(
            row=torch.randint(low=0, high=N_e, size=(N_ea,)),
            col=torch.randint(low=0, high=N_a, size=(N_ea,)),
            num_nodes=(N_e, N_a)
        )
    )

    from itertools import product

    for kind, unpool_kind in product(
        ['diffpool', 'mincut', 'dmon', 'enad'],
        ['restore', 'weighted_restore', 'matmul']
    ):

        print(f"Kind : {kind}")

        m = create_pooling_layer(
            nn=nn.Linear(in_features=D, out_features=N_e // 4, bias=False),
            kind=kind,
            in_channels=D,
            unpool_kind=unpool_kind
        )

        out = m.pool(x=x, EE=EE, EA=EA)

        print("Pooling output: ")

        print(out.x.size())
        print(out.S.size())
        print(out.EE.dense.size())
        print(out.EA.dense.size())
        print(out.aux_loss.size() if out.aux_loss is not None else None)

        out = m.unpool(
            x=out.x, 
            EE=out.EE, 
            EA=out.EA,
            out=out,
        )
        
        print("UnPooling output: ")

        print(out.x.size())
        print(out.S.size())
        print(out.EE.dense.size())
        print(out.EA.dense.size())
        print(out.aux_loss.size() if out.aux_loss is not None else None)

        print("\n\n")