import torch
from torch import Tensor
from torch.nn import functional as F
from typing import Optional, Dict, Tuple, Literal, Type, TypeVar, List
from torch_scatter import scatter
from torch_geometric import utils as gnn_utils
from pydantic import BaseModel
from dataclasses import dataclass
from src.graph_dl import ops
from sklearn.metrics import r2_score, accuracy_score, f1_score, recall_score, precision_score

T = TypeVar('T')

class BaseLossParams(BaseModel):
    pass

class __ContextMeta(type):

    def __new__(cls, name, bases, namespace, /, **kwds):
        cls = type(name, bases, namespace, **kwds)
        return dataclass(cls)

class BaseLossContext(metaclass=__ContextMeta):

    def loss(self, params : BaseLossParams) -> Tensor | Dict[str, Tensor]:
        return torch.scalar_tensor(0.0)

############### Utils :

class Losses:

    _LOSSES : Dict[str, Tuple[Type[BaseLossContext], Type[BaseLossParams]]] = {}

    @classmethod
    def get_params_class(cls, cls_ : Type[BaseLossContext]) -> Type[BaseLossParams]:
        return cls_.loss.__annotations__['params']

    @classmethod
    def get_loss(cls, name : str) -> Tuple[Type[BaseLossContext], Type[BaseLossParams]]:

        if name not in cls._LOSSES:
            raise Exception(f"loss not found '{name}'. Available : {'|'.join(cls._LOSSES)}")

        return cls._LOSSES[name]

    @classmethod
    def register(cls, loss_ctx : Type[T]) -> Type[T]: 

        name = loss_ctx.__name__.replace('Context', '')
        name = ''.join([(c if c.lower() == c else f'_{c.lower()}') for c in name]).strip('_')

        params_cls = cls.get_params_class(loss_ctx)
        cls._LOSSES[name] = (loss_ctx, params_cls)

        class new_loss_ctx(loss_ctx):

            def loss(self, params : Dict) -> Tensor | Dict[str, Tensor]:
                return super().loss(params_cls(**params))

        return new_loss_ctx

############### Constraints :

class DiversityLossParams(BaseModel):
    norm : bool = True

@Losses.register
@dataclass
class DiversityLossContext:
    h : Tensor

    def loss(self, params : DiversityLossParams) -> Tensor:

        N = self.h.size(0)
        h = self.h

        if params.norm:
            h = F.normalize(h, p=2, dim=1)

        S = h @ h.t()
        S = S / (N * (N - 1))

        loss = S.fill_diagonal_(0.0).sum()

        return loss


class ConsistencyLossParams(BaseModel):
    norm : Literal['softmax', 'sum'] = 'sum'

@Losses.register
@dataclass
class ConsistencyLossContext:
    xa : Tensor
    xe : Tensor
    a2t : Tensor
    e2t : Tensor

    def loss(self, params : ConsistencyLossParams) -> Tensor:

        eps = 1e-6

        if params.norm == 'softmax': # if unormalized logits are provided
            a2t = self.a2t.softmax(dim=0)
            e2t = self.e2t.softmax(dim=-1)
        else: # sigmoid normalized probas
            a2t = self.a2t / (self.a2t.sum(dim=0, keepdim=True) + eps) # (N, T)
            e2t = self.e2t / (self.e2t.sum(dim=-1, keepdim=True) + eps) # (M, T)

        C = a2t.t() @ self.xa # (T, N) x (N, D) => (T, D)
        S = self.xe @ C.t()   # (M, D) x (D, T) => (M, T)

        loss = -torch.mean(e2t * S)

        return loss

class SparsityLossParams(BaseLossParams):
    pass

@Losses.register
@dataclass
class SparsityLossContext:
    p : Tensor

    def loss(self, params : SparsityLossParams) -> Tensor:

        # p: NxT

        mean_activation_rate = self.p.mean()

        reg = self.p.mean(dim=0) # (,T)
        reg = reg / reg.sum(dim=0,keepdim=True)
        reg = ops.entropy(x=reg)

        return mean_activation_rate - reg
    
############### Constrastive Losses :

class SharedTopicsLossParams(BaseModel):
    norm : bool = True
    temp : float = 0.1

@Losses.register
@dataclass
class SharedTopicsLossContext:
    za : Tensor
    ze : Tensor

    def loss(self, params : SharedTopicsLossParams) -> Tensor:

        # similarity matrix
        logits = self.za @ self.ze.T
        logits = logits / self.za.size(dim=-1)

        # temperature scaling
        logits = logits / params.temp

        # positives are diagonal entries
        labels = torch.arange(self.za.size(0), device=self.za.device)

        # symmetric InfoNCE
        loss_a2e = F.cross_entropy(logits, labels)
        loss_e2a = F.cross_entropy(logits.T, labels)

        loss = (loss_a2e + loss_e2a) / 2

        return loss

class ClipLossParams(BaseModel):
    norm : bool = True
    temp : float = 0.1

@Losses.register
@dataclass
class ClipLossContext:
    za : Tensor
    ze : Tensor

    def loss(self, params : ClipLossParams) -> Tensor:

        # normalize embeddings
        if params.norm:
            za = F.normalize(self.za, dim=1)
            ze = F.normalize(self.ze, dim=1)

        # similarity matrix
        logits = za @ ze.T

        # temperature scaling
        logits = logits / params.temp

        # positives are diagonal entries
        labels = torch.arange(self.za.size(0), device=self.za.device)

        # symmetric InfoNCE
        loss_a2e = F.cross_entropy(logits, labels)
        loss_e2a = F.cross_entropy(logits.T, labels)

        loss = (loss_a2e + loss_e2a) / 2

        return loss

############### Features Reconstruction Losses :

class MSERecLossParams(BaseModel):
    pass

@Losses.register
@dataclass
class MSERecLossContext:
    a : Tensor
    b : Tensor

    def loss(self, params : MSERecLossParams) -> Tensor:
        return F.mse_loss(self.a, self.b)

class CosineRecLossParams(BaseModel):
    norm : bool = True
    norm_sim_matrix : bool = False
    eps : float = 1e-8

@Losses.register
@dataclass
class CosineRecLossContext:
    x1 : Tensor
    y1 : Tensor
    x2 : Optional[Tensor] = None
    y2 : Optional[Tensor] = None

    def loss(
        self,
        params : CosineRecLossParams
    ) -> Dict[str, Tensor]:

        x1 = self.x1
        y1 = self.y1
        x2 = self.x2 if self.x2 is not None else self.x1
        y2 = self.y2 if self.y2 is not None else self.y1

        Cx = ops.cos_sim(x1, x2, norm=params.norm)
        Cy = ops.cos_sim(y1, y2, norm=params.norm)

        if params.norm_sim_matrix:
            Cx = ops.min_max_norm(x=Cx, dim=None, eps=params.eps)
            Cy = ops.min_max_norm(x=Cy, dim=None, eps=params.eps)
        
        MSE = F.mse_loss(Cx, Cy)

        A = Cx.detach()
        B = Cy.detach()

        MAE = torch.abs(A - B).mean()

        R2 = torch.scalar_tensor(r2_score(
            A.reshape(-1, 1).flatten().cpu(), 
            B.reshape(-1, 1).flatten().cpu()
        ))

        return {
            'loss' : MSE,
            'MAE' : MAE,
            'R2' : R2
        }

class L2DistanceRecLossParams(BaseModel):
    norm : bool = True
    norm_sim_matrix : bool = False
    eps : float = 1e-8

@Losses.register
@dataclass
class L2DistanceRecLossContext:
    a : Tensor
    b : Tensor

    def dist(self, x: Tensor, y: Tensor) -> Tensor:

        norm_x = torch.sum(x ** 2, dim=-1)
        norm_y = torch.sum(y ** 2, dim=-1)

        dot = x @ y.T


        D = (norm_x[:, None] + norm_y[None, :]) - 2 * dot
        D = D.clamp(min=0).sqrt()

        return D

    def loss(
        self,
        params : L2DistanceRecLossParams
    ) -> Dict[str, Tensor]:

        A = self.dist(self.a, self.a)
        B = self.dist(self.b, self.b)

        if params.norm_sim_matrix:
            A = (A - A.min()) / (A.max() - A.min() + params.eps)

        MSE = F.mse_loss(A, B)

        A = A.detach()
        B = B.detach()

        MAE = torch.abs(A - B).mean()

        R2 = torch.scalar_tensor(r2_score(
            A.reshape(-1, 1).flatten().cpu(), 
            B.reshape(-1, 1).flatten().cpu()
        ))

        return {
            'loss' : MSE,
            'MAE' : MAE,
            'R2' : R2
        }

############### Structure Reconstruction Losses :


class MarginStructureRecLossParams(BaseModel):
    num_negs : int = 5
    margin : float = 0.3

@Losses.register
@dataclass
class MarginStructureRecLossContext:
    x : Tensor
    edge_index : Tensor

    def loss(
        self,
        params : MarginStructureRecLossParams
    ) -> Tensor:
        
        row, col = self.edge_index

        x = F.normalize(self.x, dim=1)

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
            edge_index=self.edge_index,
            num_nodes=x.size(0),
            num_neg_samples=len(pos_sim) * params.num_neg
        )

        neg_sim = (x[n_row] * x[n_col]).sum(dim=-1)

        L = F.relu(params.margin - pos_sim[n_row] + neg_sim)

        return L.mean()

    def evaluate(self, params) -> Dict[str, Tensor]:
        return {}

########################## Classification

class BCELossParams(BaseModel):
    logits : bool = True
    focal : bool = False
    gamma : float = 2.0
    alpha : float = 0.5
    negative_sample : bool = False

@Losses.register
@dataclass
class BCELossContext:
    y : Tensor
    y_hat : Tensor

    @staticmethod
    def negative_sample(y: Tensor) -> Tuple[Tensor, Tensor]:
        pos = torch.argwhere(y.gt(0.5)).t()
        neg = gnn_utils.negative_sampling(edge_index=pos, num_nodes=(y.size(0), y.size(1)))
        row, col = torch.cat([pos, neg], dim=-1)
        return row, col

    def loss(self, params : BCELossParams) -> Dict[str, Tensor]:

        if params.logits:
            bce_loss = F.binary_cross_entropy_with_logits(input=self.y_hat, target=self.y, reduction="none")
            y_hat = self.y_hat.sigmoid()
        else:
            bce_loss = F.binary_cross_entropy(input=self.y_hat, target=self.y, reduction="none")
            y_hat = self.y_hat

        y_hard = y_hat.gt(0.5).float()

        if params.focal:
            p_t = y_hat * self.y + (1 - y_hat) * (1 - self.y)
            focal_weight = (1 - p_t) ** self.gamma
            bce_loss = focal_weight * bce_loss

        if params.negative_sample:
            row, col = self.negative_sample(self.y)
            bce_loss = bce_loss[row, col]

        alpha_weight = params.alpha * self.y + (1 - params.alpha) * (1 - self.y)
        bce_loss = alpha_weight * bce_loss
        bce_loss = bce_loss.mean()

        y = self.y.cpu().gt(0.5).float()
        y_hard = y_hard.cpu()

        return {
            "loss" : bce_loss,
            "exact_accuracy" : torch.scalar_tensor(accuracy_score(y_true=y, y_pred=y_hard)),
            "accuracy" : torch.scalar_tensor(accuracy_score(y_true=y.reshape(-1), y_pred=y_hard.reshape(-1))),
            "macro_f1_score" : torch.scalar_tensor(f1_score(y_true=y, y_pred=y_hard, average="macro", zero_division=0)),
            "macro_precision" : torch.scalar_tensor(precision_score(y_true=y, y_pred=y_hard, average="macro", zero_division=0)),
            "macro_recall" : torch.scalar_tensor(recall_score(y_true=y, y_pred=y_hard, average="macro", zero_division=0)),
        }

class ElasticNetLossParams(BaseModel):
    alpha : float = 0.7

@Losses.register
@dataclass
class ElasticNetLossContext:
    w : Tensor

    def loss(self, params : ElasticNetLossParams) -> Tensor:

        l1 = self.w.abs().sum()
        l2 = (self.w ** 2).sum()

        return params.alpha * l1 + (1 - params.alpha) * l2


class IdentityLossParams(BaseModel):
    pass

@Losses.register
@dataclass
class Identity:
    value : Tensor

    def loss(self, params : IdentityLossParams) -> Tensor:
        return self.value

class AggregateParams(BaseModel):
    agg : Literal['sum', 'avg'] = 'avg'
    losses_params : List[BaseLossParams]

@Losses.register
class AggregateContext:
    losses : List[BaseLossContext]

    def loss(self, params : AggregateParams) -> Tensor:

        s = sum(loss.loss() for loss in self.losses)

        if params.agg == 'avg':
            s = s / len(self.losses)

        return s