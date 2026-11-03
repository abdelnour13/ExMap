import torch
import math
from pydantic import BaseModel
from src.graph_dl.layers.encoder import TransEncoder
from src.graph_dl.layers.reduction import Reduction, READOUT
from src.graph_dl.layers.codebook import MultiLevelCodebook, CodebookOutput
from src.graph_dl.models.base_model import TopicModel, Context, IterationContext
from src.graph_dl.pooling.cluster_pool import VQPool, AdjMatrix, EdgeIndex
from torch import nn, Tensor
from typing import Literal, List, Optional, Dict, Any, TypeVar, Callable
from torch_geometric.data import HeteroData
from src.graph_dl import losses, ops
from dataclasses import dataclass

# S = pooled_graph.S
# he_q = unpooled_graph.x
# codebook_loss = pooled_graph.metadata['codebook_loss']
# commitement_loss = pooled_graph.metadata['commitement_loss']

ENCODER_TYPE = Literal['gnn', 'mlp']

A = TypeVar("A")
E = TypeVar('E')

@dataclass
class Lazy:
    fn : Callable[...,A]
    args : Dict[str, Any]

    def get(self) -> A:
        return self.fn(**self.args)

def pick(cond : bool, arg0 : A, arg1 : E) -> A | E:
    arg0 = arg0.get() if isinstance(arg0, Lazy) else arg0
    arg1 = arg1.get() if isinstance(arg1, Lazy) else arg1
    return [arg1, arg0][int(cond)]

def lazy(fn : Callable[..., A], **kwargs) -> Lazy:
    return Lazy(fn=fn, args=kwargs)

@TopicModel.register
class VQVAE(TopicModel):

    @dataclass
    class Output:
        he : Tensor
        S : Tensor
        he_q : Tensor
        xe_hat : Tensor
        xd_hat : Tensor
        attn_weights : Tensor
        codebook_loss : Tensor
        commitement_loss : Tensor

        def context(self, batch : HeteroData, batch_size : int) -> Dict[str, losses.BaseLossContext]:

            return {
                'e_rec_loss' : losses.CosineRecLossContext(x1=batch['Entity'].x, y1=self.xe_hat),
                'cosine_rec_loss' : losses.CosineRecLossContext(
                    x1=batch['Article'].x,
                    y1=self.xd_hat
                ),
                'codebook_loss' : losses.Identity(self.codebook_loss),
                'commitement_loss' : losses.Identity(self.commitement_loss)
            }

    class Config(BaseModel):
        encoder_type : ENCODER_TYPE = 'gnn'
        decoder_type : ENCODER_TYPE = 'gnn'
        hidden_channels : Optional[List[int]] = None
        out_channels : int = 256
        variant : Literal['conv', 'attn'] = 'conv'
        activation : str = 'relu'
        norm : bool = True
        shortcut: bool = False
        shortcut_dim: Optional[int]  = None
        jk: bool = True
        readout : READOUT = 'attn'
        n_prototypes : Optional[List[int] | int] = 128
        pool : bool = False

    def __init__(
        self,
        encoder_type : ENCODER_TYPE,
        decoder_type : ENCODER_TYPE,
        in_channels  : int,
        hidden_channels : List[int],
        out_channels : int,
        variant : Literal['conv', 'attn'],
        activation : str,
        norm : bool,
        shortcut: bool,
        shortcut_dim: Optional[int] ,
        jk: bool ,
        readout : READOUT,
        n_prototypes : Optional[List[int] | int],
        pool : bool
    ) -> None:

        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.variant = variant
        self.activation = activation
        self.norm = ['none', 'batchnorm'][int(norm)]
        self.shortcut = shortcut
        self.shortcut_dim = shortcut_dim
        self.jk = jk
        self.readout = readout
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        self.pool = pool

        N : int = 1 if not isinstance(n_prototypes, list) else len(n_prototypes)
        
        self.encoder = TransEncoder(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            variant=variant,
            activation=activation,
            norm=norm,
            shortcut=shortcut,
            shortcut_dim=shortcut_dim,
            jk=jk,
            mlp=(encoder_type == 'mlp'),
            use_edges_info=True
        )

        self.decoder = TransEncoder(
            in_channels=out_channels * N,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            variant=variant,
            activation=activation,
            norm=norm,
            shortcut=shortcut,
            shortcut_dim=shortcut_dim,
            jk=jk,
            mlp=(decoder_type == 'mlp'),
            use_edges_info=False
        )

        self.reduce = Reduction(
            kind=self.readout,
            weights_nn=nn.Sequential(
                nn.Linear(in_features=self.out_channels, out_features=out_channels),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Linear(in_features=self.out_channels, out_features=1)
            ) if self.readout == 'attn' else None
        )

        self.codebook : Optional[MultiLevelCodebook] = None
        self.graph_pool : Optional[VQPool] = None

        if n_prototypes is not None:

            self.codebook = MultiLevelCodebook(
                channels=out_channels,
                levels=pick(
                    isinstance(n_prototypes, int), 
                    [n_prototypes], 
                    n_prototypes
                )
            )

        if self.codebook is not None and self.pool:
            self.graph_pool = VQPool(codebook=self.codebook)
        
    @staticmethod
    def from_config(
        config : Config, 
        ctx : Context, 
        labels : Dict[str, int] = None
    ) -> 'VQVAE':

        return VQVAE(
            in_channels = ctx.num_features[0],
            **config.model_dump()
        )

    def ste(self, 
        x : Tensor, 
        EE : AdjMatrix,
        EA : AdjMatrix,
        temp : float | Tensor = 1.0
    ) -> CodebookOutput:

        if self.graph_pool is not None:

            pooled_graph = self.graph_pool.pool(
                x=x,
                EE=EE,
                EA=EA
            )

            unpooled_graph = self.graph_pool.unpool(
                x=pooled_graph.x,
                EE=pooled_graph.EE,
                EA=pooled_graph.EA,
                out=pooled_graph
            )

            S = pooled_graph.S
            q = unpooled_graph.x
            codebook_loss = pooled_graph.metadata['codebook_loss']
            commitement_loss = pooled_graph.metadata['commitement_loss']

        elif self.codebook is not None:
            outputs = self.codebook.forward(x=x)
            q = torch.cat([out.q for out in outputs], dim=-1)
            S = torch.cat([out.S for out in outputs], dim=-1)
            codebook_loss = sum(out.codebook_loss for out in outputs) / len(outputs)
            commitement_loss = sum(out.commitement_loss for out in outputs) / len(outputs)
        else:
            q = x
            S = []
            codebook_loss = torch.scalar_tensor(0.0).to(x.device)
            commitement_loss = torch.scalar_tensor(0.0).to(x.device)

        return CodebookOutput(
            S=S, 
            q=q, 
            codebook_loss=codebook_loss,
            commitement_loss=commitement_loss
        )

    def _adj(self, batch : HeteroData) -> tuple[AdjMatrix, AdjMatrix]:

        row, col = batch['Entity', 'HASRELATIONWITH', 'Entity'].edge_index

        EE = AdjMatrix(
            _edge_index=EdgeIndex(
                row=row,
                col=col,
                num_nodes=(batch['Entity']['x'].size(0), batch['Entity']['x'].size(0))
            )
        )

        row, col = batch['Entity', 'APPEARIN', 'Article'].edge_index

        EA = AdjMatrix(
            _edge_index=EdgeIndex(
                row=row,
                col=col,
                num_nodes=(batch['Entity']['x'].size(0), batch['Article']['x'].size(0))
            )
        )

        return EE, EA

    def forward(self, 
        data : HeteroData, 
        ctx : Optional[IterationContext] = None
    ) -> Output:

        KG = {
            'x' : data['Entity'].x, 
            'edge_attr' : data['Entity', 'HASRELATIONWITH', 'Entity'].edge_attr, 
            'edge_index' : data['Entity', 'HASRELATIONWITH', 'Entity'].edge_index
        }

        ED = data['Entity', 'APPEARIN', 'Article'].edge_index
        N = data['Article'].x.size(0)

        EE, EA = self._adj(batch=data)

        ### Encoder
        he = self.encoder.forward(KG) # (N, D)

        ### STE
        cb_out = self.ste(x=he, EE=EE, EA=EA)

        KG['x'] = cb_out.q

        xe_hat = self.decoder.forward(KG)

        xd_hat, weights, _ = self.reduce.forward(
            x=xe_hat,
            edge_index=ED,
            num_nodes=N
        ).astuple()

        return self.Output(
            he=he,
            S=cb_out.S,
            he_q=cb_out.q,
            xe_hat=xe_hat,
            xd_hat=xd_hat,
            attn_weights=weights,
            codebook_loss=cb_out.codebook_loss,
            commitement_loss=cb_out.commitement_loss
        )