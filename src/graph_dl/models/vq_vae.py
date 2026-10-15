import torch
import math
from pydantic import BaseModel
from src.graph_dl.layers.encoder import TransEncoder
from src.graph_dl.layers.reduction import Reduction, READOUT
from src.graph_dl.models.base_model import TopicModel, Context, IterationContext
from src.graph_dl.layers.mlp import MLP
from torch import nn, Tensor
from typing import Literal, List, Optional, Dict
from torch_geometric.data import HeteroData
from src.graph_dl import losses, ops
from dataclasses import dataclass

ENCODER_TYPE = Literal['gnn', 'mlp']

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
                'cosine_rec_loss' : losses.CosineRecLossContext(
                    a=batch['Article'].x,
                    b=self.xd_hat
                ),
                'codebook_loss' : losses.Identity(self.codebook_loss),
                'commitement_loss' : losses.Identity(self.commitement_loss)
            }

    class Config(BaseModel):
        encoder_type : ENCODER_TYPE = 'gnn'
        encoder_type : ENCODER_TYPE = 'gnn'
        hidden_channels : Optional[List[int]] = None
        out_channels : int = 256
        variant : Literal['conv', 'attn'] = 'conv'
        activation : str = 'relu'
        norm : bool = True
        shortcut: bool = False
        shortcut_dim: Optional[int]  = None
        jk: bool = True
        readout : READOUT = 'attn'
        n_prototypes : int = 128

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
        n_prototypes : int
    ) -> None:

        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.variant = variant
        self.activation = activation
        self.norm = norm
        self.shortcut = shortcut
        self.shortcut_dim = shortcut_dim
        self.jk = jk
        self.readout = readout
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type

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
            mlp=(encoder_type == 'mlp')
        )

        self.decoder = TransEncoder(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            variant=variant,
            activation=activation,
            norm=norm,
            shortcut=shortcut,
            shortcut_dim=shortcut_dim,
            jk=jk,
            mlp=(decoder_type == 'mlp')
        )

        self.reduce = Reduction(
            kind=self.readout,
            weights_nn=nn.Sequential(
                nn.Linear(in_features=self.out_channels, out_features=out_channels),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Linear(in_features=self.out_channels, out_features=1)
            ) if self.readout == 'attn' else None
        )

        self.codebook = nn.Parameter(
            data=torch.empty(size=(out_channels, n_prototypes)),
            requires_grad=True
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.C, a=math.sqrt(5))

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

    def assign(self, x : Tensor) -> Tensor:
        S = - ops.dist(x, self.C.data)
        S = torch.softmax(S, dim=-1)
        return S

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

        ### Encoder
        he = self.encoder(KG) # (N, D)

        ### STE
        S = self.assign(he)   # (N, M)

        he_q, codebook_loss, commitement_loss = ops.STE(
            codebook=self.codebook,
            x=he,
            assignments=S
        )

        KG['x'] = he_q

        self.decoder(KG)

        xe_hat = self.decoder(he_q)

        xd_hat, weights, _ = self.reduce.forward(
            x=xe_hat,
            edge_index=ED,
            num_nodes=N
        ).astuple()

        return self.Output(
            he=he,
            S=S,
            he_q=he_q,
            xe_hat=xe_hat,
            xd_hat=xd_hat,
            attn_weights=weights,
            codebook_loss=codebook_loss,
            commitement_loss=commitement_loss
        )