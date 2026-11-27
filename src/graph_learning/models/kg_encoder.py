import torch
from pydantic import BaseModel
from src.graph_learning.layers.encoder import TransEncoder
from src.graph_learning.layers.reduction import Reduction, READOUT
from src.graph_learning.layers.activation import Activation
from src.graph_learning.layers.mlp import MLP
from src.graph_learning.models.base_model import TopicModel, Context, IterationContext
from torch import nn, Tensor
from typing import Literal, List, Optional, Dict
from torch_geometric.data import HeteroData
from src.graph_learning import losses
from dataclasses import dataclass

ENCODER_TYPE = Literal['gnn', 'mlp']

@TopicModel.register
class KGEncoder(TopicModel):

    @dataclass
    class Output:
        entities_sem_embeddings: Tensor
        entities_importance_embeddings : Tensor
        entities_embeddings : Tensor
        articles_embeddings : Tensor
        attn_weights : Tensor

        def context(self, batch : HeteroData, batch_size : int) -> Dict[str, losses.BaseLossContext]:

            return {
                'aa_cosine_rec_loss' : losses.CosineRecLossContext(
                    x1=batch['Article'].x,
                    y1=self.articles_embeddings
                ),
                'ea_cosine_rec_loss' : losses.CosineRecLossContext(
                    x1=batch['Article'].x,
                    x2=batch['Entity'].x,
                    y1=self.articles_embeddings,
                    y2=self.entities_embeddings
                )
            }

    class Config(BaseModel):
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

    def __init__(
        self,
        encoder_type : ENCODER_TYPE,
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

        self.encoder = TransEncoder(
            in_channels=in_channels,
            out_channels=2 * out_channels,
            hidden_channels=hidden_channels,
            variant=variant,
            activation=activation,
            norm=norm,
            shortcut=shortcut,
            shortcut_dim=shortcut_dim,
            jk=jk,
            mlp=(encoder_type == 'mlp')
        )

        self.decoder = MLP(
            input_dim=out_channels,
            channels=hidden_channels[::-1],
            output_dim=out_channels,
            hidden_activation=activation,
            norm=norm,
            output_activation=nn.Identity()
        )

        self.reduce = Reduction(
            kind=self.readout,
            weights_nn=nn.Linear(in_features=self.out_channels, out_features=1)
        )

    @staticmethod
    def from_config(
        config : Config, 
        ctx : Context, 
        labels : Dict[str, int] = None
    ) -> 'KGEncoder':

        return KGEncoder(
            in_channels = ctx.num_features[0],
            **config.model_dump()
        )

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
        h = self.encoder(KG)
        xe_sem, xe_imp = torch.split(h, split_size_or_sections=self.out_channels, dim=-1)
        xe_hat = self.decoder(xe_sem)

        xd_hat, weights, _ = self.reduce.forward(
            x=xe_sem,
            edge_index=ED,
            num_nodes=N,
            weights_x=xe_imp
        ).astuple()

        return self.Output(
            entities_embeddings=xe_hat,
            entities_importance_embeddings=xe_imp,
            entities_sem_embeddings=xe_sem,
            articles_embeddings=xd_hat,
            attn_weights=weights
        )