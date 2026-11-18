import torch
from pydantic import BaseModel
from src.graph_dl.layers.encoder import TransEncoder
from src.graph_dl.layers.reduction import Reduction, READOUT
from src.graph_dl.layers.multi_task_head import MultiTaskHead
from src.graph_dl.models.base_model import TopicModel, Context, IterationContext
from torch import nn, Tensor
from typing import Literal, List, Optional, Dict
from torch_geometric.data import HeteroData
from src.graph_dl import losses
from dataclasses import dataclass

ENCODER_TYPE = Literal['gnn', 'mlp']

@TopicModel.register
class ArticleClassifier(TopicModel):

    @dataclass
    class Output:
        entities_sem_embeddings: Tensor
        entities_importance_embeddings : Tensor
        articles_embeddings : Tensor
        attn_weights : Tensor
        entities_labels : Dict[str, Tensor]
        articles_labels : Dict[str, Tensor]

        def context(self, batch : HeteroData, batch_size : int) -> Dict[str, losses.BaseLossContext]:

            return {
                f'bec_loss_{label}' : losses.BCELossContext(
                    y=batch['Article'][label],
                    y_hat=y
                )
                for label, y in self.articles_labels.items()
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
        num_labels : Dict[str, int]
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
        self.num_labels = num_labels

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

        self.reduce = Reduction(
            kind=self.readout,
            weights_nn=nn.Linear(in_features=self.out_channels, out_features=1) if self.readout == 'attn' else None
        )

        self.head = MultiTaskHead(
            in_features=out_channels,
            dropout_rate=0.0,
            num_labels=num_labels
        )

    @staticmethod
    def from_config(
        config : Config, 
        ctx : Context, 
        labels : Dict[str, int] = None
    ) -> 'ArticleClassifier':

        return ArticleClassifier(
            in_channels = ctx.num_features[0],
            num_labels=labels,
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
        he, he_imp = torch.split(h, split_size_or_sections=self.out_channels, dim=-1)

        hd, weights, _ = self.reduce.forward(
            x=he,
            edge_index=ED,
            num_nodes=N,
            weights_x=he_imp
        ).astuple()

        entities_labels = self.head.forward(he)
        articles_labels = self.head.forward(hd)

        return self.Output(
            entities_importance_embeddings=he_imp,
            entities_sem_embeddings=he,
            articles_embeddings=hd,
            attn_weights=weights,
            entities_labels=entities_labels,
            articles_labels=articles_labels
        )