from pydantic import BaseModel
from src.graph_dl.layers.encoder import TransEncoder
from src.graph_dl.models.base_model import TopicModel, Context, IterationContext
from torch import Tensor
from typing import Literal, List, Optional, Dict
from torch_geometric.data import HeteroData
from src.graph_dl import losses
from dataclasses import dataclass

ENCODER_TYPE = Literal['gnn', 'mlp']

@TopicModel.register
class EncoderDecoder(TopicModel):

    @dataclass
    class Output:
        hidden_representations : Tensor
        xe_hat : Tensor

        def context(self, batch : HeteroData, batch_size : int) -> Dict[str, losses.BaseLossContext]:

            return {
                'cosine_rec_loss' : losses.CosineRecLossContext(
                    x1=batch['Entity'].x,
                    y1=self.xe_hat,
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
            in_channels=out_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels[::-1],
            variant=variant,
            activation=activation,
            norm=norm,
            shortcut=shortcut,
            shortcut_dim=shortcut_dim,
            jk=jk,
            mlp=(encoder_type == 'mlp'),
            use_edges_info=False
        )

    @staticmethod
    def from_config(
        config : Config, 
        ctx : Context, 
        labels : Dict[str, int] = None
    ) -> 'EncoderDecoder':

        return EncoderDecoder(
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

        ### Encoder
        h = self.encoder(KG)

        KG['x'] = h
        KG['edge_attr'] = None

        xe_hat = self.decoder(KG)

        return self.Output(
            hidden_representations=h,
            xe_hat=xe_hat
        )