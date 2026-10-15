from pydantic import BaseModel
from src.gnn.layers.encoder import TransEncoder
from src.gnn.layers.reduction import Reduction, READOUT
from torch import nn, Tensor
from typing import Literal, List, Optional, Tuple, Dict
from torch_geometric.data import HeteroData
from src.gnn import ops, losses
from src.gnn.topic_models.output import BaseLossContext
from dataclasses import dataclass

ENCODER_TYPE = Literal['gnn', 'mlp']
MODE = Literal['one', 'multi', 'topk']
READOUT_MODE = Literal['pre', 'post']

@dataclass
class KGEncoderOutput:
    entities_embeddings : Tensor
    articles_embeddings : Tensor
    attn_weights : Tensor

    def context(self, batch : HeteroData, batch_size : int) -> Dict[str, BaseLossContext]:

        return {
            'l_2_distance_rec_loss' : losses.CosineRecLossContext(
                a=batch['Article'].x,
                b=self.articles_embeddings
            )
        }

class KGEncoderConfig(BaseModel):
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

class KGEncoder(nn.Module):

    def __init__(
        self,
        encoder_type : ENCODER_TYPE,
        in_channels  : int,
        hidden_channels : List[int],
        out_channels : int,
        num_relations : int,
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
        self.num_relations = num_relations
        self.variant = variant
        self.activation = activation
        self.norm = norm
        self.shortcut = shortcut
        self.shortcut_dim = shortcut_dim
        self.jk = jk
        self.readout = readout

        self.encoder = TransEncoder(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            num_relations=num_relations,
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
            weights_nn=nn.Linear(in_features=self.out_channels, out_features=1)
        )

    @staticmethod
    def from_config(
        in_channels : int,
        num_relations : Tuple[int, int],
        config : KGEncoderConfig                
    ) -> 'KGEncoder':
        
        return KGEncoder(
            in_channels = in_channels,
            num_relations = num_relations,
            **config.model_dump()
        )

    def ste(self, h : Tensor, q : Tensor) -> Tuple[Tensor, Tensor, Tensor]:

        if self.ste_mode == 'one':
        
            h_quan, codebook_loss, commitement_loss = ops.STE(
                codebook=self.C.weight,
                x=h,
                assignments=q
            )

        elif self.ste_mode == 'multi':

            h_quan, codebook_loss, commitement_loss = ops.mSTE(
                codebook=self.C.weight,
                x=h,
                assignments=q,
                threshold=0,
                weighted=False
            )

        elif self.ste_mode == 'topk':

            h_quan, codebook_loss, commitement_loss = ops.topkSTE(
                codebook=self.C.weight,
                x=h,
                assignments=q,
                k=5,
                weighted=False
            )

        else:
            assert False, f"self.ste_mode = '{self.ste_mode}' is not allowed."

        return h_quan, codebook_loss, commitement_loss

    def forward(self, data : HeteroData):

        KG = {
            'x' : data['Entity'].x, 
            'edge_attr' : data['Entity', 'HASRELATIONWITH', 'Entity'].edge_attr, 
            'edge_index' : data['Entity', 'HASRELATIONWITH', 'Entity'].edge_index
        }

        ED = data['Entity', 'APPEARIN', 'Article'].edge_index
        N = data['Article'].x.size(0)

        ### Encoder
        xe_hat = self.encoder(KG)

        xd_hat, weights, _ = self.reduce.forward(
            x=xe_hat,
            edge_index=ED,
            num_nodes=N
        ).astuple()

        # for name, p in self.named_parameters():
        #     if not p.isfinite().all():
        #         print("BAD PARAM:", name)
        #         print("NaN:", p.isnan().sum())
        #         print("Inf:", p.isinf().sum())

        return KGEncoderOutput(
            entities_embeddings=xe_hat,
            articles_embeddings=xd_hat,
            attn_weights=weights
        )