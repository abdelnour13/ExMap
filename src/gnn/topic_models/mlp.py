from src.gnn import losses
from torch import nn, Tensor
from typing import List, Dict, Tuple
from src.gnn.layers import MLP
from torch_geometric.data import HeteroData
from dataclasses import dataclass, field
from pydantic import BaseModel

@dataclass
class SparseArticleEncoderOutput:
    h : Tensor
    y_hat : Tensor
    weight : Tensor

    def context(self, batch : HeteroData, batch_size : int) -> Dict[str, losses.BaseLossContext]:

        return {
            "b_c_e_loss" : losses.BCELossContext(
                y=batch['Article']['pseudo_labels'],
                y_hat=self.y_hat
            )
        }

class SparseArticleEncoderConfig(BaseModel):
    hidden_channels : List[int] = field(default_factory=lambda : [384, 384])
    activation : str = 'relu'
    norm : bool = True

class SparseArticleEncoder(nn.Module):

    def __init__(self, 
        in_channels : int,
        hidden_channels : List[int],
        activation : str,
        norm : bool,
        num_labels : int
    ) -> None:
        super().__init__()

        self.mlp = MLP(
            input_dim=in_channels,
            channels=hidden_channels[:-1],
            output_dim=hidden_channels[-1],
            hidden_activation=activation,
            norm=norm,
            output_activation=None
        )

        self.head = nn.Linear(
            in_features=hidden_channels[-1],
            out_features=num_labels
        )

    @staticmethod
    def from_config(
        in_channels : int,
        num_labels : Dict[str, int],
        config : SparseArticleEncoderConfig                
    ) -> 'SparseArticleEncoder':
        
        return SparseArticleEncoder(
            in_channels = in_channels,
            num_labels = num_labels['pseudo_labels'],
            **config.model_dump()
        )

    def forward(self, data : HeteroData):

        x = data['Article'].x

        h = self.mlp(x)
        y_hat = self.head(h)

        return SparseArticleEncoderOutput(
            h=h,
            y_hat=y_hat,
            weight=self.head.weight
        )