from torch import nn, Tensor
from typing import Dict

class MultiTaskHead(nn.Module):

    def __init__(self,
        in_features : int,
        dropout_rate : float,
        num_labels : Dict[str, int],
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.dropout_rate = dropout_rate
        self.num_labels = num_labels

        self.heads = nn.ModuleDict()
        
        for label_name, label_dim in self.num_labels.items():

            self.heads.add_module(label_name, nn.Sequential(
                nn.Dropout(p = dropout_rate) if dropout_rate > 0 else nn.Identity(),
                nn.Linear(in_features, label_dim)
            ))

    def forward(self, x : Tensor) -> Dict[str, Tensor]:
            
        return {
            label_name : m(x)
            for label_name, m in self.heads.items()
        }