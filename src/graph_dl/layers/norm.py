import torch
from torch import nn
from typing import Literal, Optional

NORM_TYPE = Literal['batchnorm', 'layernorm', 'stdnorm', 'l2norm', 'test', 'none']

class Norm(nn.Module):

    def __init__(self, type : Optional[NORM_TYPE], num_features : int):
        super().__init__()

        self.type_ : NORM_TYPE = type
        self.num_features = num_features

        if self.type_ == 'batchnorm':
            self.norm = nn.BatchNorm1d(num_features)
        elif self.type_ == 'layernorm':
            self.norm = nn.LayerNorm(num_features)
        elif self.type_ == 'test':
            self.norm = nn.BatchNorm1d(num_features)
        else:
            self.norm = nn.Identity()

    def train(self, mode = True):

        if self.type_ != 'test' or mode:
            return super().train(mode)
        else:
            self.training = False
            self.norm.track_running_stats = False
            self.norm.train(mode=True)

        return self

    def forward(self, x : torch.Tensor) -> torch.Tensor:

        if self.type_ == 'stdnorm':
            x = (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, keepdim=True) + 1e-6)
        elif self.type_ == 'l2norm':
            x = x / torch.norm(x, p=2, dim=-1, keepdim=True)
        else:
            x = self.norm(x)

        return x


if __name__ == '__main__':

    x = torch.randn(16, 32)

    for type_ in ['batchnorm', 'layernorm', 'stdnorm', 'l2norm', 'test', 'none']:
        m = Norm(type=type_, num_features=32)
        print(m(x).shape)