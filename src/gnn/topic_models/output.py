from dataclasses import dataclass
from typing import Dict
from src.gnn.losses import BaseLossContext
from torch_geometric.data import HeteroData
from torch import nn

class __OutputMeta(type):

    def __new__(cls, name, bases, namespace, /, **kwds):
        cls = type(name, bases, namespace, **kwds)
        return dataclass(cls)

class BaseModelOutput(metaclass=__OutputMeta):

    def context(self, batch : HeteroData, batch_size : int) -> Dict[str, BaseLossContext]:
        return {}

class Model(nn.Module):

    def forward(self, **kwargs) -> BaseModelOutput:
        ...