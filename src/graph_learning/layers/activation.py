from torch import nn, Tensor
from typing import Optional

class Activation(nn.Module):

    __DICT = {
        'relu' : nn.ReLU,
        'leaky_relu' : nn.LeakyReLU,
        'selu' : nn.SELU,
        'celu' : nn.CELU,
        'elu' : nn.ELU,
        'tanh' : nn.Tanh
    }

    def __init__(self, name : Optional[str] = None, **kwargs) -> None:

        super().__init__()

        self.name = name

        if name is None:
            self.act = nn.Identity()
        elif name in Activation.__DICT:
            cls_ = Activation.__DICT[name]
            self.act = cls_(**kwargs)
        else:
            raise ValueError(f"Activation : {name} is not supported, please chose one of : {','.join(list(Activation.__DICT.keys()))}")

    def forward(self,x : Tensor) -> Tensor:
        return self.act(x)