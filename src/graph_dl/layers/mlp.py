import torch
from torch import nn, Tensor
from typing import Optional
from src.graph_dl.layers.activation import Activation
from src.graph_dl.layers.norm import Norm, NORM_TYPE

class MLP(nn.Module):

    def __init__(self, 
        input_dim : int,
        channels : list[int],
        output_dim  : int,
        hidden_activation : str = 'relu',
        norm : Optional[NORM_TYPE] = None,
        output_activation : Optional[str | nn.Module] = None
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.channels = channels
        self.output_dim = output_dim
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.norm = norm

        self.hidden_activation_fn = Activation(hidden_activation) if not isinstance(hidden_activation, nn.Module) else hidden_activation
        self.output_activation_fn = Activation(output_activation) if not isinstance(output_activation, nn.Module) else output_activation

        self.layers = self._create_layers()

    def _create_layers(self) -> nn.ModuleList:

        layers = nn.ModuleList()
        dims = [self.input_dim,*self.channels, self.output_dim]

        for d_in,d_out in zip(dims[:-1],dims[1:]):

            layers.append(
                nn.Sequential(
                    nn.Linear(in_features = d_in, out_features = d_out),
                    Norm(self.norm, d_out)
                )
            )

        return layers
    
    def forward(self, x : Tensor) -> Tensor:

        for i,layer in enumerate(self.layers):

            x = layer(x)

            if i == len(self.layers) - 1:
                x = self.output_activation_fn(x)
            else:
                x = self.hidden_activation_fn(x)

        return x
    
if __name__ == '__main__':

    mlp = MLP(
        input_dim = 16,
        hidden_dim = 8,
        num_layers = 2,
        output_dim = 1
    )

    x = torch.randn(8, 16)
    y = mlp(x)

    print(y.size())
    print(y)