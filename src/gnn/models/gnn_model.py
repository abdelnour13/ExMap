import torch
from typing import Optional, Literal, Union, Tuple
from dataclasses import dataclass
from torch import nn, Tensor
from torch_geometric import nn as gnn
from torch.nn import functional as F
from src.gnn.layers.rgat import RGATConv, Activation

@dataclass
class GNNModelConfig:
    hidden_channels : Optional[int] = None
    num_layers : int = 2
    root_weight : int = True
    hidden_activation : str = 'relu'
    output_activation : Optional[str] = None
    num_bases : Optional[int] = None
    variant : Optional[Literal['attn','conv']] = 'conv'
    layer_args : Optional[dict] = None
    dropout : float = 0.0   
    norm : bool = True 

class GNNModel(nn.Module):

    def __init__(self, 
        in_channels : int,
        out_channels : int,
        num_relations : int,
        hidden_channels : Optional[int] = None,
        num_layers : int = 2,
        root_weight : int = True,
        hidden_activation : str = 'relu',
        output_activation : Optional[str] = None,
        num_bases : Optional[int] = None,
        variant : Optional[Literal['attn','conv']] = 'conv',
        layer_args : Optional[dict] = None ,
        dropout : float = 0.0,
        norm : bool = True          
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels or in_channels
        self.num_layers = num_layers
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.root_weight = root_weight
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.num_bases = num_bases
        self.variant = variant
        self.layer_args = layer_args if layer_args is not None else {}
        self.dropout = dropout
        self.norm = norm

        self.hidden_activation_fn = Activation(name = self.hidden_activation)
        self.output_activation_fn = Activation(name = self.output_activation)

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(self.num_layers):

            layer = self._create_layer(
                in_channels = (in_channels if i == 0 else self.hidden_channels),
                out_channels = self.hidden_channels
            )

            self.layers.append(layer)

            if self.norm:
                self.norms.append(nn.BatchNorm1d(self.hidden_channels))
            else:
                self.norms.append(nn.Identity())

        self.head = nn.Linear(in_features = self.hidden_channels, out_features = out_channels)

    def _create_layer(self,
        in_channels : int,
        out_channels : int
    ) -> nn.Module:

        layer = None

        if self.variant == 'conv':

            layer = gnn.RGCNConv(
                in_channels = in_channels,
                out_channels = out_channels,
                num_bases = self.num_bases,
                num_relations = self.num_relations,
                bias = True,
                **self.layer_args
            )

        elif self.variant == 'attn':

            if self.num_relations == 1:

                if "residual" not in self.layer_args:
                    self.layer_args["residual"] = True

                layer = gnn.GATConv(
                    in_channels = in_channels,
                    out_channels = out_channels,
                    **self.layer_args
                )

            else:

                layer = RGATConv(
                    in_channels = in_channels,
                    out_channels = out_channels,
                    num_relations = self.num_relations,
                    bias = True,
                    **self.layer_args
                )

        return layer
    
    def forward(self, 
        x : Tensor,
        edge_index : Tensor,
        edge_type : Optional[Tensor] = None   ,
        edge_weight : Optional[Tensor] = None,
        return_representations : bool = False     
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
                            
        input_ = {
            'x' : x,
            'edge_index' : edge_index,
            'edge_type' : edge_type,
        }

        if self.num_relations == 1 and self.variant == "attn":
            input_.pop('edge_type')
                
        for layer,norm in zip(self.layers, self.norms):
            input_['x'] = layer(**input_)
            input_['x'] = norm(input_['x'])
            input_['x'] = self.hidden_activation_fn(input_['x'])

        if self.dropout and self.training:
            input_['x'] = F.dropout(input_['x'], p = self.dropout)

        y = self.head(input_['x'])
        y = self.output_activation_fn(y)

        if return_representations:
            return x, y
        else:
            return y
    
if __name__ == '__main__':

    N_x, N_e, D_x, R = 10, 40, 16, 2

    x = torch.randn(N_x, D_x)
    edge_index = torch.randint(0, N_x, (2, N_e))
    edge_type = torch.randint(0, R, size = (N_e,))

    for num_relations in [1,2]:

        for variant in ['conv','attn']:

            print(f"Number of relations = {num_relations}, variant = {variant} : ")

            model = GNNModel(
                in_channels = D_x,
                hidden_channels = D_x // 2,
                num_layers = 2,
                out_channels = 3,
                num_relations = num_relations,
                variant = variant
            )

            mask = edge_type == 0
            y_hat = model.forward(x, edge_index[:,mask], edge_type[mask])
            print(y_hat.size(), end = "\n\n")