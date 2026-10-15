import torch
from torch import nn, Tensor
from torch_geometric import nn as gnn
from typing import Optional, Dict, List, Type, Union
from src.gnn.layers import TransGNN, Activation, RGNNLayer
from typing import Union, Literal

def _Linear(feature_name : str):

    class __Linear(nn.Linear):

        def __init__(self, 
            in_channels, 
            out_channels, 
            activation : str, 
            norm: bool, 
            bias = True, 
            device=None, 
            dtype=None, 
            **kwargs
        ):
            super().__init__(in_channels, out_channels, bias, device, dtype)
            self.activation = Activation(activation)
            self.bn = nn.BatchNorm1d(num_features=out_channels) if norm else nn.Identity()

        def forward(self, x : Tensor, **kwargs) -> Dict[str, Tensor]:

            x = self.activation(self.bn(super().forward(x)))
            
            return {
                feature_name : x
            }

    return __Linear

class GNNEncoder(nn.Module):

    def __init__(self,
        layer_cls : Type[nn.Module], 
        layers_args : List[Dict],
        feature_name : str,
        common_args : Optional[Dict] = None,
        shortcut : bool = False,
        shortcut_dim : Optional[int] = None,
        jk : bool = True, # max
        projection_dim : Optional[int] = None
    ) -> None:
        super().__init__()

        self.layer_cls = layer_cls
        self.layers_args = layers_args
        self.feature_name = feature_name
        self.common_args = common_args or {}
        self.shortcut = shortcut
        self.shortcut_dim = shortcut_dim
        self.jk = jk
        self.projection_dim = projection_dim

        if len(layers_args) == 0:
            raise Exception(f"layers_args should contain at least one layer specification.")

        if self.shortcut_dim is None and self.jk:

            assert all([
                layer_args.get("out_channels") == self.layers_args[-1].get("out_channels") 
                for layer_args in self.layers_args
            ]), "if shortcut_dim is None and jk is True then all encoder layers should have the same output dimension."

        self.layers = self.encoder_layers()
        self.shortcuts = self.shortcut_layers()
        self.jk_nn = self.jumping_knowledge()
        self.project = self.projection_layer()

    def encoder_layers(self) -> nn.ModuleList:
        
        return nn.ModuleList(modules=[
            self.layer_cls(**{
                **self.common_args,
                **layer_args,
            }) 
            for layer_args in self.layers_args
        ])

    def shortcut_layers(self) -> nn.ModuleList:

        return nn.ModuleList([
            nn.Linear(
                in_features=layer_args.get("out_channels"),
                out_features=self.shortcut_dim if self.shortcut_dim is not None else self.layers_args[-1].get("out_channels")
            ) if self.shortcut else nn.Identity()
            for layer_args in self.layers_args
        ])

    def jumping_knowledge(self) -> Optional[gnn.JumpingKnowledge]:

        if self.jk:
            return gnn.JumpingKnowledge(mode='max')

        return None

    def projection_layer(self) -> Union[nn.Identity | nn.Linear]:

        if not self.projection_dim:
            return nn.Identity()

        input_dim = (
            self.shortcut_dim 
            if self.shortcut_dim is not None and self.shortcut 
            else self.layers_args[-1].get("out_channels")
        )

        return nn.Linear(in_features=input_dim,out_features=self.projection_dim)

    def forward(self, data : Dict[str, Tensor]) -> Tensor:

        data = data.copy()
        outputs = []

        for layer, shortcut in zip(self.layers, self.shortcuts):

            ### Apply GNN Layer
            output : Dict[str, Tensor] = layer(**data)

            ### Apply shortcut to unify dimensions
            x = shortcut(output[self.feature_name])
            outputs.append(x)

            ### Update Input Data
            for k, v in output.items():

                if k in data:
                    data[k] = v

        if self.jk:
            x = self.jk_nn(outputs)
        else:
            x = outputs[-1][self.feature_name]

        x = self.project(x)

        return x

class _TransGNN(TransGNN):

    def __init__(self, 
        in_channels, 
        out_channels, 
        num_relations, 
        use_edges_info = True, 
        kg_scoring_fn = 'TransE', 
        variant = 'conv', 
        combine = 'cat', 
        num_heads = 1, 
        bias = False, 
        activation = 'relu', 
        norm = False, 
        version = 'v1'
    ):
        super().__init__(in_channels, out_channels, num_relations, use_edges_info, kg_scoring_fn, variant, combine, num_heads, bias, activation, norm, version)

    def forward(self, 
        x, 
        edge_index, 
        node_score = None, 
        edge_type = None, 
        relations = None, 
        edge_attr = None
    ) -> Dict[str, Tensor]:
        
        x, edge_attr, relations = super().forward(
            x, 
            edge_index, 
            node_score, 
            edge_type, 
            relations, 
            edge_attr
        )

        return {
            'x' : x,
            'edge_attr' : edge_attr
        }

class TransEncoder(GNNEncoder):

    __trans_gnn__ = {
        "kg_scoring_fn": "TransE",
        "combine": "cat",
        "num_heads": 1,
        "bias": True,
        "version": "v2",
    }

    def __init__(self,
        in_channels: int,
        out_channels: int,
        hidden_channels : List[int],
        num_relations: int,
        variant: Literal["conv", "attn"] = "conv",
        activation: str = "relu",
        norm: bool = False,
        shortcut: bool = False,
        shortcut_dim: int | None = None,
        jk: bool = True,
        mlp : bool = False,
        use_edges_info : bool = True
    ) -> None:

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_relations = num_relations
        self.variant = variant
        self.activation = activation
        self.norm = norm
        self.use_edges_info = use_edges_info

        channels = [in_channels, *hidden_channels]
        
        super().__init__(
            layer_cls=_TransGNN if not mlp else _Linear('x'),
            layers_args=[
                {
                    "in_channels" : in_channels,
                    "out_channels" : out_channels
                }
                for in_channels, out_channels in zip(channels[:-1], channels[1:])
            ],
            common_args={
                "num_relations" : num_relations,
                "variant" : variant,
                "activation" : activation,
                "norm" : norm,
                "use_edges_info" : self.use_edges_info,
                **self.__trans_gnn__,
            },
            shortcut=shortcut,
            shortcut_dim=shortcut_dim,
            jk=jk,
            projection_dim=self.out_channels,
            feature_name='x'
        )

class RGNNEncoder(GNNEncoder):

    def __init__(self,
        in_channels: int,
        out_channels: Optional[int],
        hidden_channels : List[int],
        num_relations : int = 1,
        activation : Optional[str] = None,
        agg : str = None,
        norm : bool = True,
        root : bool = True,
        shortcut: bool = False,
        shortcut_dim : Optional[int] = None,
        jk : bool = True,
        mlp : bool = True
    ) -> None:

        channels = [in_channels, *hidden_channels]

        super().__init__(
            layer_cls=RGNNLayer if not mlp else _Linear('x'),
            common_args={
                "expect_edges_info" : num_relations > 1,
                "activation" : activation,
                "agg" : agg,
                "norm" : norm,
                "root" : root
            },
            layers_args=[
                {
                    "in_channels" : in_channels,
                    "out_channels" : out_channels
                }
                for in_channels, out_channels in zip(channels[:-1], channels[1:])
            ],
            shortcut=shortcut,
            shortcut_dim=shortcut_dim,
            jk=jk,
            projection_dim=out_channels,
            feature_name='x'
        )

        self.num_relations = num_relations

        if num_relations > 1:
            self.R = nn.Parameter(torch.randn((num_relations, in_channels)), requires_grad=True)

    def forward(self, data : Dict[str, Tensor]) -> Tensor:

        data = data.copy()

        if ('edge_type' in data) and ('edge_attr' not in data) and (self.num_relations > 1):
            data['edge_attr'] = self.R.data[data['edge_type'],:]

        if 'edge_type' in data:
            data.pop('edge_type')        

        return super().forward(data)