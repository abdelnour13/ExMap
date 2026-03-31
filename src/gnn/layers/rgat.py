import torch
from torch import Tensor, nn
from torch.nn import functional as F
from typing import Literal, Optional
from torch_geometric import nn as gnn
from src.gnn.layers.activation import Activation

class RGATConv(nn.Module):

    def __init__(self,
        in_channels : int,
        out_channels : int,
        num_relations : int,
        pool : Literal['mean','max','attn'] = 'attn',
        bias : bool = True,
        norm : bool = False,
        activation : Optional[str] = None,
        root_weight : bool = True,
        num_bases : Optional[int] = None
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.pool = pool
        self.bias = bias
        self.norm = norm
        self.activation = activation
        self.root_weight = root_weight
        self.num_bases = num_bases

        if num_bases is not None:
            raise Exception("Num bases is yet to be implemented.")

        self.gnns = nn.ModuleList([
            gnn.GATConv(
                in_channels = in_channels,
                out_channels = out_channels,
                bias = bias
            )
            for _ in range(num_relations)
        ])

        ### Learnable Parameter For When No Relation
        self.relations = nn.Parameter(torch.randn(size = (num_relations, out_channels)), requires_grad = True)

        if self.root_weight:
            self.res = nn.Linear(in_features = in_channels, out_features = out_channels)
        else:
            self.res = nn.Identity()

        if self.pool == "attn":
            self.attn = nn.Linear(out_channels, 1)

        self.bn = nn.BatchNorm1d(num_features = out_channels) if norm else nn.Identity()
        self.act = Activation(activation)

    def message_and_aggregate(self,
        x : Tensor,
        edge_index : Tensor,
        edge_type : Tensor 
    ) -> Tensor:
        
        messages = []
        
        for type_ in range(self.num_relations):

            row = edge_index[0]

            if len(row) != 0:
                message = self.gnns[type_](x, edge_index[:,edge_type == type_])
            else:
                message = torch.clone(self.relations[type_].repeat(x.size(0), 1))

            messages.append(message)

        messages = torch.stack(messages, dim = 0)

        return messages
    
    def pool_messages(self, messages : Tensor, weights : Optional[Tensor] = None) -> Tensor:

        if self.pool == "mean":
            return messages.mean(0)
        elif self.pool == "max":
            return messages.max(0)[0]
        elif self.pool == "attn":
            messages = messages * weights
            messages = messages.sum(0)
            return messages
        else:
            raise Exception(f"Pooling {self.pool_2} Not Supported")

    def forward(self,
        x : Tensor,
        edge_index : Tensor,
        edge_type : Tensor   
    ) -> Tensor:
        
        messages = self.message_and_aggregate(
            x = x,
            edge_index = edge_index,
            edge_type = edge_type
        )

        weights = None
        
        if self.pool == "attn":
            R, B, D = messages.size()
            weights = messages.reshape(R * B, D)
            weights = self.attn(weights)
            weights = weights.reshape(R, B, 1)
            weights = F.softmax(weights, dim = 0)

        messages = self.pool_messages(messages, weights = weights)

        x = self.res(x) + messages
        x = self.bn(x)
        x = self.act(x)

        return x

if __name__ == '__main__':

    N_x, N_e, N_r = 100, 200, 3
    D = 16

    x = torch.randn(size = (N_x, D))
    edge_index = torch.randint(0, N_x, (2, N_e))
    edge_type = torch.randint(0, N_r, (N_e,))

    layer = RGATConv(
        in_channels = D,
        out_channels = D // 2,
        num_relations = N_r,
    )

    out = layer.forward(x, edge_index, edge_type)

    print(out.shape)