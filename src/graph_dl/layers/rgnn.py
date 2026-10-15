import torch
from torch import Tensor
from typing import Optional, Tuple, TypedDict
from torch import nn
from torch_scatter import scatter
from src.graph_dl.layers.activation import Activation
from torch_geometric import utils as gnn_utils

class RGNNOutput(TypedDict):
    x : Tensor
    edge_attr : Optional[Tensor]

    def astuple(self) -> Tuple[Tensor, Tensor]:
        return self.x, self.edge_attr

class RGNNLayer(nn.Module):

    def __init__(self, 
        in_channels : int,
        out_channels : int,
        use_edges_info : bool = False,
        activation : Optional[str] = None,
        agg : str = 'mean',
        norm : bool = True,
        res : bool = True
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_edges_info = use_edges_info
        self.activation = activation
        self.norm = norm
        self.agg = agg
        self.attn = (agg == 'attn')
        self.res = res

        ### Messages
        m = 2 + int(use_edges_info)

        self.w_message = nn.Linear(
            in_features=(m - 1) * in_channels,
            out_features=in_channels
        )

        ### Node And Edge Update
        self.w_res = nn.Linear(in_features=in_channels,out_features=out_channels) if self.res else nn.Identity()
        self.w_node = nn.Linear(in_features=in_channels,out_features=out_channels)
        self.w_edge = nn.Linear(in_features=3 * in_channels,out_features=out_channels) if use_edges_info else nn.Identity()

        ### Batch Norm And Activation
        self.bn_node = nn.BatchNorm1d(num_features=out_channels) if self.norm else nn.Identity()
        self.bn_edge= nn.BatchNorm1d(num_features=out_channels) if (self.norm and self.use_edges_info) else nn.Identity()
        
        self.act = Activation(name=activation)

        if self.attn:

            self.weight_nn = nn.Sequential(*[
                nn.Linear(in_features=m * in_channels, out_features=in_channels),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Linear(in_features=in_channels, out_features=1)
            ])

    def calculate_attn_weights(self, 
        x : Tensor, 
        edge_index : Tensor, 
        edge_attr : Optional[Tensor] = None
    ) -> Optional[Tensor]:

        if not self.attn:
            return None

        if self.use_edges_info:
            assert edge_attr is not None, f"use_edges_info == 'True'. but edge_type not provided."

        row, col = edge_index

        if self.use_edges_info:
            w = torch.cat([x[row,:], x[col,:], edge_attr], dim=-1)
        else:
            w = torch.cat([x[row,:], x[col,:]], dim=-1)

        w = self.weight_nn(w)

        w = gnn_utils.softmax(
            src = w,
            index = row,
            num_nodes = x.size(0),
            dim = 0
        )

        return w

    def message_and_aggregate(self, 
        x : Tensor,
        edge_index : Tensor,
        edge_attr : Optional[Tensor] = None,     
        attn_weights : Optional[Tensor] = None                   
    ) -> Tensor:

        row, col = edge_index
        src = torch.cat([x[col,:], edge_attr], dim=-1) if self.use_edges_info else x[col,:]
        src = self.w_message(src)
        
        if attn_weights is not None:
            src = src * attn_weights

        messages = scatter(
            src = src,
            index = row,
            dim = 0,
            dim_size = x.size(0),
            reduce = ('mean' if attn_weights is None else 'sum')
        )

        return messages

    def edge_update(self,
        x : Tensor,
        edge_index : Tensor,
        edge_attr : Optional[Tensor] = None                
    ) -> Tensor:
        
        row, col = edge_index
        h, t = x[row], x[col]
        r = edge_attr
        
        e = self.w_edge(torch.cat([h, t, r],dim = -1))
        
        e = self.bn_edge(e)
        e = self.act(e)

        return e

    def node_update(self,
        x : Tensor,
        messages : Tensor         
    ) -> Tensor:

        r = self.w_res(x)
        x = messages + x
        x = self.w_node(x)

        if self.res:
            x = x + r

        x = self.bn_node(x)
        x = self.act(x)

        return x

    def forward(self,
        x : Tensor,
        edge_index : Tensor,
        edge_attr : Optional[Tensor],
    ) -> RGNNOutput:

        assert (edge_attr is not None) or not self.use_edges_info

        attn_weights = self.calculate_attn_weights(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr
        )

        messages = self.message_and_aggregate(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            attn_weights=attn_weights
        )

        if edge_attr is not None:

            edge_attr = self.edge_update(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr
            )

        x = self.node_update(
            x=x,
            messages=messages
        )

        return RGNNOutput(x=x, edge_attr=edge_attr)


if __name__ == '__main__':

    N_x, N_e = 100, 200
    D = 16

    x = torch.randn(size = (N_x, D))
    edge_index = torch.randint(0, N_x, (2, N_e))
    edge_attr = torch.randn(N_e, D)

    for use_edges_info in [True, False]:

        for agg in ['mean', 'attn']:

            for root in [True, False]:

                print(f"use_edges_info = {use_edges_info}; agg = {agg}")

                layer = RGNNLayer(
                    in_channels = D,
                    out_channels = D // 2,
                    use_edges_info=use_edges_info,
                    agg=agg,
                    activation="relu",
                    norm=True
                )

                out = layer.forward(x=x, edge_index=edge_index, edge_attr=edge_attr)

                for key, value in out.items():
                    print(f"{key}.shape = {value.shape}")

                print("\n\n")