import torch
from torch import Tensor
from typing import Optional, Tuple, TypedDict
from torch import nn
from src.gnn.layers.activation import Activation
from src.gnn.layers.reduction import reduce

class RGNNOutput(TypedDict):
    x : Tensor
    edge_attr : Optional[Tensor]

    def astuple(self) -> Tuple[Tensor, Tensor]:
        return self.x, self.edge_attr

class RGNNLayer(nn.Module):

    def __init__(self, 
        in_channels : int,
        out_channels : int,
        expect_edges_info : bool = False,
        activation : Optional[str] = None,
        agg : str = None,
        norm : bool = True,
        root : bool = True
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expect_edges_info = expect_edges_info
        self.activation = activation
        self.norm = norm
        self.agg = agg
        self.attn = (agg == 'attn')
        self.root = root

        self.w_res = nn.Linear(in_features=in_channels,out_features=out_channels) if self.root else nn.Identity()
        self.w_node = nn.Linear(in_features=in_channels,out_features=out_channels)
        self.w_edge = nn.Linear(in_features=3 * in_channels,out_features=out_channels) if expect_edges_info else None

        self.bn_node = nn.BatchNorm1d(num_features=out_channels) if self.norm else nn.Identity()
        self.bn_edge= nn.BatchNorm1d(num_features=out_channels) if self.norm and self.expect_edges_info else nn.Identity()
        
        self.act = Activation(name=activation)

        if self.attn:

            if expect_edges_info:

                self.weight_nn = nn.Sequential(*[
                    nn.Linear(in_features=3 * out_channels, out_features=out_channels),
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.Linear(in_features=out_channels, out_features=1)
                ])

            else:

                self.weight_nn = nn.Sequential(*[
                    nn.Linear(in_features=2 * out_channels, out_features=out_channels),
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.Linear(in_features=out_channels, out_features=1)
                ])

    def attn_weights(self, 
        x : Tensor, 
        edge_index : Tensor, 
        edge_attr : Optional[Tensor] = None
    ) -> Optional[Tensor]:

        if not self.attn:
            return None

        if self.expect_edges_info:
            assert edge_attr is not None, f"expect_edges_info == 'True'. but edge_type not provided."

        row, col = edge_index

        if self.expect_edges_info:
            w = torch.cat([x[row,:], x[col,:], edge_attr], dim=-1)
        else:
            w = torch.cat([x[row,:], x[col,:]], dim=-1)

        w = self.weight_nn(w)

        return w

    def compute_messages(self,
        x : Tensor,
        edge_index : Tensor,
        edge_attr : Optional[Tensor] = None
    ) -> Tensor:

        weights = self.attn_weights(x=x,edge_index=edge_index,edge_attr=edge_attr)

        out = reduce(
            x=x,
            edge_index=edge_index,
            num_nodes=x.size(0),
            weights_x=None,
            weights=weights,
            kind=self.agg,
            weights_nn=None
        )

        return out.x

    def update_representations(self,
        x : Tensor,
        edge_index : Tensor,
        edge_attr : Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:

        if self.expect_edges_info:
            assert edge_attr is not None, f"expect_edges_info == 'True' > 1. but edge_type not provided."
            row, col = edge_index
            src, dst = x[row,:], x[col,:]
            edge_attr = self.w_edge(torch.cat([src, dst, edge_attr], dim=-1))

        x = self.w_node(x)

        return x, edge_attr

    def forward(self,
        x : Tensor,
        edge_index : Tensor,
        edge_attr : Optional[Tensor],
    ) -> RGNNOutput:

        r = self.w_res(x)

        x, edge_attr = self.update_representations(x=x,edge_index=edge_index,edge_attr=edge_attr)
        m = self.compute_messages(x=x,edge_index=edge_index,edge_attr=edge_attr)

        x = self.bn_node(x + m)

        if self.root:
            x = r + x

        x = self.act(x)

        if edge_attr is not None:
            edge_attr = self.act(self.bn_edge(edge_attr))

        return RGNNOutput(x=x, edge_attr=edge_attr)


if __name__ == '__main__':

    N_x, N_e = 100, 200
    D = 16

    x = torch.randn(size = (N_x, D))
    edge_index = torch.randint(0, N_x, (2, N_e))
    edge_attr = torch.randn(N_e, D)

    for expect_edges_info in [True, False]:

        for agg in ['mean', 'attn']:

            for root in [True, False]:

                print(f"expect_edges_info = {expect_edges_info}; agg = {agg}")

                layer = RGNNLayer(
                    in_channels = D,
                    out_channels = D // 2,
                    expect_edges_info=expect_edges_info,
                    agg=agg,
                    activation="relu",
                    norm=True
                )

                out = layer.forward(x=x, edge_index=edge_index, edge_attr=edge_attr)

                for key, value in out.items():
                    print(f"{key}.shape = {value.shape}")

                print("\n\n")