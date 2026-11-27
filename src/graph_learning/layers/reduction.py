from torch import nn, Tensor
from torch_geometric import utils as gnn_utils
from typing import Literal, Optional, Tuple
from torch_scatter import scatter
from dataclasses import dataclass


READOUT = Literal['max', 'attn', 'mean']

@dataclass
class ReductionOutput:
    x : Tensor
    weights : Optional[Tensor]
    norm_weights : Optional[Tensor]

    def astuple(self) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        return self.x, self.weights, self.norm_weights

def reduce(
    x : Tensor,
    edge_index : Tensor,
    num_nodes : int,
    weights_x : Optional[Tensor] = None,
    weights : Optional[Tensor] = None, # unormalized weights
    *,
    kind : READOUT,
    weights_nn : Optional[nn.Module] = None,
) -> ReductionOutput:

    row, col = edge_index
    
    if kind != 'attn':

        x = scatter(
            src=x[row,:],
            index=col,
            dim=0,
            dim_size=num_nodes,
            reduce=kind
        )

        return ReductionOutput(x=x, weights=None, norm_weights=None)

    weights_x = x if weights_x is None else weights_x

    if weights is None and weights_nn is None:
        raise Exception(f"when kind = 'attn' you should at least provide weights_nn or on the fly weights.")

    if weights is None:
        weights = weights_nn(weights_x)

    norm_weights = gnn_utils.softmax(
        src=weights[row,:],
        index=col,
        num_nodes=num_nodes,
        dim=0
    )

    x = scatter(
        src=norm_weights * x[row,:],
        index=col,
        dim=0,
        dim_size=num_nodes,
        reduce='sum'
    )

    return ReductionOutput(x=x, weights=weights, norm_weights=norm_weights)

class Reduction(nn.Module):

    def __init__(
        self, 
        kind : Literal['max', 'mean', 'attn'],
        weights_nn : Optional[nn.Module]
    ) -> None:
        super().__init__()

        self.kind = kind
        self.weights_nn = weights_nn

    def forward(self, 
        x : Tensor,
        edge_index : Tensor,
        num_nodes : int,
        weights_x : Optional[Tensor] = None,
        weights : Optional[Tensor] = None,
    ) -> ReductionOutput:

        return reduce(
            x=x,
            edge_index=edge_index,
            num_nodes=num_nodes,
            weights_x=weights_x,
            weights=weights,
            kind=self.kind,
            weights_nn=self.weights_nn
        )