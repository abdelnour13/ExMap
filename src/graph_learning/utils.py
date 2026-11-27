
import torch
from dataclasses import dataclass
from typing import Optional, Tuple
from torch import Tensor
from torch_sparse import SparseTensor
from functools import cached_property

@dataclass(frozen=True)
class EdgeIndex:
    row : Tensor
    col : Tensor
    num_nodes : Tuple[int, int]
    value : Optional[Tensor] = None

    @cached_property
    def edge_index(self) -> Tensor:
        return torch.stack([self.row, self.col], dim=0)

@dataclass(frozen=True)
class AdjMatrix:
    _dense: Optional[Tensor] = None
    _sparse: Optional[SparseTensor] = None
    _edge_index: Optional[EdgeIndex] = None

    def get_default(self) -> Tensor | SparseTensor | EdgeIndex:

        if self._edge_index is not None:
            return self._edge_index
        elif self._sparse is not None:
            return self._sparse
        else:
            return self._dense

    def __post_init__(self):
        
        v = (
            int(self._dense is not None)
            + int(self._sparse is not None)
            + int(self._edge_index is not None)
        )

        if v != 1:
            raise ValueError("Must specify exactly one value.")

    @property
    def dense(self) -> Tensor:
        
        if self._dense is not None:
            return self._dense

        if self._sparse is not None:
            dense = self._sparse.to_dense(dtype=torch.float)

        else:
            edge_index = self._edge_index
            dense = SparseTensor(
                row=edge_index.row,
                col=edge_index.col,
                value=edge_index.value,
                sparse_sizes=edge_index.num_nodes
            ).to_dense()

        object.__setattr__(self, "_dense", dense)
        return dense

    @property
    def edge_index(self) -> EdgeIndex:

        if self._edge_index is not None:
            return self._edge_index

        if self._dense is not None:
            edge_index = self._dense.argwhere().t()
            row, col = edge_index
            edge_weight = self._dense[row, col]
            num_nodes = tuple(self._dense.shape)

        else:
            row, col, edge_weight = self._sparse.coo()
            num_nodes = self._sparse.sparse_sizes()

        edge_index = EdgeIndex(
            row=row,
            col=col,
            num_nodes=num_nodes,
            value=edge_weight
        )  

        object.__setattr__(self, "_edge_index", edge_index)
        return edge_index

    @property
    def sparse(self) -> SparseTensor:

        if self._sparse is not None:
            return self._sparse

        if self._dense is not None:
            edge_index = self._dense.argwhere().t()
            row, col = edge_index
            edge_weight = self._dense[row, col]
            num_nodes = tuple(self._dense.shape)

        else:
            edge_index = self._edge_index
            row = edge_index.row
            col = edge_index.col
            edge_weight = edge_index.value
            num_nodes = edge_index.num_nodes

        sparse = SparseTensor(
            row=row,
            col=col,
            value=edge_weight,
            sparse_sizes=num_nodes
        )

        object.__setattr__(self, "_sparse", sparse)

        return sparse

if __name__ == '__main__':

    N, E = 10, 30

    edge_index = torch.randint(low=0, high=N, size=(2, E))

    adj = AdjMatrix(_edge_index=(edge_index, None))

    print(adj.edge_index)
    print(adj.sparse)
    print(adj.dense)