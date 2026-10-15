from torch_geometric.data import HeteroData, Data
from torch_geometric.loader import NeighborLoader
from typing import List, Callable, Optional
from src.commands.graph_learning.preprocessing import _subgraph_by_articles

class HybridSampler:

    def __init__(
        self,
        data : HeteroData,
        num_neighbors : List[int],
        batch_size : int,
        shuffle : bool = True,
        transform : Optional[Callable[[HeteroData], HeteroData]] = None,
        repeats : int = 1,
        **kwargs
    ) -> None:
        
        self.data = data
        self.num_neighbors = num_neighbors
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.transform = transform
        self.repeats = repeats

        self._sampler = self._create_sampler(**kwargs)

    def _create_sampler(self, **kwargs) -> NeighborLoader:

        d = Data()

        d.x = self.data['Article'].x
        d.node_id = self.data['Article'].node_id

        d.edge_index = self.data['Article', 'relate-to', 'Article'].edge_index

        return NeighborLoader(
            data=d,
            num_neighbors=self.num_neighbors,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            **kwargs
        )

    def __iter__(self):

        for _ in range(self.repeats):
        
            for batch in self._sampler:

                batch = _subgraph_by_articles(self.data, batch.node_id)

                if self.transform is not None:
                    yield self.transform(batch)
                else:
                    yield batch

    def __len__(self):
        return len(self._sampler) * self.repeats