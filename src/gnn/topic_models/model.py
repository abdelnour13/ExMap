from torch import nn
from pydantic import BaseModel
from dataclasses import dataclass
from torch_geometric.data import HeteroData
from typing import Tuple, List, Optional, Self, Dict

@dataclass
class Context:
    data : HeteroData

    @property
    def num_articles_features(self) -> int:
        return self.data['Article'].x.size(-1)

    @property
    def num_entities_features(self) -> int:
        return self.data['Entity'].x.size(-1)

    @property
    def num_articles(self) -> int:
        return self.data['Article'].x.size(0)

    @property
    def num_entities(self) -> int:
        return self.data['Entity'].x.size(0)

    @property
    def num_articles_relations(self) -> int:
        return self.data['Article','relate-to', 'Article'].edge_type.max() + 1

    @property
    def num_entities_relations(self) -> int:
        return len(self.data.predicates2id)

    @property
    def num_relations(self) -> Tuple[int, int]:
        return self.num_entities_relations, self.num_articles_relations

    @property
    def num_features(self) -> Tuple[int, int]:
        return self.num_entities_features, self.num_articles_features

    @property
    def num_labels(self, labels : List[str]) -> Dict[str, int]:

        return {
            label : self.data['Article'][label].size(dim=-1)
            for label in labels
        }

@dataclass
class IterationContext:
    num_epochs : int
    epoch : int
    iterations_per_epoch : int
    iteration : int
    training : bool

    @property
    def total_iterations(self) -> int:
        return self.iterations_per_epoch * self.num_epochs

class TopicModel(nn.Module):

    class Config(BaseModel):
        pass

    @dataclass
    class Output:
        pass

    @staticmethod
    def from_config(
        config : Config, 
        ctx : Context,
        labels : Optional[List[str]] = None
    ) -> Self:
        raise NotImplementedError

    def forward(self, 
        data : HeteroData,
        ctx : Optional[IterationContext]
    ) -> Output:
        return self.Output()