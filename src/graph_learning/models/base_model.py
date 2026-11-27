from torch import nn
from pydantic import BaseModel
from dataclasses import dataclass
from torch_geometric.data import HeteroData
from typing import Tuple, List, Optional, Self, Dict, Type, TypeVar
from src.graph_learning import losses

T = TypeVar('T')

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

    def num_labels(self, labels : List[str]) -> Dict[str, int]:

        return {
            label : self.data['Article'][label].size(-1)
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
        
        def context(self, batch : HeteroData, batch_size : int) -> Dict[str, losses.BaseLossContext]:
            pass

    __DICT__ : Dict[str, Tuple['TopicModel', Config, Output]] = {}

    @classmethod
    def validate_model_config(cls, model_name : str, config : Dict) -> Config:
        model_cls, model_config_cls, model_output_cls = cls.__DICT__[model_name]
        config = model_config_cls(**config)
        return config

    @classmethod
    def register(cls, model_cls : Type[T]) -> Type[T]:

        cls.__DICT__[model_cls.__name__] = (
            model_cls,
            model_cls.Config,
            model_cls.Output
        )

        return model_cls

    @classmethod
    def create_model(cls,
        name : str,
        config : Dict,
        ctx : Context,
        labels : Optional[List[str]] = None
    ) -> 'TopicModel':

        model_cls, cfg, out = cls.__DICT__[name]

        return model_cls.from_config(
            config=cfg(**config),
            ctx=ctx,
            labels=ctx.num_labels(labels) if labels is not None else None
        )

    @staticmethod
    def from_config(
        config : Config, 
        ctx : Context,
        labels : Optional[Dict[str, int]] = None
    ) -> Self:
        raise NotImplementedError

    def forward(self, 
        data : HeteroData,
        ctx : Optional[IterationContext] = None
    ) -> Output:
        return self.Output()