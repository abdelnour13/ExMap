from .constrastive_gnn import ConstrastiveModel, ConstrastiveModelConfig, ConstrastiveModelOutput
from .kg_encoder import KGEncoder, KGEncoderConfig, KGEncoderOutput
from .reconstructive_gnn import ReconstructiveModel, ReconstructiveModelConfig, ReconstructiveModelOutput
from .mlp import SparseArticleEncoder, SparseArticleEncoderConfig, SparseArticleEncoderOutput
from .output import Model, BaseModelOutput
from typing import TypedDict, Dict, Type, Tuple
from pydantic import BaseModel

_registery : Dict[str, Tuple[Type[Model], Type[BaseModel], Type[BaseModelOutput]]] = {
    'ConstrastiveModel' : (ConstrastiveModel, ConstrastiveModelConfig, ConstrastiveModelOutput),
    'KGEncoder' : (KGEncoder, KGEncoderConfig, KGEncoderOutput),
    'ReconstructiveModel' : (ReconstructiveModel, ReconstructiveModelConfig, ReconstructiveModelOutput),
    'SparseArticleEncoder' : (SparseArticleEncoder, SparseArticleEncoderConfig, SparseArticleEncoderOutput)
}

class ModelConfig(TypedDict):
    model_name: str
    model_params : Dict

def validate_model_config(model_name : str, config : Dict):
    model_cls, model_config_cls, model_output_cls = _registery[model_name]
    model_config_cls(**config)
    return config

def create_model(model_name : str, config : Dict, **kwargs):

    model_cls, model_config_cls, model_output_cls = _registery[model_name]

    new_kwargs = {}

    for key, value in kwargs.items():

        if key in model_cls.from_config.__annotations__:
            new_kwargs[key] = value

    return model_cls.from_config(
        **new_kwargs,
        config=model_config_cls(**config)
    )