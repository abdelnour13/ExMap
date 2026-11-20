from src.shared.command_tools import add_handlers
from src.commands.graph_learning.main import Trainer as _Trainer, Config
from api.wrappers.feature_extraction import Config as FEConfig
from typing import Self
from pydantic import computed_field

@add_handlers
class Trainer(_Trainer):

    @classmethod
    def from_config(cls, config: Config) -> Self:

        trainer = Trainer(
            experiment=config.experiment, 
            config=config.training_config, 
            configs=config.configs
        )

        return trainer