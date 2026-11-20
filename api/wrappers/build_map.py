from torch import Tensor
from src.commands.build_expertise_map import ExpertiseMapBuilder as _ExpertiseMapBuilder, Config as _Config
from src.shared.platforms import LLM
from src.shared.command_tools import add_handlers
from api.constants import SUBDOMAINS_LABEL_KEY
from typing import Dict

class Config(_Config):
    api_key : str

@add_handlers
class ExpertiseMapBuilder(_ExpertiseMapBuilder):

    def domains(self, labels : Dict[str, Tensor]) -> Tensor:
        return labels[SUBDOMAINS_LABEL_KEY]

    @classmethod
    def from_config(cls, config: Config) -> 'ExpertiseMapBuilder':

        builder = ExpertiseMapBuilder(
            config=config.conf, 
            llm=LLM(config.conf.llm, config.api_key)
        )

        return builder