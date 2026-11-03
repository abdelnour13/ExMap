import os
from typing import Dict
from src.shared.zs_classifiers import LLMBasedZeroShotClassifier, LLM, ZeroShotClassifier
from src.helpers import load_json_file
from src.shared.file_tools import FileReader, FileWriter, TextfileCache
from src.shared.command_tools.vasync import OutputDirOption, ConnectionOption, add_handlers
from src.commands.helpers import DomainConfig
from definitions import ANNOTATION_DIR, CORPUS_DIR
from pydantic import BaseModel
from functools import cached_property
from src.commands.knowledge_graph_construction.content_enrichment.annotation import CorpusAnnotatator as _CorpusAnnotatator

class CorpusAnnotatorConfig(DomainConfig('domain')):
    corpus_id : str
    batch_size : int
    buffer_size : int
    model : str
    domain : str

class Config(OutputDirOption, ConnectionOption):
    annotation_id : str
    api_key : str

    def output_dir(self):
        return os.path.join(ANNOTATION_DIR, self.annotation_id)

    @cached_property
    def config_path(self) -> str:
        return os.path.join(self.output_dir(), 'config.json')

    @cached_property
    def stage_config(self) -> CorpusAnnotatorConfig:
        return CorpusAnnotatorConfig(**load_json_file(self.config_path))

    @cached_property
    def input_filename(self) -> str:
        return os.path.join(CORPUS_DIR, self.stage_config.corpus_id, "data.jsonl")

@add_handlers
class CorpusAnnotatator(_CorpusAnnotatator):

    @classmethod
    def create_classifiers(cls, config : Config) -> Dict[str, ZeroShotClassifier]:

        models = {}

        models[f'llm:{config.stage_config.model}'] = LLMBasedZeroShotClassifier(
            llm=LLM(name=config.stage_config.model, api_key=config.api_key),
            prompt_name="binary",
            labels=[
                {
                    "name" : domain.sub_domain,
                    "description" : domain.definition,
                    "alt_names" : domain.alt_names
                }
                for domain in config.stage_config.domains
            ],
        )

        return models

    @classmethod
    def from_config(cls, config : Config):

        cache = TextfileCache(config.checkpoint_filename)
                
        cmd = CorpusAnnotatator(
            reader=FileReader
                .from_filename(config.input_filename, cache)
                .batched(config.stage_config.batch_size),
            writer=FileWriter
                .from_filename(config.output_filename)
                .buffered(config.stage_config.buffer_size, cache),
            models=cls.create_classifiers(config),
        )

        return cmd
