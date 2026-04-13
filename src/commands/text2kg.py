import os
from typing import Generator, List, OrderedDict, Dict, Optional, Self, Tuple, Type
from src.shared.command_tools import Stage, StageConfig, BaseCommand
from src.shared.handlers import Event, ConsoleLogger
from src.helpers import load_json_file
from src.helpers.config import override
from pydantic import BaseModel, computed_field
from instance import cli
from definitions import DATA_DIR
from multiprocessing import set_start_method, Process
from dataclasses import dataclass

set_start_method('spawn', force=True)

from src.commands.build_kg.cleaning import cleaning
from src.commands.build_kg.preprocessing import coreference_resolution
from src.commands.build_kg.content_enrichment import annotate, acronym_extraction, keyword_extraction
from src.commands.build_kg.preprocessing import sentence_segmentation
from src.commands.build_kg.graph_construction import triples, neo4j
from src.commands.build_kg.triplet_extraction import (
    triples_extractor, 
    validation, 
    entities_mapping, 
    knowledge_completion,
    entities_cleaning,
    predicate_cleaning
)

from pycli.field import Field

STAGES : Dict[str, Tuple[Stage, StageConfig]] = {
    "corpus_cleaning" : (cleaning.CorpusCleaner, cleaning.StageConfig),
    "coreference_resolution" : (coreference_resolution.CoreferenceResolver, coreference_resolution.Config),
    "corpus_annotation" : (annotate.CorpusAnnotatator, annotate.Config),
    "keywords_extraction" : (keyword_extraction.KeywordsExtractor, keyword_extraction.StageConfig),
    "sentence_segmentation" : (sentence_segmentation.SentenceSegmentor, sentence_segmentation.Config),
    "acronyms_extraction" : (acronym_extraction.AcronymsExtractor, acronym_extraction.Config),
    "triples_extraction" : (triples_extractor.TriplesExtractor, triples_extractor.Config),
    "triples_validation" : (validation.TriplesValidator, triples_extractor.Config),
    "entities_normalization" : (entities_cleaning.EntitiesNormalizer, entities_cleaning.Config),
    "knowledge_completion" : (knowledge_completion.KnowledgeCompletion, knowledge_completion.Config),
    "predicates_normalization" : (predicate_cleaning.PredicatesNormalizer, predicate_cleaning.Config),
    "entities_mapping" : (entities_mapping.EntitiesMapping, entities_mapping.Config),
    "graph2triples" : (triples.Graph2Triples, triples.StageConfig),
    "insert2neo4j"  : (neo4j.Neo4J, neo4j.Config)
}

class GlobalConfig(BaseModel):
    domain : str
    batch_size : int
    buffer_size : int
    resume : bool

class Override(BaseModel):
    name : str
    args : Dict

class PipeConfig(BaseModel):
    glob : GlobalConfig
    overrides : List[Override]

class Config(BaseModel):
    graph : str = Field(description="The corpus name the config file is expected to found at $DATA_DIR/$corpus/pipeline_config.json")
    stage : Optional[str] = Field(description="The stage to run, otherwise it will run the whole pipeline.")

    @computed_field
    @property
    def conf(self) -> PipeConfig:
        return PipeConfig(**load_json_file(os.path.join(DATA_DIR, self.graph, 'pipeline_config.json')))

@dataclass
class StageInfo:
    cmd_cls : Type[BaseCommand]
    config : StageConfig

@cli.register_stage(
    name="text2kg",
    description="Run the knowledge graph building pipeline based on the provided config file.",
    add_from_conf=False,
    add_save_to=False
)
class Pipeline(BaseCommand):

    def __init__(self, stages : OrderedDict[str, StageInfo]) -> None:
        self.stages = stages

    def __iter__(self) -> Generator[Event, None, None]:
        
        for i, (name, stage) in enumerate(self.stages.items()):

            print(f"({i+1}/{len(self.stages)}) Starting stage : {name}.\n")

            p = Process(target=self.create_and_run, args=(stage.cmd_cls, stage.config))
            p.start()
            p.join()
        
        yield from []

    @classmethod
    def from_config(cls, config : Config) -> Self:

        overrides = {
            override.name : override.args
            for override in config.conf.overrides
        }

        commands = OrderedDict()

        input_filename = os.path.join(DATA_DIR, config.graph, "raw", "data.jsonl")

        for stage_name in overrides.keys():

            output_dir = os.path.join(DATA_DIR, config.graph, stage_name)

            cmd_cls, cmd_config_cls = STAGES[stage_name]

            cmd_config = cmd_config_cls(
                input_filename=input_filename,
                output_dir=output_dir,
                **override(config.conf.glob.model_dump(), overrides[stage_name]),
            )

            commands[stage_name] = StageInfo(cmd_cls=cmd_cls, config=cmd_config)

            input_filename = cmd_config.output_filename

            if stage_name == config.stage:
                break

        return Pipeline(stages={ config.stage : commands[config.stage]}) \
            .add_handler(ConsoleLogger("Text2KG"))
    
    @staticmethod
    def create_and_run(cmd_cls : Type[BaseCommand], cmd_config : StageConfig) -> None:
        cmd = cmd_cls.from_config(cmd_config)
        cmd.run()        
