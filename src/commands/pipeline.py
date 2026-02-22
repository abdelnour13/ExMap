import os
from typing import Generator, List, OrderedDict, Dict, Optional, Self, Tuple, Type
from src.shared.command_tools import Stage, StageConfig, BaseCommand
from src.shared.handlers import Event
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
from src.commands.build_kg.triplet_extraction import triples_extractor, triples_postprocessing, validation, entities_metadata, entities_mapping
from src.commands.build_kg.graph_construction import triples, neo4j

STAGES : Dict[str, Tuple[Stage, StageConfig]] = {
    "cleaning" : (cleaning.CorpusCleaner, cleaning.StageConfig),
    "coreference_resolution" : (coreference_resolution.CoreferenceResolver, coreference_resolution.Config),
    "annotate" : (annotate.DocumentsAnnotator, annotate.Config),
    "keyword_extraction" : (keyword_extraction.KeywordExtractor, keyword_extraction.StageConfig),
    "sentence_segmentation" : (sentence_segmentation.SentenceSegmenter, sentence_segmentation.Config),
    "acronym_extraction" : (acronym_extraction.AcronymExtractor, acronym_extraction.Config),
    "extract_triples" : (triples_extractor.TriplesExtractor, triples_extractor.Config),
    "triples_validation" : (validation.TriplesLLMValidator, triples_extractor.Config),
    "clean_triples" : (triples_postprocessing.TriplesPostprocessor, triples_postprocessing.Config),
    "entities_mapping" : (entities_mapping.EntityMapper, entities_mapping.Config),
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
    graph : str
    stage : Optional[str]

    @computed_field
    @property
    def conf(self) -> PipeConfig:
        return PipeConfig(**load_json_file(os.path.join(DATA_DIR, self.graph, 'pipeline_config.json')))

@dataclass
class StageInfo:
    cmd_cls : Type[BaseCommand]
    config : StageConfig

@cli.register_stage(
    name="pipeline",
    description="Run the pipeline based on the provided config file.",
    add_from_conf=False,
    add_save_to=False
)
class Pipeline(BaseCommand):

    def __init__(self, 
        stages : OrderedDict[str, StageInfo]
    ) -> None:
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

        return Pipeline(stages={ config.stage : commands[config.stage]})
    
    @staticmethod
    def create_and_run(cmd_cls : Type[BaseCommand], cmd_config : StageConfig) -> None:
        cmd = cmd_cls.from_config(cmd_config)
        cmd.run()        
