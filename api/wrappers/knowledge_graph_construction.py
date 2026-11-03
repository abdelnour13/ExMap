import os
import json
from typing import Generator, Dict, Self, Tuple, Type, Optional
from src.shared.command_tools import Stage, StageConfig, BaseCommand, OutputDirOption, add_handlers, ConnectionOption, Connection
from multiprocessing.connection import Listener
from src.shared.handlers import Event, EventLevel, ConsoleLogger
from src.helpers import load_json_file
from src.helpers.config import override
from pydantic import BaseModel, computed_field, model_validator
from definitions import rel_path, ANNOTATION_DIR, KGS_DIR
from multiprocessing import set_start_method, Process

ADDR = ("localhost", 0)
set_start_method('spawn', force=True)

from src.commands.knowledge_graph_construction.preprocessing import coreference_resolution, cleaning, sentence_segmentation
from src.commands.knowledge_graph_construction.content_enrichment import annotation, acronym_extraction, keyword_extraction
from src.commands.knowledge_graph_construction.triplet_extraction import triples_extractor, knowledge_completion
from src.commands.knowledge_graph_construction.graph_cleaning import (
    llm_based_predicates_cleaning,
    llm_based_entities_cleaning,
    rule_based_predicates_cleaning,
    entities_cleaning
)

from pycli.field import Field

STAGES : Dict[str, Tuple[Type[Stage], Type[StageConfig]]] = {
    "corpus_cleaning" : (cleaning.CorpusCleaner, cleaning.StageConfig),
    "coreference_resolution" : (coreference_resolution.CoreferenceResolver, coreference_resolution.Config),
    "corpus_annotation" : (annotation.CorpusAnnotatator, annotation.Config),
    "keywords_extraction" : (keyword_extraction.KeywordsExtractor, keyword_extraction.StageConfig),
    "sentence_segmentation" : (sentence_segmentation.SentenceSegmentor, sentence_segmentation.Config),
    "acronyms_extraction" : (acronym_extraction.AcronymsExtractor, acronym_extraction.Config),
    "triples_extraction" : (triples_extractor.TriplesExtractor, triples_extractor.Config),
    "entities_normalization" : (entities_cleaning.EntitiesNormalizer, entities_cleaning.Config),
    "knowledge_completion" : (knowledge_completion.KnowledgeCompletion, knowledge_completion.Config),
    "rule_based_predicate_cleaning" : (rule_based_predicates_cleaning.PredicatesNormalizer, rule_based_predicates_cleaning.Config),
    "llm_based_predicate_cleaning" : (llm_based_predicates_cleaning.IterativePredicatesCleaning, llm_based_predicates_cleaning.Config),
    "llm_based_entities_cleaning" : (llm_based_entities_cleaning.IterativeLLMBasedEntitiesCleaning, llm_based_entities_cleaning.Config)
}

class GlobalConfig(BaseModel):
    domain : str
    batch_size : int
    buffer_size : int
    resume : bool

class StageInfo(BaseModel):
    name : str
    config : StageConfig

class PipeConfig(BaseModel):
    corpus_id : str
    glob: GlobalConfig
    stages: Dict[str, StageInfo]

    @model_validator(mode="before")
    @classmethod
    def validate_stages(cls, data):

        stages = data.get("stages")
        glob = data.get("glob")
        corpus_id = data.get('corpus_id')

        if (
            (stages is None) or 
            (glob is None) or 
            (corpus_id is None)
        ):
            return data

        validated = {}
        input_filename = os.path.join(ANNOTATION_DIR, *corpus_id.split('/'))

        for stage_name, args in stages.items():
            
            name = args.pop('stage')

            _, ConfigClass = STAGES[name]

            config = ConfigClass(**{
                "input_filename" : input_filename,
                "out" : os.path.join(KGS_DIR, data['graph_id'], stage_name),
                **override(glob, args)
            })

            validated[stage_name] = StageInfo(name=name, config=config)

            input_filename = config.output_filename

        data["stages"] = validated

        return data

class Config(OutputDirOption, ConnectionOption):
    graph_id : str = Field(description=f"Name of folder located in {rel_path(KGS_DIR)} and contains a `config.json` file.")
    stage : Optional[str] = Field(description=f"The stage to execute , if no value is provided the whole pipeline will get executed.", default=None)

    def output_dir(self) -> str:
        return os.path.join(KGS_DIR, self.graph_id)
    
    @computed_field
    @property
    def config_path(self) -> str:
        return os.path.join(self.output_dir(), 'config.json')

    @computed_field
    @property
    def conf(self) -> PipeConfig:

        config = load_json_file(self.config_path)

        return PipeConfig(**{
            "graph_id" : self.graph_id,
            **config
        })

def create_and_run(cmd_cls : Type[BaseCommand], cmd_config : StageConfig) -> None:
    cmd = cmd_cls.from_config(cmd_config)
    cmd.remove_handlers(cls=ConsoleLogger)
    cmd.run()  

@add_handlers
class Pipeline(BaseCommand):

    def __init__(self, 
        stages : Dict[str, StageInfo], 
        stage : Optional[str] = None
    ) -> None:
        super().__init__()
        self.stages = stages
        self.stage = stage
        self.listener = Listener(ADDR)

    def __iter__(self) -> Generator[Event, None, None]:

        try:
        
            found = False
            total = len(self.stages) if self.stage is None else 1

            for i, (name, config) in enumerate(self.stages.items()):

                ### Check if the current stage is the one to execute
                if self.stage is not None:

                    if found:
                        break

                    if self.stage == name:
                        found = True
                    else:
                        continue

                yield self.make_info_event(progress=min(i, total-1)/total,message=f"Starting stage {name}")

                ### Get Stage Class & Create output directory
                cls_name = config.name
                cmd_cls = STAGES[cls_name][0]
                os.makedirs(config.config.output_dir(), exist_ok=True)

                ### Start Process
                config.config.connection = Connection(host=self.listener.address[0], port=self.listener.address[1])
                process = Process(target=create_and_run, args=(cmd_cls, config.config))

                try:

                    process.start()

                    with self.listener.accept() as conn:

                        while True:

                            try:

                                event = conn.recv()
                                event['level'] = EventLevel[event['level']]
                                event = Event(**event)
                                event.task = f"{name}::{event.task}"

                                yield event

                            except EOFError:
                                break

                    process.join()

                finally:

                    if process.is_alive():
                        process.terminate()

                    process.join()

                yield self.make_info_event(progress=min(i+1, total)/total,message=f"Completed stage {name}")

        finally:
            self.listener.close()

        yield self.make_info_event(progress=1.0,message=f"Pipeline execution completed.")
        
    @classmethod
    def from_config(cls, config : Config) -> Self:
        instance = Pipeline(stages=config.conf.stages, stage=config.stage)
        return instance

if __name__ == '__main__':
    instance = Pipeline.from_config(config=Config(graph_id="test"))
    instance.run()