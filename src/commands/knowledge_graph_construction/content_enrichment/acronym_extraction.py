import re
from src.shared.file_tools import FileWriter, FileReader, TextfileCache
from src.shared.command_tools import Stage, StageConfig, add_handlers
from src.shared.handlers import Event, EventLevel
from pydantic import Field
from typing import Dict, List
from instance import cli
from src.methods.acronyms_extraction import AcronymExtractor, AcronymExtractorRegistry
from returns.pipeline import is_successful

class Config(StageConfig):

    model : str = Field(
        default="qa:timpal0l/mdeberta-v3-base-squad2",
        description="The method used for Acronym Extraction."
    )

    ### Default threshold was modified
    device : int = Field(
        default=-1,
        description="The device to run the model on if available."
    )

@cli.register_stage(
    name="acronyms_extraction",
    description=(
        "Extract acronyms from the documents relying on a method that combines regular expressions "
        "and QA models."
    ),
    add_from_conf=False,
    add_save_to=False
)
@add_handlers
class AcronymsExtractor(Stage):

    def __init__(self, 
        reader : FileReader, 
        writer : FileWriter, 
        extractor : AcronymExtractor
    ) -> None:
        super().__init__(reader, writer)

        self.extractor = extractor

    def before_start(self):

        yield Event(
            progress=0.0,
            level=EventLevel.INFO,
            extra={
                'total_processed' : 0,
                'total_saved' : 0,
                'total_skipped' : 0
            },
            message=f"Loading {self.extractor.__class__.__name__}..."
        )

        self.extractor.load()

        yield Event(
            progress=0.0,
            level=EventLevel.INFO,
            extra={
                'total_processed' : 0,
                'total_saved' : 0,
                'total_skipped' : 0
            },
            message=f"{self.extractor.__class__.__name__} Loaded."
        )
    
    def transform(self, doc : Dict) -> Dict:
        return self.transform_chunk([doc])
        
    def transform_chunk(self, chunk : List[Dict]) -> List[Dict]:

        responses = self.extractor.extract([
            doc['sentence'] 
            for doc in chunk
        ])

        errors = []
        records = []
        docs = []

        for doc,response in zip(chunk, responses):

            if is_successful(response):

                acronyms = response.unwrap()

                docs.append(doc)    

                records.append({
                    **doc,
                    "acronyms" : [
                        *doc.get('acronyms', []),
                        *acronyms
                    ]
                })
            else:
                errors.append(response.failure())

        return docs, records, errors

    @classmethod
    def from_config(cls, config : Config) -> Stage:

        cache = TextfileCache(config.checkpoint_filename)
        
        return AcronymsExtractor(
            reader=FileReader
                .from_filename(config.input_filename, cache)
                .batched(config.batch_size),
            writer=FileWriter
                .from_filename(config.output_filename)
                .buffered(config.buffer_size, cache),
            extractor=AcronymExtractorRegistry.create(
                name=config.model,
                device=config.device
            )
        )

