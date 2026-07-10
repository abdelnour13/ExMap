import dotenv
from typing import List, Dict, TypedDict
from src.shared.command_tools import Stage, StageConfig, add_handlers
from src.shared.file_tools import FileWriter, FileReader, TextfileCache, Cache
from returns.pipeline import is_successful
from src.shared.triples_extractors import TriplesExtractorFactory, BaseTriplesExtractor
from pycli import Field
from definitions import ENV
from instance import cli

class Entity(TypedDict):
    name : str
    types : List[str]

dotenv.load_dotenv(ENV)

class Config(StageConfig):

    type : str = Field(
        default='two-step-llm/deepseek-chat',
        description=f"The type of the extractor used.",
    )

@cli.register_stage(
    name="triples_extraction",
    description=(
        "Extract triples from sentences json file using an LLM-Based triples extractor componenet."
    ),
    add_from_conf=False,
    add_save_to=False
) 
@add_handlers
class TriplesExtractor(Stage):

    def __init__(self, 
        reader : FileReader, 
        writer : FileWriter, 
        cache : Cache,
        triple_extractor : BaseTriplesExtractor,
    ) -> None:
        super().__init__(reader, writer, cache)
        self.triple_extractor = triple_extractor

    def transform(self, doc : Dict) -> Dict:
        return self.transform_chunk([doc])
    
    def transform_chunk(self, chunk : List[Dict]) -> List[Dict]:

        responses = self.triple_extractor.extract([doc['sentence'] for doc in chunk])

        errors = []
        records = []
        docs = []

        for doc,response in zip(chunk, responses):

            if is_successful(response):

                triples = response.unwrap()
                docs.append(doc)
                
                for i,triple in enumerate(triples):

                    records.append({
                        "id" : f"{doc['id']}_{i}",
                        "head" : triple["head"],
                        "predicate" : triple["predicate"],
                        "tail" : triple["tail"],
                        "sentence" : doc
                    })
                    
            else:
                errors.append(response.failure())

        return docs, records, errors
    
    @classmethod
    def from_config(cls, config : Config) -> 'TriplesExtractor':

        cache = TextfileCache(config.checkpoint_filename)
        
        return TriplesExtractor(
            reader=FileReader
                .from_filename(config.input_filename, cache)
                .batched(config.batch_size),
            writer=FileWriter
                .from_filename(config.output_filename)
                .buffered(config.buffer_size),
            cache=cache,
            triple_extractor=TriplesExtractorFactory.from_name(config.type),
        )