import dotenv
from typing import List, Dict
from src.shared.command_tools import Stage, StageConfig, add_handlers
from src.shared.file_tools import FileWriter, FileReader, TextfileCache
from returns.pipeline import is_successful
from src.shared.triples_extractors import BaseTriplesExtractor, TriplesExtractorFactory
from pycli import Field
from pydantic import computed_field
from definitions import ENV
from instance import cli

dotenv.load_dotenv(ENV)

class Config(StageConfig):

    type : str = Field(
        default='two-step',
        description=f"The type of the extractor used.",
    )

    llm : str = Field(default="deepseek-chat",description="The LLM to be used for triple extraction.")

    @computed_field
    @property
    def api_key(self) -> str:
        return dotenv.get_key(ENV, f"{self.llm.upper()}_API_KEY")

@cli.register_stage(
    name="extract_triples",
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
        batch_size : int, 
        triple_extractor : BaseTriplesExtractor
    ) -> None:
        super().__init__(reader, writer, batch_size)
        self.triple_extractor = triple_extractor

    def transform(self, doc : Dict) -> Dict:
        return self.transform_chunk([doc])
    
    def transform_chunk(self, chunk : List[Dict]) -> List[Dict]:

        responses = self.triple_extractor.extract([doc['sentence'] for doc in chunk])
        records = []

        for doc,response in zip(chunk, responses):

            if is_successful(response):

                triples = response.unwrap()
                
                for i,triple in enumerate(triples):

                    records.append({
                        "id" : f"{doc['id']}_{i}",
                        "head" : triple.head,
                        "predicate" : triple.predicate,
                        "tail" : triple.tail,
                        "sentence" : doc
                    })
                    
            else:
                ### Update number of failed attempts stats
                pass

        return records
    
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
            triple_extractor=TriplesExtractorFactory.from_llm(
                api_key=config.api_key,
                type=config.type,
                llm_name=config.llm
            )
        )