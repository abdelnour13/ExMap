from typing import List, Dict
from src.shared.command_tools import Stage, BaseConfig, command
from src.shared.file_tools import FileWriter, FileReader
from returns.pipeline import is_successful
from src.shared.triples_extractors import BaseTriplesExtractor, TriplesExtractorFactory
from pycli import Field, Model
from instance import cli

class Config(BaseConfig):

    type : str = Field(
        default='two-step',
        description=f"The type of the extractor used.",
    )

    llm : str = Field(default="deepseek-chat",description="The LLM to be used for triple extraction.")

    api_key : str = Field(description="The Api Key needed to use the LLM through OpenAI API.")

class TriplesExtractor(Stage):

    def __init__(self, 
        reader : FileReader, 
        writer : FileWriter, 
        batch_size : int, 
        buffer_size : int,
        triple_extractor : BaseTriplesExtractor
    ) -> None:
        super().__init__(reader, writer, batch_size, buffer_size)
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
                        "document_id" : doc['document_id'],
                        "sentence_id" : doc['id'],
                        "sentence" : doc['sentence'],
                        "head" : triple.head,
                        "predicate" : triple.predicate,
                        "tail" : triple.tail
                    })
                    
            else:
                ### Update number of failed attempts stats
                pass

        return records

@cli.register_command(
    name="extract_triples",
    description=(
        "Extract triples from sentences json file using an LLM-Based triples extractor componenet."
    ),
    model=Model(Config, spread=False),
    add_from_conf=False,
    add_save_to=False
) 
@command
def extract_triples(config : Config) -> 'TriplesExtractor':
    
    return TriplesExtractor(
        reader=FileReader.from_filename(config.input_filename),
        writer=FileWriter.from_filename(config.output_filename),
        batch_size=config.batch_size,
        buffer_size=config.buffer_size,
        triple_extractor=TriplesExtractorFactory.from_llm(
            api_key=config.api_key,
            type=config.type,
            llm_name=config.llm
        )
    )