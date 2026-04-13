import dotenv
import asyncio
from openai import RateLimitError, APIError
from typing import List, Dict, Iterable
from src.shared.command_tools import Stage, StageConfig, add_handlers
from src.shared.file_tools import FileWriter, FileReader, TextfileCache
from src.shared.llms import get_by_name, LLM
from returns.result import Result
from src.helpers import make_openai_request
from returns.pipeline import is_successful
from pycli import Field
from pydantic import computed_field
from definitions import ENV
from instance import cli
from openai import AsyncOpenAI

dotenv.load_dotenv(ENV)

class Config(StageConfig):

    llm : str = Field(default="deepseek-chat",description="The LLM to be used for triple extraction.")

    @computed_field
    @property
    def api_key(self) -> str:
        return dotenv.get_key(ENV, f"{self.llm.upper()}_API_KEY")

@cli.register_stage(
    name="triples_validation",
    description=(
        "Validate triples using an LLM-as-judge"
    ),
    add_from_conf=False,
    add_save_to=False
) 
@add_handlers()
class TriplesValidator(Stage):

    def __init__(self, 
        reader : FileReader, 
        writer : FileWriter, 
        client : AsyncOpenAI,
        llm : LLM
    ) -> None:
        super().__init__(reader, writer)
        self.client = client
        self.llm = llm

    async def validate(self, doc : Dict) -> Result[Dict, RateLimitError | APIError]:

        messages = [
            {
                "role": "system",
                "content": (
                    "For the task of validating extracted triples from scientific articles abstracts, "
                    "given a triple and the sentence used to extract it , answer the following two questions : \n"
                    "1- Does the triple -assuming no prior knowledge of the sentence- present a usefull and a meaningful fact ? \n"
                    "2- Does the triple present a fact that is present in the given sentence ? \n"
                    "Answer formatting : \n"
                    "- Format the answer as follows : [ANSWER_Q1],[ANSWER_Q2]. Where 'ANSWER_Q1' and 'ANSWER_Q1' should be either 'yes' or 'no'. \n"
                    "- No additional explanation should be provided with the answer.\n"
                ),
            },
            {
                "role": "user",
                "content": (
                    f'The sentence : "{doc['sentence']['sentence']}". \n'
                    f'The triple : ({doc['head']})-[{doc['predicate']}]-({doc['tail']}).'
                )
            }
        ]

        response = await make_openai_request(
            client=self.client,
            model=self.llm.name,
            messages=messages
        )

        if is_successful(response):

            response = response.unwrap()
            completion = response.choices[0].message.content

            answers = [
                answer.strip('\'" ').lower() == 'yes'
                for answer in completion.split(',')
            ]

            doc['validation_result'] = answers

            return Result.from_value(doc)

        else:
            return Result.from_failure(response.failure())

    def transform(self, doc : Dict) -> Dict:
        return self.transform_chunk([doc])
    
    async def async_transform_chunk(self, chunk : Iterable[Dict]) -> List[Result[Dict, RateLimitError | APIError]]:

        return await asyncio.gather(*[
            self.validate(document) 
            for document in chunk
        ])

    def transform_chunk(self, chunk : List[Dict]) -> List[Dict]:

        results =  asyncio.run(self.async_transform_chunk(chunk))
        records = []

        for doc,result in zip(chunk, results):

            if is_successful(result):

                doc = result.unwrap()
                records.append(doc)
                    
            else:
                ### Update number of failed attempts stats
                pass

        return records
    
    @classmethod
    def from_config(cls, config : Config) -> Stage:

        cache = TextfileCache(config.checkpoint_filename)
        llm = get_by_name(config.llm)
        
        return TriplesValidator(
            reader=FileReader
                .from_filename(config.input_filename, cache)
                .batched(config.batch_size),
            writer=FileWriter
                .from_filename(config.output_filename)
                .buffered(config.buffer_size),
            client=AsyncOpenAI(
                api_key=config.api_key,
                base_url=llm.url
            ),
            llm=llm
        )