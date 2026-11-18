import asyncio
from src.shared.registry import registry
from abc import ABC, abstractmethod
from typing import List, Union, TypedDict, Dict, Any
from returns.result import Result
from openai import APIError, RateLimitError
from src.shared.platforms import LLM
from returns.pipeline import is_successful

class Triple(TypedDict):
    head : str
    predicate : str
    tail : str

class BaseTriplesExtractor(ABC):

    @abstractmethod
    def extract(self, texts : List[str]) -> List[Result[List[Triple], Exception]]:
        ...

TriplesExtractorRegistry = registry(BaseTriplesExtractor)

class LLMBasedTripleExtractor(BaseTriplesExtractor):

    def __init__(self, llm : LLM) -> None:
        super().__init__()
        self.llm = llm

    @abstractmethod
    def extract(self, texts : List[str]) -> List[Result[List[Triple], Union[APIError,RateLimitError]]]:
        ...

@TriplesExtractorRegistry.register("one-step-llm:{llm}")
class OneStepLLMTripleExtractor(LLMBasedTripleExtractor):

    def update_kwargs(kwargs : Dict[str, Any], others : Dict[str, Any]) -> Dict[str, Any]:

        return { 
            **kwargs, 
            'llm' : LLM(name=others['llm']) 
        }

    def extract(self, texts : List[str]) -> List[Result[List[Triple], Union[APIError,RateLimitError]]]:

        ouputs = self.llm.batch_make_async_request(
            prompt_name='OneStepTripleExtraction',
            inputs=[
                { "sentence" : text } 
                for text in texts
            ]
        )

        return [output.map(lambda x : x['triples']) for output in ouputs]

@TriplesExtractorRegistry.register("two-step-llm:{llm}")
class TwoStepLLMTripleExtractor(LLMBasedTripleExtractor):

    def update_kwargs(kwargs : Dict[str, Any], others : Dict[str, Any]) -> Dict[str, Any]:
            
        return { 
            **kwargs, 
            'llm' : LLM(name=others['llm']) 
        }

    async def extract_entities(self, text : str) -> Result[List[str], Union[APIError,RateLimitError]]:

        output = await self.llm.make_async_request(
            prompt_name='EntitiesExtraction',
            input={ "text" : text } 
        )

        output = output.map(lambda x : x["entities"])

        return output

    async def extract_triples(self, text : str) -> Result[List[Triple], Union[APIError,RateLimitError]]:

        entities = await self.extract_entities(text)

        if not is_successful(entities):
            return entities

        entities = entities.unwrap()

        output = await self.llm.make_async_request(
            prompt_name='TriplesExtraction',
            input={ 
                "text" : text, 
                "entities" : entities 
            } 
        )

        return output \
            .map(lambda x : x['triples']) \
            .map(lambda x : [t for t in x if (t['head'] in entities) and (t['tail'] in entities)])

    async def batch_extract(self, texts : List[str]) -> List[Result[List[Triple], Union[RateLimitError,APIError]]]:
    
        return await asyncio.gather(*[
            self.extract_triples(text)
            for text in texts
        ])
    
    def extract(self, texts : List[str]) -> List[Result[List[Triple], Union[RateLimitError,APIError]]]:
        return asyncio.run(self.batch_extract(texts))

if __name__ == '__main__':

    examples = ["Machine Learning is used in advanced cancer detection."]

    extractor = TriplesExtractorRegistry.create("one-step-llm:deepseek-flash")
    print(extractor.__class__)
    print(extractor.extract(["Machine Learning is used in advanced cancer detection."]))

    extractor = TriplesExtractorRegistry.create("two-step-llm:deepseek-flash")
    print(extractor.__class__)
    print(extractor.extract(["Machine Learning is used in advanced cancer detection."]))
