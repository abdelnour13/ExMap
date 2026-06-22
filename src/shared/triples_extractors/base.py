import asyncio
from abc import ABC, abstractmethod
from typing import List, Union, Dict, TypedDict
from returns.result import Result
from openai import APIError, RateLimitError, AsyncOpenAI
from src.helpers import make_openai_request


class Triple(TypedDict):
    head : str
    predicate : str
    tail : str

class BaseTriplesExtractor(ABC):

    @abstractmethod
    def extract(self, texts : List[str]) -> List[Result[List[Triple], Exception]]:
        ...

class LLMBasedTripleExtractor(BaseTriplesExtractor):

    MAX_ATTEMPTS : int = 3

    def __init__(self, 
        client : AsyncOpenAI,
        model : str
    ) -> None:
        super().__init__()
        self.client = client
        self.model = model

    async def make_request(self, messages : List[Dict]):

        return await make_openai_request(
            self.client,
            self.model,
            messages,
            self.MAX_ATTEMPTS
        )

    @abstractmethod
    async def extract(self, texts : List[str]) -> List[Result[List[Triple], Union[APIError,RateLimitError]]]:
        ...