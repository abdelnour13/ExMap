import asyncio
from abc import ABC, abstractmethod
from typing import List, Union, Dict
from returns.result import Result
from dataclasses import dataclass
from openai import APIError, RateLimitError, AsyncOpenAI

@dataclass
class Triple:
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

        for attempt in range(self.MAX_ATTEMPTS):

            try:
                
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0,  # No randomness
                    top_p=1,  # Disable sampling
                    seed=42,  # Seed
                )

                return Result.from_value(response)
            
            except (RateLimitError, APIError) as e:

                if attempt == self.MAX_ATTEMPTS - 1:
                    return Result.from_failure(e)

                wait = 2 ** (0.8 * attempt)
                await asyncio.sleep(wait)

    @abstractmethod
    def extract(self, texts : List[str]) -> List[Result[List[Triple], Union[APIError,RateLimitError]]]:
        ...