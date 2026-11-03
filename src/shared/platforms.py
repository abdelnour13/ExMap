import asyncio
from dataclasses import dataclass
from openai import Client, AsyncClient
from typing import Any, Dict, List
from functools import cached_property
from src.envirenment import env
from src.shared.prompts import PromptsManager, Result, JSONDecodeError, ValidationError, RateLimitError, APIError

__URLS__ = {
    "deepseek" : "https://api.deepseek.com",
    "gemini" : "https://generativelanguage.googleapis.com/v1beta/openai/"
}

@dataclass(frozen=True)
class LLM:
    name : str
    api_key : str

    @property
    def platform(self) -> str:
        return self.name.split('-')[0]

    @property
    def url(self) -> str:
        return __URLS__[self.platform]

    @cached_property
    def client(self) -> Client:

        return Client(
            api_key=env.get_api_key(self.platform),
            base_url=self.url
        )

    @cached_property
    def async_client(self) -> Client:

        return AsyncClient(
            api_key=env.get_api_key(self.platform),
            base_url=self.url
        )

    async def make_async_request(self, 
        prompt_name: str,
        input: Dict[str, Any],
        lang: str = 'en',
        validate_input: bool = True,
        validate_output: bool = True                 
    ) -> Result[Dict, APIError | RateLimitError | JSONDecodeError | ValidationError]:

        result = await PromptsManager.make_request(
            prompt_name=prompt_name,
            client=self.async_client,
            model_name=self.name,
            input=input,
            lang=lang,
            validate_input=validate_input,
            validate_output=validate_output
        )

        return result

    async def _gather_make_async_request(self, 
        prompt_name: str,
        inputs: List[Dict[str, Any]],
        lang: str = 'en',
        validate_input: bool = True,
        validate_output: bool = True                 
    ) -> List[Result[Dict, APIError | RateLimitError | JSONDecodeError | ValidationError]]:

        result = await asyncio.gather(*[
            PromptsManager.make_request(
                prompt_name=prompt_name,
                client=self.async_client,
                model_name=self.name,
                input=input,
                lang=lang,
                validate_input=validate_input,
                validate_output=validate_output
            )
            for input in inputs
        ])

        return result

    async def batch_make_async_request(self, 
        prompt_name: str,
        inputs: List[Dict[str, Any]],
        lang: str = 'en',
        validate_input: bool = True,
        validate_output: bool = True                 
    ) -> List[Result[Dict, APIError | RateLimitError | JSONDecodeError | ValidationError]]:
        
        return asyncio.run(self._gather_make_async_request(
            prompt_name=prompt_name,
            inputs=inputs,
            lang=lang,
            validate_output=validate_output,
            validate_input=validate_input
        ))