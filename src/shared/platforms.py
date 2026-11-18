import asyncio
from dataclasses import dataclass
from openai import Client, AsyncClient
from typing import Any, Dict, List
from functools import cached_property
from src.envirenment import env
from src.shared.prompts import PromptsManager, RequestReturnType

__URLS__ = {
    "deepseek" : "https://api.deepseek.com",
    "gemini" : "https://generativelanguage.googleapis.com/v1beta/openai/"
}

@dataclass(frozen=True)
class LLM:
    name : str
    api_key : str = None # Local models don't need it

    @staticmethod
    def available_platforms(self) -> List[str]:
        return list(__URLS__.keys())

    def __post_init__(self) -> None:

        if self.api_key is None:

            api_key = env.get_api_key(self.platform) # currently i'm only using paid providers

            if api_key is None:
                raise Exception(f"Api key for platform {self.platform} is neither provided nor it is registered in `env.json`.")

            object.__setattr__(self, 'api_key', api_key)


    @property
    def platform(self) -> str:
        return self.name.split('-')[0]

    @property
    def url(self) -> str:
        return __URLS__[self.platform]

    @cached_property
    def client(self) -> Client:

        return Client(
            api_key=self.api_key,
            base_url=self.url
        )

    @cached_property
    def async_client(self) -> Client:

        return AsyncClient(
            api_key=self.api_key,
            base_url=self.url
        )

    async def make_async_request(self, 
        prompt_name: str,
        input: Dict[str, Any],
        lang: str = 'en',
        validate_input: bool = True,
        validate_output: bool = True                 
    ) -> RequestReturnType[Dict]:

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
    ) -> List[RequestReturnType[Dict]]:

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

    def batch_make_async_request(self, 
        prompt_name: str,
        inputs: List[Dict[str, Any]],
        lang: str = 'en',
        validate_input: bool = True,
        validate_output: bool = True                 
    ) -> List[RequestReturnType[Dict]]:
        
        return asyncio.run(self._gather_make_async_request(
            prompt_name=prompt_name,
            inputs=inputs,
            lang=lang,
            validate_output=validate_output,
            validate_input=validate_input
        ))