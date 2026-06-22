from openai import AsyncOpenAI
from src.shared.llms import LLM
from .two_step_llm import LLMBasedTripleExtractor, TwoStepLLMTriplesExtractorWithOntology
from .one_step_llm import OneStepLLMTriplesExtractor
from typing import Literal

LLMBasedType = Literal['two-step','one-step']

class TriplesExtractorFactory:
    
    @staticmethod
    def from_llm(  
        api_key : str,
        type : LLMBasedType = 'two-step',
        llm_name : str = 'deepseek-chat',
    ) -> LLMBasedTripleExtractor:
        
        llm = LLM.from_name(llm_name)

        if type == 'two-step':

            return TwoStepLLMTriplesExtractorWithOntology(
                client=AsyncOpenAI(
                    api_key=api_key,
                    base_url=llm.url
                ),
                model=llm.name
            )
        
        elif type == "one-step":

            return OneStepLLMTriplesExtractor(
                client=AsyncOpenAI(
                    api_key=api_key,
                    base_url=llm.url
                ),
                model=llm.name
            )
        
        else:
            raise ValueError(f"Unavailable type : {type}")