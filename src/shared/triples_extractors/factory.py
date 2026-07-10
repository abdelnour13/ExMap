from src.shared.env import env
from .two_step_llm import TwoStepLLMTriplesExtractor
from .one_step_llm import OneStepLLMTriplesExtractor
from .base import BaseTriplesExtractor

class TriplesExtractorFactory:
    
    @staticmethod
    def from_name(name : str) -> BaseTriplesExtractor:

        type_, llm = name.split('/')

        if type_ == 'one-step-llm':

            return OneStepLLMTriplesExtractor(
                client=env.llms[llm].async_client,
                model=llm
            )
        
        elif type_ == 'two-step-llm':

            return TwoStepLLMTriplesExtractor(
                client=env.llms[llm].async_client,
                model=llm
            )
        
        else:
            raise ValueError(f"Unrecognized triple extractor type {type_}")