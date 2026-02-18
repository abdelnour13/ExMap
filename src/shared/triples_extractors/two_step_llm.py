import asyncio
from .base import Triple, LLMBasedTripleExtractor
from openai import RateLimitError, APIError
from returns.result import Result
from typing import List, Union, Tuple, Optional
from returns.pipeline import is_successful

ENTITIES_PROMPT = (
"For the task of constructing a knowledge graph from abstracts of scientific articles, you will be provided with a text fragment "
"and are required to extract the valid entities from the provided text. Valid entities should abide by the following rules : \n"
"1. An Entity must be clearly identifiable in the provided text.\n"
"2. An Entity should represent a well-defined concept or a characteristic / property of that concept.\n"
"3. The Entity name should only include adjectives or modifiers that are intrinsic to the identity or meaning of the entity, "
"not subjective or contextual descriptors. For example: \n"
"   - 'most discussed challenges in KG embedding' is not acceptable, but 'KG embedding' is acceptable; the 'most discussed' part is subjective, "
"while the challenges should be expressed by means of the predicate 'has-challenge'.\n"
"4. An Entity should not contain more than one concept; it should be a unique concept.\n"
"5. The answer should be expressed as comma-separated values: [ENTITY_1],[ENTITY_2],...[ENTITY_N], and no additional explanation "
"should be provided with the answer."
)

TRIPLES_PROMPT = (
"For the task of constructing a knowledge graph from abstracts of scientific articles. You will be provided with a text fragment "
"along with a set of entities and the goal is to link between the entities to produce triples that reflect the facts presented in "
"the text, while abiding to the following rule : \n"
"1. A triple is composed of three parts : a head, a predicate and an entity.\n"
"2. The triple should also preseve the logical flow tail => predicate => entity."
"3. A predicate must describe the relationship between the two entities and be an english verb of 1-3 tokens."
"4. A Triple must be loyal to the original text and should not present a fact that is not present in the provided text, do not invent or assume any relations.\n"
"5. The answer must be encoded as follows (pipe seprated) : ([HEAD_1],[PREDICATE_1],[HEAD_2])|([HEAD_1],[PREDICATE_1],[HEAD_2]),...\n"
"6. No additional explanation should be provided with the answer."
"7. If no triples could be extracted return an empty response."
)

class TwoStepLLMTriplesExtractor(LLMBasedTripleExtractor):

    async def extract_entities(self, text : str) -> Result[List[str], Union[RateLimitError,APIError]]:

        messages = [
            {
                "role": "system",
                "content": ENTITIES_PROMPT,
            },
            {
                "role": "user",
                "content": text
            }
        ]

        response = await self.make_request(messages)

        return response \
            .map(lambda response : response.choices[0].message.content) \
            .map(self.parse_entities)

    async def batch_extract_entities(self, batch : List[str]) -> List[Result[List[str], Union[RateLimitError,APIError]]]:
        tasks = [self.extract_entities(text) for text in batch]
        results = await asyncio.gather(*tasks)
        return results
    
    def parse_entities(self, entities : str) -> List[str]:

        return list(map(
            lambda x : x.strip("[]\n "),
            entities.split(',')
        ))
    
    def parse_triple(self, triple : str) -> Optional[Triple]:

        triple = triple.strip('()\n ')

        try:

            head, predicate, tail = triple.split(',')

            head = head.strip('[]\n ')
            predicate = predicate.strip('[]\n ')
            tail = tail.strip('[]\n ')

            return Triple(head,predicate,tail)

        except:
            return None
        
    def _check_entities(self, triple : Triple, entities : List[str]) -> bool:
        entities = [entity.lower() for entity in entities]
        return (triple.head.lower() in entities) and (triple.tail.lower() in entities)
    
    async def extract_triples(self, 
        text : str,
        entities : List[str]
    ) -> Result[List[Triple], Union[RateLimitError,APIError]]:

        messages = [
            {
                "role" : "system",
                "content" : TRIPLES_PROMPT
            },
            {
                "role" : "user",
                "content" : f"""
                    - Text : {text}
                    - Entities : {' | '.join(entities)}
                """
            }
        ]

        response = await self.make_request(messages=messages)

        return response \
            .map(lambda response : response.choices[0].message.content) \
            .map(lambda content : list(
                    filter(
                        lambda x : (x is not None) and self._check_entities(x, entities), 
                        map(self.parse_triple, content.split('|'))
                    )
                )
            )
    
    async def batch_extract_triples(self, batch : List[Tuple[str, List[str]]]) -> List[Result[List[Triple], Union[RateLimitError,APIError]]]:
        tasks = [self.extract_triples(text, entities) for text,entities in batch]
        results = await asyncio.gather(*tasks)
        return results

    async def extract_async(self, texts : List[str]) -> List[Result[List[Triple], Union[RateLimitError,APIError]]]:

        ### Extract Entities
        entities_responses = await self.batch_extract_entities(texts)

        ### Prepare Inputs for Triples Extraction
        inputs : List[Tuple[str,List[str]]] = []
        triples_responses : List[Result[List[Triple], Union[RateLimitError,APIError]]] = []

        for text,response in zip(texts, entities_responses):

            if is_successful(response):
                response = response.unwrap()
                inputs.append((text,response))
            else:
                triples_responses.append(Result.from_failure(response.failure()))

        ### Extract Triples
        triples_responses.extend(await self.batch_extract_triples(inputs))

        return triples_responses
    
    def extract(self, texts : List[str]) -> List[Result[List[Triple], Union[RateLimitError,APIError]]]:
        return asyncio.run(self.extract_async(texts))