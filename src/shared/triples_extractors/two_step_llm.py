import asyncio
from openai import APIError, RateLimitError, AsyncOpenAI
from .base import Triple, LLMBasedTripleExtractor
from openai import RateLimitError, APIError
from returns.result import Result
from typing import List, Union, Optional, TypedDict, Dict
from returns.pipeline import is_successful
from src.env import EntityType

# ENTITIES_PROMPT = (
# "For the task of constructing a knowledge graph from abstracts of scientific articles, you will be provided with a text fragment "
# "and are required to extract the valid entities from the provided text. Valid entities should abide by the following rules : \n"
# "1. An Entity must be clearly identifiable in the provided text.\n"
# "2. An Entity should represent a well-defined concept or a characteristic / property of that concept.\n"
# "3. The Entity name should only include adjectives or modifiers that are intrinsic to the identity or meaning of the entity, "
# "not subjective or contextual descriptors. For example: \n"
# "   - 'most discussed challenges in KG embedding' is not acceptable, but 'KG embedding' is acceptable; the 'most discussed' part is subjective, "
# "while the challenges should be expressed by means of the predicate 'has-challenge'.\n"
# "4. An Entity should not contain more than one concept; it should be a unique concept.\n"
# "5. The answer should be expressed as comma-separated values: [ENTITY_1],[ENTITY_2],...[ENTITY_N], and no additional explanation "
# "should be provided with the answer."
# )

# TRIPLES_PROMPT = (
# "For the task of constructing a knowledge graph from abstracts of scientific articles. You will be provided with a text fragment "
# "along with a set of entities and the goal is to link between the entities to produce triples that reflect the facts presented in "
# "the text, while abiding to the following rule : \n"
# "1. A triple is composed of three parts : a head, a predicate and an entity.\n"
# "2. The triple should also preseve the logical flow tail => predicate => entity."
# "3. A predicate must describe the relationship between the two entities and be an english verb of 1-3 tokens."
# # "4. Add the verb 'is'/'are' when the following predicates are encountered : 'challenge for'."
# "4. A Triple must be loyal to the original text and should not present a fact that is not present in the provided text, do not invent or assume any relations.\n"
# "5. The answer must be encoded as follows (pipe seprated) : ([HEAD_1],[PREDICATE_1],[HEAD_2])|([HEAD_1],[PREDICATE_1],[HEAD_2]),...\n"
# "6. No additional explanation should be provided with the answer."
# "7. If no triples could be extracted return an empty response."
# )

# class TwoStepLLMTriplesExtractor(LLMBasedTripleExtractor):

#     async def extract_entities(self, text : str) -> Result[List[str], Union[RateLimitError,APIError]]:

#         messages = [
#             {
#                 "role": "system",
#                 "content": ENTITIES_PROMPT,
#             },
#             {
#                 "role": "user",
#                 "content": text
#             }
#         ]

#         response = await self.make_request(messages)

#         return response \
#             .map(lambda response : response.choices[0].message.content) \
#             .map(self.parse_entities)

#     async def batch_extract_entities(self, batch : List[str]) -> List[Result[List[str], Union[RateLimitError,APIError]]]:
#         tasks = [self.extract_entities(text) for text in batch]
#         results = await asyncio.gather(*tasks)
#         return results
    
#     def parse_entities(self, entities : str) -> List[str]:

#         return list(map(
#             lambda x : x.strip("[]\n "),
#             entities.split(',')
#         ))
    
#     def parse_triple(self, triple : str) -> Optional[Triple]:

#         triple = triple.strip('()\n ')

#         try:

#             head, predicate, tail = triple.split(',')

#             head = head.strip('[]\n ')
#             predicate = predicate.strip('[]\n ')
#             tail = tail.strip('[]\n ')

#             return Triple(head,predicate,tail)

#         except:
#             return None
        
#     def _check_entities(self, triple : Triple, entities : List[str]) -> bool:
#         entities = [entity.lower() for entity in entities]
#         return (triple.head.lower() in entities) and (triple.tail.lower() in entities)
    
#     async def extract_triples(self, 
#         text : str,
#         entities : List[str]
#     ) -> Result[List[Triple], Union[RateLimitError,APIError]]:

#         messages = [
#             {
#                 "role" : "system",
#                 "content" : TRIPLES_PROMPT
#             },
#             {
#                 "role" : "user",
#                 "content" : f"""
#                     - Text : {text}
#                     - Entities : {' | '.join(entities)}
#                 """
#             }
#         ]

#         response = await self.make_request(messages=messages)

#         return response \
#             .map(lambda response : response.choices[0].message.content) \
#             .map(lambda content : list(
#                     filter(
#                         lambda x : (x is not None) and self._check_entities(x, entities), 
#                         map(self.parse_triple, content.split('|'))
#                     )
#                 )
#             )
    
#     async def batch_extract_triples(self, batch : List[Tuple[str, List[str]]]) -> List[Result[List[Triple], Union[RateLimitError,APIError]]]:
#         tasks = [self.extract_triples(text, entities) for text,entities in batch]
#         results = await asyncio.gather(*tasks)
#         return results

#     async def extract_async(self, texts : List[str]) -> List[Result[List[Triple], Union[RateLimitError,APIError]]]:

#         ### Extract Entities
#         entities_responses = await self.batch_extract_entities(texts)

#         ### Prepare Inputs for Triples Extraction
#         inputs : List[Tuple[str,List[str]]] = []
#         triples_responses : List[Result[List[Triple], Union[RateLimitError,APIError]]] = []

#         for text,response in zip(texts, entities_responses):

#             if is_successful(response):
#                 response = response.unwrap()
#                 inputs.append((text,response))
#             else:
#                 triples_responses.append(Result.from_failure(response.failure()))

#         ### Extract Triples
#         triples_responses.extend(await self.batch_extract_triples(inputs))

#         return triples_responses
    
#     def extract(self, texts : List[str]) -> List[Result[List[Triple], Union[RateLimitError,APIError]]]:
#         return asyncio.run(self.extract_async(texts))
    

class Entity(TypedDict):
    name : str
    type : str

class TypedTriple(TypedDict):
    head : EntityType
    predicate : str
    tail : EntityType

class TwoStepLLMTriplesExtractorWithOntology(LLMBasedTripleExtractor):

    ENTITIES_PROMPT = (

        "You are an expert NLP assistant specializing in high-quality knowledge graph extraction from scientific article abstracts."
        "Your task is to extract all key technical concepts, domain-specific phrases, and fine-grained information units of type {type} from the provided text fragment. Do not limit extraction to proper nouns or canonical terms."
        "Strictly adhere to the following extraction rules: \n"

        "1. **Extract Meaning-Bearing Units**: Extract noun phrases and technical expressions that carry specific meaning in the passage. These include methods, metrics, system components, stages, tasks, and domain concepts — even if they are not proper nouns or widely known.\n"
        "2. **Text-Bound**: The entities must be clearly unfairness of classification modelsidentifiable and explicitly present within the provided passage.\n"

        "3. **Atomic Concepts**: Each extracted entity must describe a single, standalone concept. Do not group multiple separate concepts joined by conjunctions.\n"
        "   - *Example*: \"generator network and discriminator network\" → extract \"generator network\", \"discriminator network\" Output Format: Return the extracted entities strictly as a comma-separated list: [ENTITY_1],[ENTITY_2],...[ENTITY_N]. Do not include any introductory text, formatting blocks, explanations, or extra commentary.\n"

        "4. **Preserve Passage-Specific Modifiers**: Include adjectives or modifiers that the author deliberately uses to specify a unique subdomain, stage, or context. Do NOT strip away modifiers simply because they are temporal, contextual, or passage-specific. Only exclude modifiers that are purely subjective or evaluative (e.g., \"most discussed\", \"very important\", \"challenging\"). \n"
        "   - *Example (keep)*: \"early-stage optimization\" (the author wrote this instead of just \"optimization\" for a reason) \n"
        "   - *Example (keep)*: \"simulation-driven estimation\" (specifies the type of estimation) \n"
        "   - *Example (remove)*: \"most discussed challenges in KG embedding\" (\"most discussed\" is subjective) \n"

        "5. **No Bias Toward Canonical Terms**: Do not privilege well-known or widely recognized technical terms over passage-specific, fine-grained phrases. If the author introduces a novel framing or a specific compound phrase, extract it as an entity."

        "6. **Acronym Standardization**: If both an acronym and its expansion appear together in the text (regardless of order), format the extracted entity using the canonical form: \"[EXPANSION] (ACRONYM)\"."

        "Output Format: Return the extracted entities strictly as a comma-separated list: [ENTITY_1],[ENTITY_2],...[ENTITY_N]. Do not include any introductory text, formatting blocks, explanations, or extra commentary."
    )

    TRIPLES_PROMPT = """You are an expert NLP assistant specializing in Knowledge Graph extraction. Your task is to extract semantic triples (Subject, Predicate, Object) from a provided text passage.

        ### Strict Constraints
        1. **Entity Filtering:** The `subject` and the `object` of every single triple MUST be explicitly chosen from the **Allowed Entities** list below. 
        2. **Zero Tolerance for Outside Entities:** If a relationship exists in the text but involves an entity NOT on the allowed list, you must ignore it. Do not normalize or guess close matches.
        3. **Factual Accuracy:** Only extract relations explicitly stated in the text. Do not extrapolate or use outside knowledge.
        4. **Predicate Format:** Keep predicates concise, clear, and in lowercase (e.g., "employed_by", "founded", "located_in").

        ### Output Format
        Return the result in the following format : ([HEAD_1],[PREDICATE_1],[TAIL_1])|([HEAD_2],[PREDICATE_2],[TAIL_2])|...etc 
    """

    ENTITIES_CLASSIFICATION_PROMPT = """You are an expert NLP assistant specializing in entities classification. Your ask is to - given a text passage and an entity - to pick the entity type that suits it the best from the following list :

    """

    def parse_entities(self, entities : str) -> List[str]:

        return list(map(
            lambda x : x.strip("[]\n "),
            entities.split(',')
        ))
    
    async def classify_entity_type(self,
        entity: str,
        triples: List[Triple],
        type_schema: List[EntityType],
    ) -> Result[List[str], Union[RateLimitError,APIError]]:


        schema_text = "\n\n".join([
            f"""
                TYPE: {t.type}
                DEFINITION: {t.definition}
            """
            for t in type_schema
        ])

        context = '\n'.join([
            t['head'] + '->' + t['predicate'] + '->' + t['tail']  
            for t in triples
        ])

        prompt = f"""
            You are a high-precision scientific entity type classifier.

            TASK:
            Given an entity and its context, assign the MOST appropriate type(s).

            You MUST:
            - You only suggest 1-3 type, the strongest ones.
            - Prefer the most specific type
            - Treat the provided entity as an atomic concept do not let individual tokens dominate the decision but focus on what the whole entity represents 
            in the text.

            ENTITY:
            {entity}

            CONTEXT:
            {context}

            ### Output Format:
            - Return the relevant types as a comma seperated : TYPE_1,TYPE_2...
            - Ordered from most to least relevant.
            - No additional explanation is required.
        """

        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

        response = await self.make_request(messages)

        if not is_successful(response):
            return Result.from_failure(response.failure())
        
        response = response.unwrap()
        types = response.choices[0].message.content
        types = [type_.strip() for type_ in types.split(',')]

        print(types)

        prompt = f"""
            You are given:

            FREELY EXTRACTED TYPES:
            {', '.join(types)}

            FORMAL TAXONOMY (ONLY valid output labels):
            { ' | '.join([t.type for t in type_schema]) }

            TASK:
            Map each extracted type to the closest label(s) in the formal taxonomy.

            STRICT RULES:
            - Use ONLY labels from the formal taxonomy.
            - Do NOT modify, extend, or annotate labels.
            - Each input type must map to a flat list of labels.
            - Do NOT create subtypes or combined labels.
            - Do NOT use colons, hyphens, or hierarchical formatting.
            - Do NOT insert line breaks inside values.
            - If uncertain, return the closest single label.

            OUTPUT FORMAT (STRICT):
            One line per input type:

            input_type -> LABEL_1, LABEL_2

            No explanations. No extra text.
        """

        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

        response = await self.make_request(messages)

        if not is_successful(response):
            return Result.from_failure(response.failure())
        
        response = response.unwrap()
        types = response.choices[0].message.content
        types = [type_.strip() for type_ in types.split(',')]

        return Result.from_value(types)
    
    async def classify_entities(self,
        entities: List[str],
        triples: List[Triple],
        type_schema: List[EntityType],
    ) -> Result[Dict[str, List[str]], Union[RateLimitError,APIError]]:
        
        results = await asyncio.gather(*[
            self.classify_entity_type(
                entity, 
                [t for t in triples if t['head'] == entity or t['tail'] == entity], 
                type_schema
            ) 
            for entity in entities
        ])

        entity2types = {}

        for result, entity in zip(results, entities):

            if not is_successful(result):
                return Result.from_failure(result.failure())
                        
            entity2types[entity] = result.unwrap()

        return Result.from_value(entity2types)
            
    async def extract_entities(self, passage : str) -> Result[List[str], Union[RateLimitError,APIError]]:

        messages = [
            {
                "role": "system",
                "content": self.ENTITIES_PROMPT
            },
            {
                "role": "user",
                "content": passage
            }
        ]

        response = await self.make_request(messages)

        return response \
            .map(lambda response : response.choices[0].message.content) \
            .map(self.parse_entities)

    
    def parse_triple(self, triple : str) -> Optional[Triple]:

        triple = triple.strip('()\n ')

        try:

            head, predicate, tail = triple.split(',')

            head = head.strip('[]\n ')
            predicate = predicate.strip('[]\n ')
            tail = tail.strip('[]\n ')

            return Triple(head=head,predicate=predicate,tail=tail)

        except Exception as e:
            return None
    
    async def extract_triples(self, 
        text : str,
        entities : List[str]
    ) -> Result[List[Triple], Union[RateLimitError,APIError]]:

        messages = [
            {
                "role" : "system",
                "content" : self.TRIPLES_PROMPT
            },
            {
                "role" : "user",
                "content" : f"""
                    - Passage : {text}
                    - Entities : {' | '.join(entities)}
                """
            }
        ]

        response = await self.make_request(messages=messages)

        response = response \
            .map(lambda response : response.choices[0].message.content) \
            .map(lambda content : list(
                    filter(
                        lambda x : (x is not None) and self._check_entities(x, entities), 
                        map(self.parse_triple, content.split('|'))
                    )
                )
            )
                
        return response
    
    def _check_entities(self, triple : Triple, entities : List[str]) -> bool:
        entities = [entity.lower() for entity in entities]
        return (triple['head'].lower() in entities) and (triple["tail"].lower() in entities)
    
    async def extract_from_passage(self, passage : str, type_schema : List[Entity]) -> Result[List[TypedTriple], Union[RateLimitError,APIError]]:

        entities = await self.extract_entities(passage)

        if not is_successful(entities):
            return Result.from_failure(entities.failure())

        triples = await self.extract_triples(passage, [entity for entity in entities.unwrap()])

        if not is_successful(triples):
            return Result.from_failure(triples.failure())
        
        entity2types = await self.classify_entities(entities.unwrap(), triples.unwrap(), type_schema)

        print(entity2types.unwrap())

        return triples
    
    async def batch_extract(self, texts : List[str], type_schema : List[Entity]) -> List[Result[List[Triple], Union[RateLimitError,APIError]]]:

        return await asyncio.gather(*[
            self.extract_from_passage(text, type_schema)
            for text in texts
        ])
    
    def extract(self, texts : List[str], type_schema : List[Entity]) -> List[Result[List[Triple], Union[RateLimitError,APIError]]]:
        return asyncio.run(self.batch_extract(texts, type_schema))