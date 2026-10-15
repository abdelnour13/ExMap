import re
import json
from json import JSONDecodeError
from pydantic import BaseModel, create_model, model_validator, ValidationError
from typing import Type, Dict, List, Any
from src.helpers import make_openai_request
from returns.pipeline import is_successful
from returns.result import Result
from openai import AsyncOpenAI, RateLimitError, APIError
from openai.types.chat import ChatCompletion
from openai import AsyncClient

__TYPES_MAPPING__ = {
    type_.__name__ : type_
    for type_ in [str, bool , int, float, list]
}

def resolve_type(attr_type : str) -> Type:

    if '[' not in attr_type or ']' not in attr_type:
        return __TYPES_MAPPING__[attr_type]

    inner_type = re.findall(r"\[\w+\]", attr_type)[0].strip('[]')
    main_type = re.sub(fr'\[{inner_type}\]', '', attr_type)

    inner_type = __TYPES_MAPPING__[inner_type]
    main_type = __TYPES_MAPPING__[main_type]

    return main_type[inner_type]
    

def parse_model_from_prompt(prompt : str) -> Type[BaseModel]:

    attributes_spec : List[str] = re.findall(r"\{\w+:[\w\[\]]+\}", prompt)

    attributes = {}

    for attr_spec in attributes_spec:
        attr_name, attr_type = attr_spec.strip('{}').split(':')
        attr_type = resolve_type(attr_type)
        attributes[attr_name] = attr_type

    return create_model("input", **attributes)

def processed_user_prompt(user_prompt) -> str:

    return re.sub(
        r"\{(\w+)\s*:\s*[^}]+\}",
        r"{\1}",
        user_prompt,
    )

_MISSING : Type[BaseModel] = create_model("missing")

class PromptContent(BaseModel):
    system_prompt : str 
    user_prompt : str
    output_model : Type[BaseModel]
    input_model : Type[BaseModel] = _MISSING

    @model_validator(mode="after")
    def validate_user_prompt(self):

        if self.input_model is _MISSING:
            self.input_model = parse_model_from_prompt(self.user_prompt)

        self.user_prompt = processed_user_prompt(self.user_prompt)

        return self

class PromptsManager:

    __PROMPTS__ : Dict[str, Type] = {}

    @classmethod
    def register_prompt(cls, prompt : Type) -> Type:
        prompt.prompts = cls._get_all_prompt(prompt)
        cls.__PROMPTS__[prompt.__name__] = prompt
        return prompt

    @classmethod
    def _get_all_prompt(cls, prompt : Type) -> Dict[str, PromptContent]:

        result = {}

        for attr, prompt_content in prompt.__dict__.items():

            if not isinstance(prompt_content, PromptContent):
                continue

            if not attr.endswith('_prompt'):
                continue

            lang = attr.removesuffix('_prompt')
            result[lang] = prompt_content

        return result

    @classmethod
    def get_all_prompts(cls, prompt_name : str) -> Dict[str, PromptContent]:

        if prompt_name not in cls.__PROMPTS__:
            raise Exception(f"Prompt '{prompt_name}' doesn't exist.")

        return cls.__PROMPTS__[prompt_name].prompts

    @classmethod
    def get_prompt(cls, prompt_name : str, lang : str) -> PromptContent:

        all_prompts = cls.get_all_prompts(prompt_name)

        if lang not in all_prompts:
            raise Exception(f"{lang} version of prompt {prompt_name} is not available.")

        return all_prompts[lang]

    @classmethod
    async def make_request(cls, 
        prompt_name : str,
        client : AsyncClient,
        model_name : str,
        input : Dict[str, Any], 
        lang : str = 'en',
        validate_input : bool = True,
        validate_output : bool = True,
    ) -> Result[Dict, APIError | RateLimitError | JSONDecodeError | ValidationError]:

        prompt = cls.get_prompt(prompt_name=prompt_name, lang=lang)

        if validate_input:
            prompt.input_model(**input)

        messages = [
            {
                "role" : "system",
                "content" : prompt.system_prompt
            },
            {
                "role" : "user",
                "content" : prompt.user_prompt.format(**input)
            }
        ]

        result = await make_openai_request(
            client=client,
            model=model_name,
            messages=messages,
            max_attempts=3
        )

        result = result.map(lambda x : x.choices[0].message.content)

        if is_successful(result):

            try:
                result = Result.from_value(json.loads(result))
            except Exception as e:
                result = Result.from_failure(e)
        

        if validate_output and is_successful(result):

            result = result.unwrap()

            try:
                prompt.output_model(**result)
                result = Result.from_value(result)
            except ValidationError as e:
                result = Result.from_failure(e)

        return result

@PromptsManager.register_prompt
class EquivelantPredicateTokensQuerying:

    en_prompt = PromptContent(

        system_prompt = """

            You are selecting which candidate **tokens** can replace a target token in a **knowledge-graph predicate** without significant information loss.

            Important: these tokens are **not complete predicates**. They are words that appear as parts of predicates. Evaluate each token according to the **semantic role it plays within a predicate**, not simply as an isolated dictionary word.

            **Only consider candidates that have the same grammatical role and part of speech as the target token; do not select morphological or derivational variants with a different grammatical role (e.g. `approximate` ≠ `approximation` ≠ `approximately`).**

            Be strict. Select a candidate only if replacing the target token with it would preserve essentially the same relation and meaning in a KG predicate.

           The goal of this step:
           - Ignore adjectives and adverbes that emphasizes the predicate , for example : "is critical challenge in" ~ "is challenge in" ~ "pose challenge for".
           - Map predicates that serve the same role : "is challenge in" ~ "poses problem in" | avoid ~ prevent mitigate.
           

            The target token itself should always be included.

            Return **only valid JSON**, with no Markdown or additional text, using exactly:

            {
            "selected": ["<target>", "<candidate1>", "<candidate2>"]
            }

        """,

        user_prompt = """
            ### Target :
            {target:str}

            ### Candidates :
            {candidates:list[str]}
        """,

        output_model=create_model('output', selected=list[str])

    )

@PromptsManager.register_prompt
class EquivelantPredicatesQuerying:

    en_prompt = PromptContent(

        system_prompt = """

            You are selecting which candidate **knowledge-graph predicates** can replace a target predicate without significant information loss.

            Unlike token-level comparison, the inputs here are **complete predicates**. Evaluate the **entire predicate**, including all words, grammatical structure, direction of the relation, and semantic meaning.

            Select a candidate only if replacing the target predicate with it would preserve essentially the same relation and meaning.

            The predicates must also express the relation in the **same direction**. For example, `important_for` and `importance_of` should not automatically be considered equivalent if their subject/object roles differ.

            If no predicate in the provided set matches the target predicate then an empty list must be returned.

            Along with the set of selected predicates a representative predicate should also be selected that better represent the entire set.

            Return **only valid JSON**, with no Markdown or additional text, using exactly:

            {
                "selected": ["<target>", "<candidate1>", "<candidate2>"],
                "representative" : "<representative_predicate>"
            }

        """,

        user_prompt = """
            ### Target :
            {target:str}

            ### Candidates :
            {candidates:list[str]}
        """,

        output_model=create_model('output', selected=list[str], representative=str)

    )

@PromptsManager.register_prompt
class RepresentativePredicateQuerying:

    en_prompt = PromptContent(

        system_prompt = """

            You are given a set of knowledge-graph predicates that are considered semantically equivalent or closely interchangeable.

            Your task is to propose **one representative predicate** that can replace all predicates in the set while preserving their shared meaning and relation.

            The representative predicate must:
            - Express the **common semantic relation** shared by the predicates.
            - Be semantically equivalent to the predicates as a group, not merely related to them.
            - Be a **natural, clear, and commonly understandable KG predicate**.
            - Preserve the **direction of the relation** (subject → object).
            - Avoid introducing a stronger, weaker, or more specific meaning than the shared relation.
            - Prefer a simple and general formulation over an unnecessarily specific one.
            - Not simply copy one predicate if another formulation better represents the common meaning.

            If the predicates contain minor grammatical or lexical variations, normalize them into a single canonical predicate.

            Return **only valid JSON** with exactly this structure:

            {
                "representative": "<predicate>",
            }

        """,

        user_prompt = """
            ### Predicates
            {predicates:list[str]}
        """,

        output_model=create_model('output', representative=str)

    )

@PromptsManager.register_prompt
class EquivalentEntitiesQuerying:

    en_prompt = PromptContent(

        system_prompt = """
            You are selecting which candidate knowledge-graph entities are
            semantically equivalent to a target entity and can replace it
            without significant information loss.

            Evaluate the complete meaning and identity of each entity,
            considering its name, abbreviations, synonyms, alternative
            spellings, lexical variations, and any other available
            information in the entity labels.

            Select a candidate only if it refers to the same underlying
            entity or concept as the target, such that replacing the
            target with the candidate would preserve its identity and
            meaning.

            Distinguish strict semantic equivalence from semantic
            relatedness. Entities must not be considered equivalent merely
            because they:
            - Belong to the same category or domain.
            - Have similar names or share common words.
            - Are associated with, used by, or dependent on one another.
            - Have a broader, narrower, or otherwise different meaning.
            - Represent different instances of the same type of entity.

            For example, synonyms referring to the same concept may be
            equivalent, but a method and its application, a model and its
            architecture, or a tool and its developer are not equivalent
            merely because they are closely related.

            Preserve distinctions in specificity and meaning. Do not merge
            a specific entity into a broader category or a broader concept
            into one of its specific instances.

            The selected set must include the target entity itself.
            If no candidate other than the target is semantically
            equivalent, return a list containing only the target.

            Along with the selected entities, choose one representative
            entity that best represents the identity and shared meaning
            of the selected set.

            The representative must be semantically equivalent to every
            selected entity. Prefer a clear, canonical, unambiguous name.
            Do not introduce a new entity or a broader or narrower concept.

            Return only valid JSON, with no Markdown or additional text,
            using exactly this structure:

            {
                "selected": ["<target>", "<candidate1>", "<candidate2>"],
                "representative": "<representative_entity>"
            }
        """,

        user_prompt = """
            ### Target:
            {target:str}

            ### Candidates:
            {candidates:list[str]}
        """,

        output_model=create_model(
            'output',
            selected=list[str],
            representative=str
        )
    )

@PromptsManager.register_prompt
class RepresentativeEntityQuerying:

    en_prompt = PromptContent(

        system_prompt = """
            You are given a set of knowledge-graph entities that are
            considered semantically equivalent and refer to the same
            underlying entity or concept.

            Your task is to propose one representative entity name that
            can replace all entities in the set while preserving their
            shared identity and meaning.

            The representative entity must:
            - Refer to the same underlying entity or concept as every
              entity in the set.
            - Preserve the semantic identity and specificity shared by
              the entities.
            - Be a natural, clear, canonical, and commonly understandable
              entity name.
            - Prefer a standard name over abbreviations, spelling
              variations, informal names, or redundant lexical variants.
            - Expand an abbreviation only when its intended full form
              is unambiguous.
            - Normalize capitalization, spelling, and superficial
              formatting differences when appropriate.
            - Avoid introducing a broader category, narrower concept,
              related entity, or new meaning.
            - Avoid combining distinct entities into a single name
              merely because they belong to the same domain.
            - Prefer an established canonical name over an unnecessarily
              descriptive or artificially constructed name.

            Do not simply select the longest name or the most general
            term. Choose the name that most accurately identifies the
            common entity or concept.

            If the input contains lexical or grammatical variations
            referring to the same concept, normalize them into one
            canonical entity name.

            If multiple entities appear to have different meanings or
            identities despite being provided as a set, do not invent
            a broader concept to unify them. Prefer a representative
            that preserves the identity shared by the set, and use an
            existing entity name when no safe canonical normalization
            is possible.

            Return only valid JSON, with no Markdown or additional text,
            using exactly this structure:

            {
                "representative": "<entity>"
            }
        """,

        user_prompt = """
            ### Entities:
            {entities:list[str]}
        """,

        output_model=create_model(
            'output',
            representative=str
        )
    )