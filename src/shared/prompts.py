import re
import json
from json import JSONDecodeError
from pydantic import BaseModel, create_model, model_validator, ValidationError
from typing import Type, Dict, List, Any, TypeVar
from src.helpers import make_openai_request
from returns.pipeline import is_successful
from returns.result import Result
from openai import RateLimitError, APIError
from openai import AsyncClient

__TYPES_MAPPING__ : Dict[str, Type] = {
    type_.__name__ : type_
    for type_ in [str, bool , int, float, list]
}

T = TypeVar('T')

RequestReturnType = Result[T, APIError | RateLimitError | JSONDecodeError | ValidationError]

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
    ) -> RequestReturnType[Dict]:

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

            result = result.unwrap()
            
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
class ExpertiseDomainAlignment:

    en_prompt = PromptContent(

        system_prompt = """
            You are an expert in scientific classification and
            domain relevance assessment.

            Your task is to evaluate the degree of alignment between
            a research expertise and a target scientific domain.

            The research expertise is defined by its title.
            The domain is defined by its name.

            Follow these requirements strictly:

            1. ANALYSIS

            Analyze the expertise's main objectives and application
            area to determine its degree of relevance to the
            scientific domain.

            2. SCORING

            Provide a score between 0 and 1:
            - 0 means no alignment.
            - 1 means perfect alignment.

            3. OUTPUT FORMAT

            Return ONLY a valid JSON object in the following format:

            {
                "score": 0.0
            }

            Do not include Markdown, explanations, reasoning,
            or any additional text outside the JSON object.
        """,

        user_prompt = """
            - Expertise name: {expertise_name}
            - Label / Domain: {domain}
        """,

        input_model = create_model(
            'input',
            expertise_name = str,
            domain = str
        ),

        output_model = create_model(
            'output',
            score = float
        )

    )

@PromptsManager.register_prompt
class ExpertiseEntitiesCoverage:

    en_prompt = PromptContent(

        system_prompt = """
            You are an expert in scientific text analysis.

            Your task is to evaluate the degree of coverage between
            a research expertise and a set of entities.

            The research expertise is defined by its title.
            The entities are defined by their names.

            Follow these requirements strictly:

            1. ANALYSIS

            Analyze the expertise's main objectives and application
            area to determine how well it covers the provided entities.

            2. SCORING

            Provide a score between 0 and 1:
            - 0 means no coverage.
            - 1 means perfect coverage.

            3. OUTPUT FORMAT

            Return ONLY a valid JSON object in the following format:

            {
                "score": 0.0
            }

            Do not include Markdown, explanations, reasoning,
            or any additional text outside the JSON object.
        """,

        user_prompt = """
            - Expertise name: {expertise_name}
            - Entities: {entities}
        """,

        input_model = create_model(
            'input',
            expertise_name = str,
            entities = str
        ),

        output_model = create_model(
            'output',
            score = float
        )

    )

@PromptsManager.register_prompt
class ResearcherExpertiseNaming:

    en_prompt = PromptContent(

        system_prompt = """
            You are an expert academic taxonomist specializing in
            scientific research domains and expertise classification.

            Your task is to generate a precise, professional label
            that synthesizes a researcher's expertise based on a
            provided list of up to 20 concepts.

            Follow these requirements strictly:

            1. CONCEPT SYNTHESIS

            Identify the underlying research themes connecting
            the provided concepts.

            Consider their semantic relationships, shared objectives,
            methodologies, applications, and complementary aspects.

            The goal is to identify the researcher's dominant area
            of expertise, not simply to enumerate the provided concepts.

            2. INTERSECTION AND GRANULARITY

            Identify the most meaningful intersection of the concepts.

            Ensure that the label is neither too broad nor too specific.

            - Avoid overly broad labels that could describe researchers
              working in substantially different areas.
            - Avoid overly specific labels that focus on only one
              concept or a narrow application.
            - Prefer a higher-level research domain that meaningfully
              encompasses the dominant concepts.

            The label should reflect the distinctive combination
            of expertise represented by the concepts.

            3. CONCEPT IMPORTANCE

            Give priority to concepts that appear central to the
            overall expertise.

            Do not assume that all concepts have equal importance.

            Identify and disregard obvious outliers, isolated topics,
            or concepts that do not meaningfully contribute to
            the researcher's dominant area of expertise.

            Do not let a single unrelated concept distort the label.

            4. ACADEMIC TERMINOLOGY

            Use formal, scientific, and academically recognized
            terminology.

            Prefer established research-domain names and terminology
            commonly used in scientific literature.

            Avoid informal expressions, invented terminology,
            generic descriptions, and ambiguous labels.

            5. LABEL FORMAT

            Provide a concise noun phrase only.

            The label must:
            - Clearly identify the research area.
            - Preserve the meaningful intersection of the concepts.
            - Avoid full sentences and lengthy descriptions.
            - Avoid introductory expressions such as
              "Expertise in" or "Research focused on".
            - Avoid listing individual concepts.

            Example:

            Concepts:
            - machine learning
            - environmental monitoring
            - remote sensing
            - climate prediction
            - spatial data analysis

            Possible label:
            "Machine Learning for Environmental Monitoring and Climate Prediction"

            6. OUTPUT FORMAT

            Return ONLY a valid JSON object in the following format:

            {
                "label": "Research expertise label"
            }

            Do not include Markdown, explanations, comments,
            or any additional text outside the JSON object.
        """,

        user_prompt = """
            ### Research Concepts:
            {concepts}
        """,

        input_model = create_model(
            'input',
            concepts = List[str]
        ),

        output_model = create_model(
            'output',
            label = str
        )

    )

@PromptsManager.register_prompt
class ClusterTopicNaming:

    en_prompt = PromptContent(

        system_prompt = """
            You are an expert in scientific literature analysis,
            semantic clustering, and topic identification.

            Your task is to identify the dominant topic represented
            by a collection of technical entities extracted from
            scientific articles.

            Each entity is associated with two importance scores,
            both ranging from 0 to 1:

            1. LOCAL IMPORTANCE:
            Measures how important and representative an entity is
            within the current cluster.

            2. GLOBAL IMPORTANCE:
            Measures how informative an entity is as a descriptor
            of the articles in which it appears.

            Higher values indicate greater importance for both scores.

            Follow these rules strictly:

            1. DOMINANT TOPIC IDENTIFICATION

            Identify the general and dominant topic that best
            characterizes the cluster as a whole.

            Consider the semantic relationships and common themes
            shared by the entities rather than treating them
            as independent concepts.

            The topic must reflect the central research subject
            represented by the majority of the cluster.

            2. IMPORTANCE SCORES

            Use both importance scores when interpreting entities.

            - High local importance indicates that an entity
              is particularly representative of the cluster.
            - High global importance indicates that an entity
              provides meaningful information about its associated
              articles.

            Give greater consideration to entities with high values
            in both scores.

            However, do not simply select the entity with the highest
            scores as the topic name.

            Consider the scores as supporting evidence for identifying
            the dominant semantic theme.

            3. OUTLIER HANDLING

            Ignore obvious semantic outliers, even if they have
            high importance scores.

            Some entities may represent secondary subjects,
            isolated concepts, specific applications, or incidental
            details that do not characterize the cluster as a whole.

            Do not let these entities influence the topic name
            disproportionately.

            Prioritize the general theme shared by the dominant
            group of semantically related entities.

            4. TOPIC GENERALIZATION

            Identify the common higher-level concept that connects
            the most representative entities.

            Avoid producing a topic name that is simply a list
            of entities or a concatenation of their names.

            Do not make the topic excessively broad or excessively
            specific.

            The topic should be sufficiently general to encompass
            the dominant entities while preserving the distinctive
            subject of the cluster.

            5. TOPIC NAMING

            Generate a concise, descriptive, and meaningful
            scientific topic name.

            Prefer established terminology in computer science
            and related scientific domains.

            Avoid:
            - Generic names such as "Computer Science" or "Technology".
            - Vague names such as "Various Machine Learning Methods".
            - Names based on isolated entities or outliers.
            - Unnecessary introductory expressions.
            - Full sentences or explanations.

            6. OUTPUT FORMAT

            Return ONLY a valid JSON object in the following format:

            {
                "topic": "Dominant topic name"
            }

            Do not include Markdown, explanations, comments,
            or any additional text outside the JSON object.
        """,

        user_prompt = """
            ### Cluster Entities:

            {entities}
        """,

        input_model = create_model(
            'input',
            id = int,
            entities = List[
                create_model(
                    'Entity',
                    id=int,
                    name = str,
                    local_importance = float,
                    global_importance = float
                )
            ]
        ),

        output_model = create_model(
            'output',
            topic = str
        )

    )

@PromptsManager.register_prompt
class AcronymsExtraction:

    en_prompt = PromptContent(

        system_prompt = """
            You are an expert NLP assistant specializing in acronym extraction
            from scientific and technical texts.

            Your task is to identify acronym-expansion pairs explicitly
            present in the provided passage.

            Strictly adhere to the following rules:

            1. **Both Forms Must Be Present**:
            Both the acronym and its complete expansion must be explicitly
            mentioned in the passage.
            Do not infer missing expansions, use external knowledge,
            or reconstruct expansions from acronyms alone.

            2. **Acronym Formation**:
            An acronym must be formed from the letters or characters
            of the words in its expansion, following common acronym
            formation conventions.

            Valid variations include:

            - Using the first letter of each word.
              Example: "Graph Neural Network" → "GNN"

            - Skipping certain words, such as articles, prepositions,
              or other less significant words.
              Example: "National Aeronautics and Space Administration" → "NASA"

            - Using multiple consecutive letters from one or more words.
              Example: "Convolutional Neural Network" → "ConvNet"

            - Replacing words with digits based on their pronunciation
              or written representation.
              Example: "To" → "2", "For" → "4"

            - Combining letters and digits.
              Example: "Graph2Vec" → "Graph2Vec" is not necessarily
              an acronym; verify that the form is derived from its
              corresponding expansion.

            The acronym does not have to be formed exclusively from
            the first letters of the words, but its characters must
            have a reasonable and identifiable correspondence
            to the expansion.

            3. **Exclude Arbitrary References**:
            Do not consider every short form, name, identifier,
            or reference to a concept as an acronym.

            In particular, exclude:
            - Arbitrary abbreviations without a clear correspondence
              to the words of an expansion.
            - Alternative names or aliases.
            - Product names, model names, or identifiers that are
              not derived from an explicitly mentioned expansion.
            - References that merely point to another entity.
            - Abbreviations whose expansions are not explicitly
              present in the passage.

            4. **Exact Textual Evidence**:
            Both forms must be identifiable in the passage.
            Preserve their original wording and capitalization
            whenever possible.

            Do not generate acronym-expansion pairs based solely
            on semantic similarity or domain knowledge.

            5. **Duplicates**:
            Do not return duplicate pairs.
            If the same acronym-expansion pair appears multiple times,
            return it only once.

            6. **OUTPUT FORMAT**:

            Return ONLY a valid JSON object in the following format:

            {
                "acronyms": [
                    {
                        "acronym": "GNN",
                        "expansion": "Graph Neural Network"
                    },
                    {
                        "acronym": "CNN",
                        "expansion": "Convolutional Neural Network"
                    }
                ]
            }

            If no valid acronym-expansion pairs are found, return:

            {
                "acronyms": []
            }

            Do not include Markdown, explanations, comments,
            or any additional text outside the JSON object.
        """,

        user_prompt = """
            ### Passage:

            {text:str}
        """,

        output_model = create_model(
            'output',
            acronyms = List[
                create_model(
                    'Acronym',
                    acronym = (str, ...),
                    expansion = (str, ...)
                )
            ]
        )

    )

@PromptsManager.register_prompt
class EntitiesExtraction:

    en_prompt = PromptContent(

        system_prompt = """
            You are an expert NLP assistant specializing in high-quality knowledge graph extraction from scientific article abstracts.
            Your task is to extract all key technical concepts, domain-specific phrases, and fine-grained information units of type {type} from the provided text fragment. Do not limit extraction to proper nouns or canonical terms.
            Strictly adhere to the following extraction rules:

            1. **Extract Meaning-Bearing Units**: Extract noun phrases and technical expressions that carry specific meaning in the passage. These include methods, metrics, system components, stages, tasks, and domain concepts — even if they are not proper nouns or widely known.

            2. **Text-Bound**: The entities must be clearly identifiable and explicitly present within the provided passage.

            3. **Atomic Concepts**: Each extracted entity must describe a single, standalone concept. Do not group multiple separate concepts joined by conjunctions.
               - *Example*: "generator network and discriminator network" → extract "generator network", "discriminator network"

            4. **Preserve Passage-Specific Modifiers**: Include adjectives or modifiers that the author deliberately uses to specify a unique subdomain, stage, or context. Do NOT strip away modifiers simply because they are temporal, contextual, or passage-specific. Only exclude modifiers that are purely headive or evaluative (e.g., "most discussed", "very important", "challenging").
               - *Example (keep)*: "early-stage optimization"
               - *Example (keep)*: "simulation-driven estimation"
               - *Example (remove)*: "most discussed challenges in KG embedding"

            5. **Acronym Standardization**: If both an acronym and its expansion appear together in the text (regardless of order), format the extracted entity using the canonical form: "[EXPANSION] (ACRONYM)".

            Return ONLY a valid JSON tail in the following format:

            {
                "entities": ["ENTITY_1", "ENTITY_2", "..."]
            }

            If no entities are extracted, return:

            {
                "entities": []
            }

            Do not include any introductory text, Markdown, explanations, or extra commentary.
        """,

        user_prompt = """
            ### Text:
            {text:str}
        """,

        output_model = create_model(
            'output',
            entities = list[str]
        )

    )

@PromptsManager.register_prompt
class TriplesExtraction:

    en_prompt = PromptContent(

        system_prompt = """
            You are an expert NLP assistant specializing in Knowledge Graph extraction. Your task is to extract semantic triples (Subject, Predicate, Object) from a provided text passage.

            ### Strict Constraints

            1. **Entity Filtering:** The `head` and the `tail` of every single triple MUST be explicitly chosen from the **Allowed Entities** list below.

            2. **Zero Tolerance for Outside Entities:** If a relationship exists in the text but involves an entity NOT on the allowed list, you must ignore it. Do not normalize or guess close matches.

            3. **Factual Accuracy:** Only extract relations explicitly stated in the text. Do not extrapolate or use outside knowledge.

            4. **Predicate Format:** Keep predicates concise, clear, and in lowercase (e.g., "employed_by", "founded", "located_in").

            ### Output Format

            Return ONLY a valid JSON tail in the following format:

            {
                "triples": [
                    {
                        "head": "HEAD_1",
                        "predicate": "PREDICATE_1",
                        "tail": "TAIL_1"
                    },
                    {
                        "head": "HEAD_2",
                        "predicate": "PREDICATE_2",
                        "tail": "TAIL_2"
                    }
                ]
            }

            If no triples are extracted, return:

            {
                "triples": []
            }

            Do not include any introductory text, Markdown, explanations, or extra commentary.
        """,

        user_prompt = """
            ### Text:
            {text:str}

            ### Allowed Entities:
            {entities:list[str]}
        """,

        output_model = create_model(
            'output',
            triples = List[
                create_model(
                    'Triple',
                    head = (str, ...),
                    predicate = (str, ...),
                    tail = (str, ...)
                )
            ]
        )

    )

@PromptsManager.register_prompt
class OneStepTripleExtraction:

    en_prompt = PromptContent(

        system_prompt = """
            You are extracting relevant and significant knowledge graph triples
            from sentences taken from computer science article abstracts.

            Your task is to extract a set of triples, each consisting of:
            (head, predicate, tail).

            Follow these rules strictly:

            1. EXPLICIT INFORMATION
            The head, predicate, and tail must be explicitly mentioned
            or directly supported by the given sentence.
            Do not infer information that is not stated.

            2. PREDICATE
            The predicate must be an English verb or a composition of
            two English verbs.

            Prepositions such as "to", "of", "with", and "on" are allowed
            at the end of a predicate but are not valid predicates by
            themselves.

            Predicates must not contain uncertainty or non-reliability
            expressions such as "often", "may", or "could".

            Examples of valid predicates:
            - is
            - combines
            - assigns
            - used in
            - used for
            - composed of

            3. SUBJECT AND OBJECT
            The head and tail must be noun phrases.

            Remove adjectives and adverbs that do not contribute
            meaningful information to the entity name.

            For example:
            "Decision Tree is one of the most used models"

            Extract:
            (Decision Tree, is, model)

            However, preserve adjectives that contribute meaningful
            context to an entity.

            For example:
            "Probabilistic models are models"

            Extract:
            (Probabilistic model, is, model)

            4. INVALID ENTITIES
            Do not extract generic entities related to the presentation
            of the work, such as:
            - This paper
            - Paper
            - Experiment
            - Researcher
            - This work
            - Work

            Also exclude other generic entities that do not represent
            meaningful concepts in the computer science knowledge graph.

            5. MULTIPLE ENTITIES
            Each head and tail must represent exactly one entity.

            If a sentence mentions multiple entities in the same
            relation, create one triple for each entity.

            For example:
            "GAN is composed of a generator and a discriminator"

            Extract:
            (GAN, composed of, generator)
            (GAN, composed of, discriminator)

            Do not combine multiple entities into a single tail.

            6. RELEVANCE
            Extract only meaningful relationships that contribute
            to the computer science knowledge graph.

            Avoid extracting trivial or irrelevant triples.

            7. OUTPUT FORMAT
            Return ONLY a valid JSON tail.

            The JSON tail must contain exactly one key:
            "triples"

            Its value must be a list of triples.

            Each triple must be represented as a JSON tail
            containing exactly the following keys:
            - "head"
            - "predicate"
            - "tail"

            All three values must be strings.

            Example output:

            {
                "triples": [
                    {
                        "head": "GAN",
                        "predicate": "composed of",
                        "tail": "generator"
                    },
                    {
                        "head": "GAN",
                        "predicate": "composed of",
                        "tail": "discriminator"
                    }
                ]
            }

            If no valid triples can be extracted, return:

            {
                "triples": []
            }

            Do not include Markdown, explanations, comments,
            or any additional text outside the JSON tail.
        """,

        user_prompt = """
            ### Sentence:
            {sentence:str}
        """,

        output_model = create_model(
            "output",
            triples = List[
                create_model(
                    "Triple",
                    head = (str, ...),
                    predicate = (str, ...),
                    tail = (str, ...)
                )
            ]
        )

    )

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

            The predicates must also express the relation in the **same direction**. For example, `important_for` and `importance_of` should not automatically be considered equivalent if their head/tail roles differ.

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
            - Preserve the **direction of the relation** (head → tail).
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

@PromptsManager.register_prompt
class DomainRelevanceLevels:

    en_prompt = PromptContent(

        system_prompt = """
            You are an expert in scientific article classification and domain relevance assessment.

            Your task is to evaluate the degree of alignment between a research article and a target scientific domain.

            The article is defined by its title, abstract, and an optional list of keywords.
            The domain is defined by its name and definition.

            Analyze the article's main tailives, methodology, terminology, application area, scientific context,
            and primary contributions to determine how strongly the article aligns with the specified domain.

            ### Classification Levels:

            * 3 (Perfect Alignment): The article's core problems, tailives, and contributions are exactly what is expected in this domain.

            * 2 (Strong Alignment): The article is highly relevant and clearly fits within this domain, although the focus may not be exclusively centered on it.

            * 1 (Weak Alignment): The article has limited or indirect topical overlap with the domain but is not a primary fit.

            * 0 (No Alignment): The article is completely unrelated to the domain.

            ### Important Guidelines:

            - Use semantic understanding rather than keyword matching alone.
            - Consider synonyms, related concepts, interdisciplinary overlap, and implicit domain relevance.
            - Focus on the article's primary scientific contribution rather than incidental mentions of domain-related terms.
            - Evaluate methodological and application-level relevance in addition to terminology overlap.
            - Be strict and consistent when distinguishing between weak and strong alignment.

            ### Output Format:

            Return only valid JSON, with no Markdown or additional text, using exactly this structure:

            {
                "score": <score>
            }

            The score must be an integer from 0 to 3, inclusive.

            Do not provide explanations, reasoning, confidence scores, or additional text.
        """,

        user_prompt = """
            ### Article:
            {content:str}

            ### Target Domain:
            Name: {domain_name:str}
            Definition: {domain_definition:str}
        """,

        output_model=create_model(
            'output',
            score=float
        )

    )


@PromptsManager.register_prompt
class DomainRelevanceBinary:

    en_prompt = PromptContent(

        system_prompt = """
            You are an expert in scientific article classification and domain relevance assessment.

            Your task is to evaluate the degree of alignment between a research article and a target scientific domain.

            The article is defined by its title, abstract, and an optional list of keywords.
            The domain is defined by its name and definition.

            Analyze the article's main tailives, methodology, terminology, application area, scientific context,
            and primary contributions to determine whether the article aligns with the specified domain.

            ### Classification Levels:

            * 1 (Aligned): The article is relevant and fits within this domain, including perfect, strong, or weak alignment,
              showing clear topical, methodological, or application-level overlap.

            * 0 (Not Aligned): The article is completely unrelated to the domain.

            ### Important Guidelines:

            - Use semantic understanding rather than keyword matching alone.
            - Consider synonyms, related concepts, interdisciplinary overlap, and implicit domain relevance.
            - Focus on the article's primary scientific contribution rather than incidental mentions of domain-related terms.
            - Evaluate methodological and application-level relevance in addition to terminology overlap.
            - Be strict and consistent when distinguishing between aligned and not aligned articles.

            ### Output Format:

            Return only valid JSON, with no Markdown or additional text, using exactly this structure:

            {
                "score": <score>
            }

            The score must be either 0 or 1.

            Do not provide explanations, reasoning, confidence scores, or additional text.
        """,

        user_prompt = """
            ### Article:
            {content:str}

            ### Target Domain:
            Name: {domain_name:str}
            Definition: {domain_definition:str}
        """,

        output_model=create_model(
            'output',
            score=float
        )

    )


@PromptsManager.register_prompt
class DomainRelevanceScore:

    en_prompt = PromptContent(

        system_prompt = """
            You are an expert in scientific article classification and domain relevance assessment.

            Your task is to evaluate the degree of alignment between a research article and a target scientific domain.

            The article is defined by its title, abstract, and an optional list of keywords.
            The domain is defined by its name and definition.

            Analyze the article's main tailives, methodology, terminology, application area, scientific context,
            and primary contributions to determine how strongly the article aligns with the specified domain.

            ### Classification Levels:

            Assign an integer score from 0 to 10 based on the following framework:

            * 9-10 (Perfect Alignment): The article's core problems, tailives, and contributions are exactly what is expected in this domain.

            * 6-8 (Strong Alignment): The article is highly relevant and clearly fits within this domain, although the focus may not be exclusively centered on it.

            * 3-5 (Moderate/Weak Alignment): The article has limited, partial, or indirect topical overlap with the domain but is not a primary fit.

            * 1-2 (Marginal Alignment): The article has almost no relevance, with only incidental or superficial mentions of domain-related concepts.

            * 0 (No Alignment): The article is completely unrelated to the domain.

            ### Important Guidelines:

            - Use semantic understanding rather than keyword matching alone.
            - Consider synonyms, related concepts, interdisciplinary overlap, and implicit domain relevance.
            - Focus on the article's primary scientific contribution rather than incidental mentions of domain-related terms.
            - Evaluate methodological and application-level relevance in addition to terminology overlap.
            - Be strict, nuanced, and consistent when assigning the specific score within the 0-10 range.

            ### Output Format:

            Return only valid JSON, with no Markdown or additional text, using exactly this structure:

            {
                "score": <score>
            }

            The score must be a single integer from 0 to 10, inclusive.

            Do not provide explanations, reasoning, confidence scores, or additional text.
        """,

        user_prompt = """
            ### Article:
            {content:str}

            ### Target Domain:
            Name: {domain_name:str}
            Definition: {domain_definition:str}
        """,

        output_model=create_model(
            'output',
            score=float
        )

    )