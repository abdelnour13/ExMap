import re
import string
import spacy
from textblob import Word
from nltk import pos_tag, word_tokenize
from typing import List, Tuple, Dict
from src.helpers import load_json_file
from src.shared.command_tools import StageConfig, Stage, add_handlers
from src.shared.file_tools import FileWriter, FileReader, TextfileCache
from pycli import Field
from pydantic import computed_field, BaseModel
from instance import cli
from definitions import nlp_dir

class Stop(BaseModel):
    entities : List[str]
    predicates : List[str]
    standard_verbs : List[str]
    tokens : List[str]
    tail : List[str]

class Nlp(BaseModel):
    stop : Stop
    propositions : List[str]
    mapping : Dict[str, Tuple[str, bool]]

class Config(StageConfig):

    domain : str = Field(
        default="computer_science", 
        description="The domain that describes nature of the topics discussed articles used to create the graph."
    )

    @computed_field
    @property
    def nlp(self) -> Dict[str, str]:
        return Nlp(**load_json_file(nlp_dir(self.domain)))

@cli.register_stage(
    name="entities_normalization",
    description=(
        "Apply a set of rules to clean the triples and retain the main verb in the predicate."
    ),
    add_from_conf=False,
    add_save_to=False
)
@add_handlers()
class EntitiesNormalizer(Stage):

    MAX_ENTITY_LEN = 6

    def __init__(
        self,
        reader: FileReader,
        writer: FileWriter,
        domain_nlp : Nlp
    ) -> None:
        super().__init__(reader, writer)
        self.domain_nlp = domain_nlp
        self.nlp = spacy.load("en_core_web_sm")

    def normalize_entity(self, entity: str) -> str | None:

        # Remove punctuations
        translator = str.maketrans("", "", string.punctuation)

        # Remove hyphens and back-slash
        entity = entity.replace("-", " ")
        entity = entity.replace("/", " ")

        # Remove numbers 
        entity = re.sub("[0-9]+", "", entity)
        entity = re.sub(r"\s+", " ", entity)
        entity = entity.translate(translator)

        # Lowercasing & stripping leading and trailing whitespaces
        entity = entity.lower()

        # Remove articles from the begining of the entity name
        parts = word_tokenize(entity)

        if len(parts) == 0:
            return None

        if parts[0] in ["a", "as", "the", "an"]:

            parts = parts[1:]
            entity = " ".join(parts)

        if len(parts) == 0:
            return None

        # Singularize the last plural token
        # For example after this : "graph based models" becomes "graph based model"
        tokens_tags = pos_tag(parts)

        if tokens_tags[-1][1] == 'NNS':
            entity = ' '.join([token for token, tag in tokens_tags[:-1]]) + ' ' + Word(tokens_tags[-1][0]).singularize()

        entity = entity.strip()

        return entity
    
    def transform(self, doc: Dict) -> Dict:

        doc['head'] = self.normalize_entity(doc['head'])
        doc['tail'] = self.normalize_entity(doc['tail'])

        normalized_acronyms = []

        for acronym in doc['sentence']['acronyms']:

            acronym['expansion'] = self.normalize_entity(acronym['expansion'])

            if acronym['expansion'] is not None:
                normalized_acronyms.append(acronym)

        doc['sentence']['acronyms'] = normalized_acronyms

        return doc
    
    def skip(self, doc):
        return doc['head'] is None or doc['tail'] is None
    
    @classmethod
    def from_config(cls, config: Config) -> Stage:

        cache = TextfileCache(config.checkpoint_filename)

        return EntitiesNormalizer(
            reader=FileReader
                .from_filename(config.input_filename, cache)
                .batched(config.batch_size),
            writer=FileWriter
                .from_filename(config.output_filename)
                .buffered(config.buffer_size, cache),
            domain_nlp=config.nlp
        )