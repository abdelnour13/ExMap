import re
import string
import spacy
from nltk import pos_tag, word_tokenize
from typing import List, Tuple, Dict
from dataclasses import dataclass
from src.helpers import load_json_file
from src.shared.command_tools import StageConfig, Stage, add_handlers
from src.shared.file_tools import FileWriter, FileReader, TextfileCache
from src.shared.language_detectors import LanguageDetector
from dataclasses import dataclass
from pycli import Field
from pydantic import computed_field, BaseModel
from instance import cli
from definitions import nlp_dir

class Stop(BaseModel):
    entities : List[str]
    predicates : List[str]
    standard_verbs : List[str]
    tokens : List[str]

class Nlp(BaseModel):
    stop : Stop
    propositions : List[str]
    mapping : Dict[str, Tuple[str, bool]]

class Config(StageConfig):

    domain : str = Field(default="computer_science", description="The domain that describes nature of the topics discussed articles used to create the graph.")

    @computed_field
    @property
    def nlp(self) -> Dict[str, str]:
        return Nlp(**load_json_file(nlp_dir(self.domain)))

@dataclass(frozen=True)
class Token:
    value: str
    tag: str

    @property
    def simple_tag(self) -> str:
        return self.tag[:2]

@dataclass(frozen=True)
class Predicate:
    tokens: List[Token]

    @property
    def tags(self) -> List[str]:
        return [token.tag for token in self.tokens]

    @property
    def values(self) -> List[str]:
        return [token.value for token in self.tokens]

    @property
    def predicate(self) -> str:
        return " ".join(self.values)

    @property
    def pattern(self) -> str:
        return " ".join(self.tags)

    @property
    def simple_pattern(self) -> str:
        return " ".join([token.simple_tag for token in self.tokens])
    
    def matches(self, expression : str) -> bool:

        i = 0
        parts = expression.split(' ')

        for part in parts:

            if i >= len(self.tokens):
                return False

            ### This means simple tag
            if part.startswith("@"):
                
                if self.tokens[i].simple_tag not in part.strip('@').split('|'):
                    return False
                
            ### This means tag
            if part.startswith("#"):
                
                if self.tokens[i].tag not in part.strip('#').split('|'):
                    return False
                
            ### *
            elif part.startswith("*"):

                if len(part) > 2:
                    i += int(part.strip('*')) - 1                    

            ### Word
            else:

                if self.tokens[i].value not in part.split('|'):
                    return False

            i += 1

        if i > len(self.tokens):
            return False

        return True

@cli.register_stage(
    name="clean_triples",
    description=(
        "Apply a set of rules to clean the triples and retain the main verb in the predicate."
    ),
    add_from_conf=False,
    add_save_to=False
)
@add_handlers
class TriplesPostprocessor(Stage):

    MAX_PRED_LEN = 4
    MAX_ENTITY_LEN = 6

    def __init__(
        self,
        reader: FileReader,
        writer: FileWriter,
        batch_size: int,
        language_detector: LanguageDetector,
        domain_nlp : Nlp
    ) -> None:
        super().__init__(reader, writer, batch_size)
        self.domain_nlp = domain_nlp
        self.nlp = spacy.load("en_core_web_sm")
        self.language_detector = language_detector

    def base_normalize(self, entity: str) -> str:

        translator = str.maketrans("", "", string.punctuation)

        entity = entity.replace("-", " ")
        entity = entity.replace("/", " ")

        entity = re.sub("[0-9]+", "", entity)
        entity = re.sub(r"\s+", " ", entity)
        entity = entity.translate(translator)

        entity = entity.lower()
        entity = entity.strip()

        parts = entity.split(" ")

        if parts[0] in ["a", "as", "the", "an"]:
            entity = " ".join(parts[1:])

        return entity

    def lemmetize(self, predicate: str) -> str:
        return " ".join([token.lemma_ for token in self.nlp(predicate)])

    def make_predicate(self, predicate: str) -> Predicate:

        tokens = pos_tag(word_tokenize(predicate))
        tokens = [Token(token, tag) for token, tag in tokens]

        return Predicate(tokens)

    def clean_predicate(self, predicate: Predicate) -> Tuple[Predicate, bool, str]:

        is_passive = False
        E2 = None

        ### Remove stop words
        predicate = Predicate(
            [
                token
                for token in predicate.tokens
                if token.tag != "DT"
                and token.value not in self.domain_nlp.stop.tokens
            ]
        )

        if len(predicate.tags) == 0:
            return predicate, is_passive, E2

        ### Passive Form
        if predicate.values[-1] in ["by", "using"]:
            is_passive = True
            predicate = Predicate(tokens=predicate.tokens[:-1])

        ### Retain core action
        if len(predicate.tokens) >= 2:

            if predicate.values[0] in ["is", "was", "are", "be", "were", "being"] and (
                predicate.tags[1][:2] == "VB" 
                or predicate.values[1] in ["more", "less"]
            ):
                predicate = Predicate(tokens=predicate.tokens[1:])

            if len(predicate.tokens) >= 3:

                if (
                    predicate.tags[0] == "MD"
                    and predicate.tags[1] in ["is", "be", "was", "are", "were", "being"]
                    and predicate.tags[2][:2] == "VB"
                ):
                    predicate = Predicate(tokens=predicate.tokens[2:])

                if predicate.tags[0] in ["MD", "TO"]:
                    predicate = Predicate(tokens=predicate.tokens[1:])

                if (
                    predicate.predicate.startswith("used to")
                    and predicate.predicate.startswith("use for")
                    and predicate.predicate.startswith("use to")
                    and predicate.predicate.startswith("used for")
                    and predicate.predicate.startswith("used to")
                    and predicate.predicate.startswith("proposed to")
                    and predicate.predicate.startswith("applied to")
                    and predicate.predicate.startswith("managed to")
                    and predicate.predicate.startswith("proved to")
                    and predicate.predicate.startswith("aims to")
                    and predicate.predicate.startswith("aims at")
                    and predicate.predicate.startswith("focus on")
                    and predicate.predicate.startswith("appropriate for")
                    and predicate.predicate.startswith("able to")
                    and predicate.predicate.startswith("allowing to")
                    and predicate.predicate.startswith("consists in")
                    and predicate.predicate.startswith("has potential")
                    and predicate.predicate.startswith("design to")
                ):
                    predicate = Predicate(tokens=predicate.tokens[2:])

                if predicate.pattern.startswith("JJ TO") and len(predicate.tags):
                    predicate = Predicate(tokens=predicate.tokens[2:])

                if predicate.pattern == "IN NN TO":
                    predicate = Predicate(tokens=predicate.tokens[1:])

                if (  ### TODO: check this rule
                    len(predicate.tags) > 4 and predicate.tokens[4].simple_tag == "VB"
                ):
                    predicate = Predicate(tokens=predicate.tokens[4:])

        if len(predicate.tags) > 1:

            if predicate.values[0] in [
                "enable",
                "enables" "allow",
                "allows",
                "permit",
                "permits",
                "allow",
                "allows",
                "help",
                "helps",
            ]:
                predicate = Predicate(tokens=predicate.tokens[1:])

            if predicate.tags[0] in ["MD", "TO"]:
                predicate = Predicate(tokens=predicate.tokens[1:])

        ### Special Maps
        if predicate.predicate in self.domain_nlp.mapping:
            predicate, is_passive = self.domain_nlp.mapping.get(predicate.predicate)
            predicate = self.make_predicate(predicate)

        ### Parts to be moved to tail entity
        if len(predicate.tags) > 3 and predicate.pattern in ["JJ NN IN", "NNS NNS IN"]:
            E2 = f"{{E2}} {predicate.values[1]}"
            predicate = Predicate(tokens=predicate.tokens[0:1])

        if (
            len(predicate.tags) > 3
            and predicate.pattern in ["VBG NN IN"]
            and predicate.values[1].endswith("tion")
        ):
            E2 = f"{{E2}} {predicate.values[1]}"
            predicate = Predicate(tokens=predicate.tokens[0:1])

        if (
            len(predicate.tags) > 3
            and predicate.predicate.startswith("led to")
            and predicate.tags[2] == "NN"
        ):
            E2 = f"{{E2}} {predicate.values[2]}"
            predicate = Predicate(tokens=predicate.tokens[0:2])

        return predicate, is_passive, E2

    def is_valid_predicate(self, pred: Predicate) -> bool:

        is_valid = len(pred.predicate) > 1
        is_valid = len(pred.tags) > 0 and len(pred.tags) <= self.MAX_PRED_LEN

        is_valid = is_valid and not any(
            [stop_rel in pred.values for stop_rel in self.domain_nlp.stop.predicates]
        )

        is_valid = is_valid and "CC" not in pred.tags

        is_valid = is_valid and not all(
            [token.value in self.domain_nlp.propositions for token in pred.tokens]
        )

        is_valid = is_valid and pred.pattern not in [
            "IN",
            "TO",
            "JJ NN IN",
            "NN IN NN IN",
            # 'JJ IN',
            # 'MD VB',
            # 'VB RB',
            # 'JJ NN',
            # 'NN RB',
            # 'RB TO',
            # 'NN JJ',
            # 'IN NN',
            # 'IN IN',
            # 'NN WR',
            # 'VB WR',
            # 'JJ VB',
            # 'RB JJ',
            # 'WR VB',
            # 'NN NN CD NN',
            # 'VB DT',
            # 'IN JJ',
            # 'FW NN',
            # 'JJ JJ'
        ]


        is_valid = is_valid

        return is_valid

    def is_valid_entity(self, entity: str) -> bool:

        is_valid = len(entity) > 1

        tokens = word_tokenize(entity)

        is_valid = len(tokens) > 0 and len(tokens) <= self.MAX_ENTITY_LEN

        is_valid = is_valid and not entity.isnumeric()

        is_valid = is_valid = is_valid and not any(
            [stopword in tokens for stopword in self.domain_nlp.stop.entities]
        )

        return is_valid

    def is_valid_triple(self, triple: Tuple[str, Predicate, str]) -> bool:

        head, pred, tail = triple

        return (
            self.is_valid_entity(head)
            and self.is_valid_predicate(pred)
            and self.is_valid_entity(tail)
            and (head != tail)
        )

    def get_predicate_group(self, predicate: Predicate) -> Predicate:

        if len(predicate.tags) == 1:
            return predicate

        predicate = Predicate(
            [
                token
                for token in predicate.tokens
                if token.value not in self.domain_nlp.propositions
            ]
        )

        if len(predicate.tags) > 1:

            predicate = Predicate(
                [
                    token
                    for token in predicate.tokens
                    if token.value not in self.domain_nlp.stop.standard_verbs
                ]
            )

        return predicate

    def skip(self, doc: Dict) -> Dict:
        return doc is None

    def transform(self, doc: Dict) -> Dict:

        head, tail, predicate = doc["head"], doc["tail"], doc["predicate"]

        ### Initial Normalization
        head = self.base_normalize(head)
        tail = self.base_normalize(tail)
        predicate = self.base_normalize(predicate)

        predicate = self.make_predicate(predicate)

        ### Pre-Cleaning validation
        if len(predicate.tags) == 1 and predicate.tokens[0].value.endswith('ing'):
            return None

        ### Clean Predicate
        predicate, is_passive, e2 = self.clean_predicate(predicate)

        ### Handle Passive Form & E2
        if is_passive:
            head, tail = tail, head

        if e2 is not None:
            tail = e2.format(E2=tail)

        ### Validate Triple
        if not self.is_valid_triple((head, predicate, tail)):
            return None

        ### Get Predicate Group
        predicate_group = self.get_predicate_group(predicate).predicate

        ### Lemmatization
        predicate = self.lemmetize(predicate.predicate)

        ### Construct Triple
        triple = {
            "id": doc["id"],
            "sentence": doc["sentence"],
            "head": head,
            "predicate": predicate,
            "tail": tail,
            "predicate_group": predicate_group,
        }

        return triple

    @classmethod
    def from_config(cls, config: Config) -> "TriplesPostprocessor":

        cache = TextfileCache(config.checkpoint_filename)

        return TriplesPostprocessor(
            reader=FileReader
                .from_filename(config.input_filename, cache)
                .batched(config.batch_size),
            writer=FileWriter
                .from_filename(config.output_filename)
                .buffered(config.buffer_size, cache),
            language_detector=LanguageDetector.create("lingua"),
            domain_nlp=config.nlp
        )