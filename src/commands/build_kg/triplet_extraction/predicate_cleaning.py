import re
import string
import spacy
from nltk import pos_tag, word_tokenize
from typing import List, Tuple, Dict, Callable
from dataclasses import dataclass
from src.helpers import load_json_file
from src.shared.command_tools import StageConfig, Stage, add_handlers
from src.shared.file_tools import FileWriter, FileReader, TextfileCache
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
    name="predicates_normalization",
    description=(
        "Apply a set of rules to clean the triples and retain the main verb in the predicate."
    ),
    add_from_conf=False,
    add_save_to=False
)
@add_handlers
class PredicatesNormalizer(Stage):

    MAX_PRED_LEN = 4
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

    def base_normalize(self, entity: str) -> str:

        # Remove punctuations
        translator = str.maketrans("", "", string.punctuation)

        # Remove hyphen and back-slash
        entity = entity.replace("-", " ")
        entity = entity.replace("/", " ")

        # Remove numbers & handle multiple spaces
        entity = re.sub("[0-9]+", "", entity)
        entity = re.sub(r"\s+", " ", entity)
        entity = entity.translate(translator)

        # Lower-casing & stripping leading and trailing white spaces
        entity = entity.lower()
        entity = entity.strip()

        return entity

    def lemmetize(self, predicate: str) -> str:
        return " ".join([token.lemma_ for token in self.nlp(predicate)])

    def make_predicate(self, predicate: str) -> Predicate:
        tokens = pos_tag(word_tokenize(predicate))
        tokens = [Token(token, tag) for token, tag in tokens]
        return Predicate(tokens)

    def normalize_predicate(self, predicate: Predicate) -> Tuple[Predicate, bool, str]:

        is_passive = False
        E2 = None

        to_be_forms = ["is", "was", "are", "be", "were", "being"]

        def make(*tokens : Token):
            return Predicate(list(tokens))
        
        def select(predicate : Predicate, *indices : int) -> Predicate:
            return make(*[token for i,token in enumerate(predicate.tokens) if i in indices])
        
        def filter_predicate(predicate : Predicate, condition : Callable[[int, Token], bool]) -> Predicate:
            return make(*[token for i,token in enumerate(predicate.tokens) if condition(i, token)])

        ### Remove stop words
        predicate = filter_predicate(
            predicate, 
            lambda i, token : (token.tag != "DT") and (token.value not in self.domain_nlp.stop.tokens)
        )

        ### Remove tail stop words
        if len(predicate.tokens) >= 1:
            
            predicate = filter_predicate(
                predicate,
                lambda i, token : (i != len(predicate.tokens) - 1) or (token.value not in self.domain_nlp.stop.tail)
            )

        if len(predicate.tags) == 0:
            return predicate, is_passive, E2
        
        ### Passive Form
        if predicate.values[-1] in ["by", "using"]:
            is_passive = True
            predicate = Predicate(tokens=predicate.tokens[:-1])

        ### Retain core action
        if len(predicate.tokens) >= 2:

            if (predicate.values[0] in ["have", "has", "had"] and 
                predicate.values[1] in ["been"]
            ):
                predicate = self.make_predicate(" ".join(["be"] + predicate.values[2:]))

        if len(predicate.tokens) >= 2:

            if predicate.values[0] in to_be_forms and (
                predicate.tags[1][:2] == "VB" 
                or predicate.values[1] in ["more", "less"]
            ):
                predicate = Predicate(tokens=predicate.tokens[1:])

        if len(predicate.tokens) >= 3:

            if (
                predicate.tags[0] == "MD"
                and predicate.tokens[1].value in to_be_forms
                and predicate.tags[2][:2] == "VB"
            ):
                predicate = Predicate(tokens=predicate.tokens[2:])

            if predicate.tags[0] in ["MD", "TO"]:
                predicate = Predicate(tokens=predicate.tokens[1:])

            if (
                predicate.predicate.startswith("used to")
                or predicate.predicate.startswith("use for")
                or predicate.predicate.startswith("use to")
                or predicate.predicate.startswith("used for")
                or predicate.predicate.startswith("used to")
                or predicate.predicate.startswith("proposed to")
                or predicate.predicate.startswith("applied to")
                or predicate.predicate.startswith("managed to")
                or predicate.predicate.startswith("proved to")
                or predicate.predicate.startswith("aims to")
                or predicate.predicate.startswith("aims at")
                or predicate.predicate.startswith("focus on")
                or predicate.predicate.startswith("appropriate for")
                or predicate.predicate.startswith("able to")
                or predicate.predicate.startswith("allowing to")
                or predicate.predicate.startswith("consists in")
                or predicate.predicate.startswith("has potential")
                or predicate.predicate.startswith("design to")
                or predicate.predicate.startswith("allow to")
                or predicate.predicate.startswith("allows to")
                or predicate.predicate.startswith("allowed to")
                or predicate.predicate.startswith("allows for")
                or predicate.predicate.startswith("allowed for")
            ):
                predicate = Predicate(tokens=predicate.tokens[2:])

            if predicate.pattern.startswith("JJ TO") and len(predicate.tags) >= 3:
                predicate = Predicate(tokens=predicate.tokens[2:])

            if predicate.pattern == "IN NN TO":
                predicate = Predicate(tokens=predicate.tokens[1:])

            if (  ### TODO: check this rule
                len(predicate.tags) > 4 and 
                predicate.tokens[4].simple_tag == "VB"
            ):
                predicate = Predicate(tokens=predicate.tokens[4:])

        if len(predicate.tags) > 1:

            if predicate.values[0] in [
                "enable",
                "enables" 
                "allow",
                "allows",
                "permit",
                "permits",
                "help",
                "helps",
            ]:
                predicate = Predicate(tokens=predicate.tokens[1:])

            if predicate.tags[0] in ["MD", "TO"]:
                predicate = Predicate(tokens=predicate.tokens[1:])

        if (
            len(predicate.tags) >= 4 and 
            predicate.matches("#VB #JJ") and 
            predicate.values[1] not in ["similar", "superior"]
        ):
            return select(predicate, 0, *range(2, len(predicate.tags))), is_passive, E2

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

        is_valid = is_valid and pred.simple_pattern not in [
            "IN",
            "TO",
            "JJ NN IN",
            "NN IN NN IN",
            'JJ IN',
            'MD VB',
            'VB RB',
            'JJ NN',
            'NN RB',
            'RB TO',
            'NN JJ',
            'IN NN',
            'IN IN',
            'NN WR',
            'VB WR',
            'JJ VB',
            'RB JJ',
            'WR VB',
            'NN NN CD NN',
            'VB DT',
            'IN JJ',
            'FW NN',
            'JJ JJ'
        ]

        is_valid = is_valid

        return is_valid

    def is_valid_entity(self, entity: str) -> bool:

        is_valid = len(entity) > 1

        tokens = word_tokenize(entity)

        is_valid = len(tokens) > 0 and len(tokens) <= self.MAX_ENTITY_LEN and len(entity) > 1

        is_valid = is_valid and entity.replace(' ', '').isalpha()

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
        predicate = self.base_normalize(predicate)
        predicate = self.make_predicate(predicate)

        ### Clean Predicate
        predicate, is_passive, e2 = self.normalize_predicate(predicate) # self.clean_predicate(predicate)

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
    def from_config(cls, config: Config) -> Stage:

        cache = TextfileCache(config.checkpoint_filename)

        return PredicatesNormalizer(
            reader=FileReader
                .from_filename(config.input_filename, cache)
                .batched(config.batch_size),
            writer=FileWriter
                .from_filename(config.output_filename)
                .buffered(config.buffer_size, cache),
            domain_nlp=config.nlp
        )