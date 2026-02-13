import re
import string
import spacy
from nltk import pos_tag, word_tokenize
from typing import List, Tuple, Dict
from dataclasses import dataclass
from src.shared.command_tools import BaseConfig, command, Stage
from src.shared.file_tools import FileWriter, FileReader
from src.shared.language_detectors import LanguageDetector
from dataclasses import dataclass
from pycli import Model
from instance import cli

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


class TriplesPostprocessor(Stage):

    MAX_PRED_LEN = 4
    MAX_ENTITY_LEN = 6

    STOPWORDS_ENTITIES = [
        "interest",
        "researcher",
        "researchers",
        "phd",
        "thesis",
        "study",
        "studies",
        "approach",
        "work",
        "article",
        "chapter",
        "section",
        "author",
        "results",
        "experiment",
        "experiments",
        "objective",
        "reasearch",
        "conference",
        "contribution",
        "survey",
        "paper",
        "experimental",
        "comparision",
        "we",
        "it",
        "research",
        "literature",
        "review",
        "make",
        "our",
        "proposed",
        "result",
        "supplementary",
    ]

    STOPWORDS_RELATIONS = [
        "suggest",
        "introduce",
        "show",
        "present",
        "can",
        "cannot",
        "will",
        "could",
        "like",
        "hard",
        "impossible",
        "not",
        "n't",
        "unable",
        "difficult",
        "negatively",
        "fail",
        "no",
        "became",
        "should",
        "must",
        "may",
        "without",
        "est",
        "non",
        "incomplete",
        "lack",
        "unavailable",
    ]

    PROPOSITIONS = [
        "of",
        "on",
        "in",
        "with",
        "into",
        "onto",
        "between",
        "below",
        "above",
        "under",
        "by",
        "for",
        "at",
        "over",
        "throught",
        "have",
        "to",
        "from",
    ]

    STANDARD_VERBS = [
        "be",
        "have",
    ]

    def __init__(
        self,
        reader: FileReader,
        writer: FileWriter,
        batch_size: int,
        buffer_size: int,
        language_detector: LanguageDetector,
    ) -> None:
        super().__init__(reader, writer, batch_size, buffer_size)
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

        maps = {
            # "such as" : ('is',True),
            "is namely": ("have abbreviation", False),
            "is abbreviated as": ("have abbreviation", False),
            "are abbreviated as": ("have abbreviation", False),
            "is abbreviation of": ("have abbreviation", True),
            "is useful for": ("used for", False),
            "find applications in": ("used for", False),
            "are solution for": ("solve", False),
            "is referred to as": ("have abbreviation", False),
        }

        ### Remove stop words
        predicate = Predicate(
            [
                token
                for token in predicate.tokens
                if token.tag != "DT"
                and token.value
                not in ["a", "the", "an", "also", "key", "challenge", "new"]
            ]
        )

        if len(predicate.tags) == 0:
            return predicate, is_passive, E2

        ### Passive Form
        if predicate.values[-1] in ["by", "using"]:
            is_passive = True
            predicate = Predicate(tokens=predicate.tokens[:-1])

        ### Retain core action
        if len(predicate.tokens) > 2:

            if predicate.values[0] in ["is", "was", "are", "be", "were", "being"] and (
                predicate.tags[1][:2] == "VB" or predicate.values[1] in ["more", "less"]
            ):
                predicate = Predicate(tokens=predicate.tokens[1:])

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
        if predicate.predicate in maps:
            predicate, is_passive = maps.get(predicate.predicate)
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
            [stop_rel in pred.values for stop_rel in self.STOPWORDS_RELATIONS]
        )

        is_valid = is_valid and "CC" not in pred.tags

        is_valid = is_valid and not all(
            [token.value in self.PROPOSITIONS for token in pred.tokens]
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


        is_valid = is_valid # and self.language_detector.is_english(pred.predicate)

        return is_valid

    def is_valid_entity(self, entity: str) -> bool:

        is_valid = len(entity) > 1

        tokens = word_tokenize(entity)

        is_valid = len(tokens) > 0 and len(tokens) <= self.MAX_ENTITY_LEN

        is_valid = is_valid and not entity.isnumeric()

        is_valid = is_valid = is_valid and not any(
            [stopword in tokens for stopword in self.STOPWORDS_ENTITIES]
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
                if token.value not in self.PROPOSITIONS
            ]
        )

        if len(predicate.tags) > 1:

            predicate = Predicate(
                [
                    token
                    for token in predicate.tokens
                    if token.value not in self.STANDARD_VERBS
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

        ### Clean Predicate
        predicate = self.make_predicate(predicate)
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
            "sentence_id": doc["sentence_id"],
            "document_id": doc["document_id"],
            "sentence": doc["sentence"],
            "head": head,
            "predicate": predicate,
            "tail": tail,
            "predicate_group": predicate_group,
        }

        return triple

@cli.register_command(
    name="clean_triples",
    description=(
        "Apply a set of rules to clean the triples and retain the main verb in the predicate."
    ),
    model=Model(BaseConfig, spread=False),
    add_from_conf=False,
    add_save_to=False
)
@command
def postprocess_triples(config: BaseConfig) -> "TriplesPostprocessor":

    return TriplesPostprocessor(
        reader=FileReader.from_filename(config.input_filename),
        writer=FileWriter.from_filename(config.output_filename),
        batch_size=config.batch_size,
        buffer_size=config.buffer_size,
        language_detector=LanguageDetector.create("lingua"),
    )