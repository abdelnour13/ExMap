import re
import string
import spacy
from typing import Dict
from src.shared.command_tools import StageConfig, Stage, add_handlers
from src.shared.file_tools import FileWriter, FileReader, TextfileCache
from src.commands.knowledge_graph_construction.graph_cleaning.utils import Predicate
from src.commands.knowledge_graph_construction.graph_cleaning.rules import (
    STOP_PREDICATES,
    PROPOSITIONS,
    INVALID_PATTERNS,
    RULES,
    apply_all_rules,
    TOKENS_TO_REMOVE
)
from instance import cli

class Config(StageConfig):
    pass

@cli.register_stage(
    name="rule_based_predicate_cleaning",
    description=(
        "Apply a set of rules to clean the triples and retain the main verb in the predicate."
    ),
    add_from_conf=False,
    add_save_to=False
)
@add_handlers
class PredicatesNormalizer(Stage):

    MAX_PRED_LEN = 4

    def __init__(
        self,
        reader: FileReader,
        writer: FileWriter,
    ) -> None:
        super().__init__(reader, writer)
        self.nlp = spacy.load("en_core_web_sm")
        self._cache : Dict[str, str | bool] = {}

    def base_normalize(self, entity: str) -> str:

        # Remove punctuations
        entity = entity.replace('_', ' ')
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

    def is_valid_predicate(self, pred : Predicate) -> bool:

        is_valid = len(pred.predicate) > 1
        is_valid = is_valid and len(pred.tags) > 0 and len(pred.tags) <= self.MAX_PRED_LEN
        is_valid = is_valid and not any([stop_rel in pred.words for stop_rel in STOP_PREDICATES])
        is_valid = is_valid and "CC" not in pred.simple_tags
        is_valid = is_valid and not all([token.word in PROPOSITIONS for token in pred.tokens])
        is_valid = is_valid and pred.simple_tag not in INVALID_PATTERNS

        return is_valid

    def skip(self, doc: Dict) -> Dict:
        return doc is None

    def normalize_predicate(self, predicate : str) -> str:
        return apply_all_rules(predicate, RULES)

    def transform(self, doc: Dict) -> Dict:

        head, tail, unormalized_predicate = doc["head"], doc["tail"], doc["predicate"]

        unormalized_predicate = ' '.join([token for token in unormalized_predicate.split(' ') if token not in TOKENS_TO_REMOVE])

        if len(unormalized_predicate.strip()) == 0:
            return None

        ### Initial Normalization
        unormalized_predicate = self.base_normalize(unormalized_predicate)

        ### Clean Predicate
        if unormalized_predicate in self._cache:

            template = self._cache[unormalized_predicate]["template"]
            predicate_lemma = self._cache[unormalized_predicate]["lemma"]
            is_valid = self._cache[unormalized_predicate]["is_valid"]

            template = template.format(head=head, tail=tail)
            head, predicate, tail = template.split(' | ')

        else:

            template = self.normalize_predicate(unormalized_predicate) # self.clean_predicate(predicate)
            formatted_template = template.format(head=head, tail=tail)
            head, predicate, tail = formatted_template.split(' | ')
            head, predicate, tail = head.strip(), predicate.strip(), tail.strip()

            ### Validate Triple
            is_valid = self.is_valid_predicate(Predicate.from_string(predicate))

            ### Lemmatization
            predicate_lemma = self.lemmetize(predicate)

            self._cache[unormalized_predicate] = {
                "template" : template,
                "lemma" : predicate_lemma,
                "is_valid" :  is_valid   
            }

        if not is_valid:
            return None

        ### Construct Triple
        triple = {
            "id": doc["id"],
            "sentence": doc["sentence"],
            "head": head,
            "predicate": predicate_lemma,
            "tail": tail,
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
        )