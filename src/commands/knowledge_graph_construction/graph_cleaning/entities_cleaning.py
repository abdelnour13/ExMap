import re
import string
import spacy
import pandas as pd
from textblob import Word
from nltk import pos_tag, word_tokenize
from typing import Dict, Optional
from src.shared.handlers import Event, EventLevel
from src.shared.command_tools import StageConfig, Stage, add_handlers
from src.shared.file_tools import FileWriter, FileReader, TextfileCache
from pycli import Field
from instance import cli
from src.commands.knowledge_graph_construction.graph_cleaning.rules import (
    STOP_ENTITIES
)

class Config(StageConfig):
    acronym_threshold : float = 0.75

@cli.register_stage(
    name="entities_normalization",
    description=(
        "Apply a set of rules to clean the triples and retain the main verb in the predicate."
    ),
    add_from_conf=False,
    add_save_to=False
)
@add_handlers
class EntitiesNormalizer(Stage):

    MAX_ENTITY_LEN = 6

    def __init__(
        self,
        reader: FileReader,
        writer: FileWriter,
        acronym_threshold : float
    ) -> None:
        super().__init__(reader, writer)
        self.acronym_threshold = acronym_threshold
        self.acronyms_map : pd.DataFrame = None
        self.mapped_acronyms = 0

    def build_acronyms_map(self, reader : FileReader) -> pd.DataFrame:

        acronyms_map = [
            {
                "acronym" : "ml",
                "expansion" : "machine learning",
                "doc_id" : 0,
                "score" : 1.0
            }
        ]

        for records, _ in reader:

            for record in records:
                    
                for acronym in record['sentence']['acronyms']:
                
                    acronyms_map.append({
                        "acronym" : acronym['acronym'],
                        "expansion" : acronym['expansion'],
                        "doc_id" : record['sentence']['document']['id'],
                        "score" : acronym['score']
                    })

        acronyms_map = pd.DataFrame(acronyms_map, columns=["acronym", "expansion", "doc_id", "score"])

        def _pick_acronym(x : pd.Series) -> dict:

            expansion, score = sorted(zip(x['expansion'], x['score']), key=lambda x : x[1])[-1]

            return {
                "expansion" : expansion,
                "score" : score
            }

        acronyms_map = acronyms_map.groupby(by=['acronym', 'doc_id']).agg(list).apply(
            _pick_acronym,
            axis=1
        )

        acronyms_map = pd.DataFrame(data=acronyms_map.tolist(), index=acronyms_map.index).reset_index()

        acronyms_map['acronym'] = acronyms_map['acronym'].str.lower()
        acronyms_map['expansion'] = acronyms_map['expansion'].apply(self.normalize_entity)

        acronyms_map = acronyms_map.groupby(by=['acronym', 'expansion']).agg({
            'doc_id' : list,
            'score' : 'max'
        })

        acronyms_map = acronyms_map.query(f"score >= {self.acronym_threshold}")

        return acronyms_map

    def normalize_entity(self, entity: str) -> str | None:

        # Remove punctuations
        translator = str.maketrans("", "", "".join(set(string.punctuation).difference("()")))

        # Remove hyphens and back-slash
        entity = entity.replace("-", " ")
        entity = entity.replace("/", " ")

        # Remove extra spaces & punctuations & parenthesis 
        entity = re.sub(r'\([^)]*\)', '', entity)
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

        if not self.is_valid_entity(entity):
            return None

        return entity

    def is_valid_entity(self, entity : str) -> bool:

        return (
            (len(entity) >= 2) and 
            (not any([token in STOP_ENTITIES for token in entity.split(' ')])) and
            (len(entity.split(' ')) <= self.MAX_ENTITY_LEN)
        )
    
    def before_start(self):

        self.acronyms_map = self.build_acronyms_map(self.reader)

        yield self.make_info_event(
            progress=None,
            message="Acronyms Map Created"
        )

    def search_for_acronym(self, 
        acronym : str, 
        doc_id : str,
        map : pd.DataFrame
    ) -> str | None:
                
        try:
            candidates = map.loc[acronym].reset_index()
        except:
            return None
        
        candidates = candidates[candidates['doc_id'].apply(lambda x : doc_id in x)]

        if len(candidates) == 0:
            return None
        
        return candidates.iloc[0]['expansion']
    
    def replace_acronyms(self,
        entity : str,
        doc_id : str,
        map : pd.DataFrame                     
    ) -> str:
        
        tokens = entity.split(' ')
        
        new_tokens = []

        for token in tokens:

            expansion = self.search_for_acronym(token, doc_id, map)

            if expansion is None:
                new_tokens.append(token)
            else:
                self.mapped_acronyms += 1
                new_tokens.append(expansion)

        return ' '.join(new_tokens)
    
    def make_event(self,
        progress : Optional[float],
        level : EventLevel,
        message : str,
        task : str = "main",
        extra : Optional[Dict] = None
    ) -> Event:
        
        return super().make_event(
            progress=progress, 
            level=level, 
            message=message,
            task=task,
            extra={
                'total_processed' : self.total_processed,
                'total_saved' : self.total_saved,
                'total_skipped' : self.total_skipped,
                'total_errors' : self.total_errors,
                'total_acronyms_mapped' : self.mapped_acronyms,
                **(extra or {})
            }
        ) 
    
    def transform(self, doc: Dict) -> Dict:

        doc['head'] = self.normalize_entity(doc['head'])
        doc['tail'] = self.normalize_entity(doc['tail'])

        if (doc['head'] is None) or (doc['tail'] is None):
            return doc

        doc['head'] = self.replace_acronyms(doc['head'], doc['sentence']['document']['id'], self.acronyms_map)
        doc['tail'] = self.replace_acronyms(doc['tail'], doc['sentence']['document']['id'], self.acronyms_map)

        normalized_acronyms = []
        normalized_keywords = {}

        for acronym in doc['sentence']['acronyms']:

            acronym['expansion'] = self.normalize_entity(acronym['expansion'])

            if acronym['expansion'] is not None:
                normalized_acronyms.append(acronym)

        for keyword in doc['sentence']['document']['keywords']:

            keyword['keyword'] = self.normalize_entity(keyword['keyword'])

            if keyword['keyword'] is None:
                continue

            keyword['keyword'] = self.replace_acronyms(
                keyword['keyword'], 
                doc['sentence']['document']['id'], 
                self.acronyms_map
            )

            if keyword['keyword'] not in normalized_keywords:
                normalized_keywords[keyword['keyword']] = keyword

        doc['sentence']['acronyms'] = normalized_acronyms
        doc['sentence']['document']['keywords'] = list(normalized_keywords.values())

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
            acronym_threshold=config.acronym_threshold
        )