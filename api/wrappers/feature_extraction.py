"""
Knowledge Graph Feature Extraction Pipeline
============================================
Processes JSONL triples into an embedded, annotated graph ready for GNN training.

Stages:
  1. Parse  – Build graph structure from raw JSONL records
  2. Filter – Prune by connectivity, coverage, and label criteria
  3. Embed  – Encode nodes/edges with SentenceTransformer
  4. Score  – Add generality, relevancy, PageRank, degree metadata
  5. Save   – Write training tensors + Neo4j-ready metadata (link preserved)

Entry-point:
    python feature_extraction.py                          # defaults
    python feature_extraction.py --corpus my-corpus
    python feature_extraction.py --min-entity-articles 3
"""

import os
from typing import List
from pydantic import BaseModel, computed_field
from definitions import FEATURES_DIR, KGS_DIR
from src.envirenment import env, Domain
from src.helpers.utils import load_json_file
from src.shared.command_tools import OutputDirOption, ConnectionOption, add_handlers
from src.shared.file_tools import FileReader
from src.commands.feature_extraction.main import FeatureExtractor as _FeatureExtractor, Config, FilterConfig
from api.wrappers.knowledge_graph_construction import Config as KGConfig, PipeConfig
from api.constants import SUBDOMAINS_LABEL_KEY

class FeatureExtractionConfig(BaseModel):
    kg_id : str
    model_name: str
    batch_size: int
    device: int
    filter : FilterConfig
    stage : str =  'llm_based_entities_cleaning'

    @computed_field
    @property
    def kg_config(self) -> PipeConfig:
        return KGConfig(graph_id=self.kg_id, api_key="no-need").conf

    @computed_field
    @property
    def domains(self) -> List[Domain]:
        domains_id = self.kg_config.glob.domain
        return env.fields[domains_id].domains

    @computed_field
    @property
    def label_name(self) -> str:
        return SUBDOMAINS_LABEL_KEY

    @computed_field
    @property
    def raw_path(self) -> str:
        return os.path.join(KGS_DIR, self.kg_id, self.stage, 'data.jsonl')
    
class Config(OutputDirOption, ConnectionOption):
    features_id : str

    def output_dir(self) -> str:
        return os.path.join(FEATURES_DIR, self.features_id)
    
    @computed_field
    @property
    def config_path(self) -> str:
        return os.path.join(self.output_dir(), 'config.json')
    
    @computed_field
    @property
    def conf(self) -> FeatureExtractionConfig:
        return FeatureExtractionConfig(**load_json_file(self.config_path))

@add_handlers
class FeatureExtractor(_FeatureExtractor):

    @classmethod
    def from_config(cls, config: Config) -> "FeatureExtractor":

        fe = cls(
            reader=FileReader.from_filename(filename=config.conf.raw_path),
            filter=config.conf.filter,
            model_name=config.conf.model_name,
            batch_size=config.conf.batch_size,
            device=config.conf.device,
            graph_dir=config.output_dir(),
            domains=config.conf.domains,
            label_name=config.conf.label_name
        )

        return fe