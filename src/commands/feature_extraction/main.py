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
from src import helpers
from src.shared.file_tools import FileReader
from src.shared.command_tools import BaseCommand, OutputDirOption, add_handlers, ConnectionOption
from src.shared.handlers import Event
from src.commands.feature_extraction.graph_utils import Graph
from src.commands.feature_extraction.graph_filter import FilterConfig, GraphFilter
from src.commands.feature_extraction.graph_builder import GraphBuilder
from src.commands.feature_extraction.graph_embedder import GraphEmbedder
from src.commands.knowledge_graph_construction.main import Config as KGConfig, PipeConfig
from src.envirenment import Domain, env
from typing import Generator, List
from pydantic import BaseModel, computed_field
from definitions import KGS_DIR, FEATURES_DIR
from instance import cli

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
        return KGConfig(graph_id=self.kg_id).conf

    @computed_field
    @property
    def domains(self) -> List[Domain]:
        domains_id = self.kg_config.glob.domain
        return env.fields[domains_id].domains

    @computed_field
    @property
    def label_name(self) -> str:
        return "deepseek-chat"

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
        return FeatureExtractionConfig(**helpers.load_json_file(self.config_path))

@cli.register_stage(
    name="feature_extraction",
    description=(
        "Transforms raw JSONL triples into an embedded, annotated knowledge graph "
        "ready for GNN training. Includes parsing, filtering, sentence-bert embedding, "
        "PageRank scoring, and dual-format output (graph.pt + metadata.json)."
    ),
    add_from_conf=False,
    add_save_to=False,
)
@add_handlers
class FeatureExtractor(BaseCommand):

    def __init__(self, 
        reader : FileReader, 
        filter : FilterConfig,
        model_name : str,
        batch_size : int,
        device : str,
        graph_dir : str,
        domains : List[Domain],
        label_name : str
    ) -> None:
        
        super().__init__()

        self.reader = reader
        self.filter = filter
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.graph_dir = graph_dir
        self.domains = domains
        self.label_name = label_name

    def __iter__(self) -> Generator[Event, None, None]:

        ### Stage 01 : Graph Building
        self.send_info_event(0.0, message="Starting Feature Extraction.")
        self.send_info_event(0.0, message="Starting Stage Graph Construction.")
        graph = GraphBuilder(self, self.reader, self.domains).build()
        self.send_info_event(1/3, message="Stage Graph Construction was completed.")

        ### Stage 02 : Filtering
        self.send_info_event(1/3, message="Starting Stage Graph Fieltering.")
        graph = GraphFilter(self, self.filter, self.label_name).filter_graph(graph)
        self.send_info_event(2/3, message="Stage Graph Filtering was completed.")

        ### Stage 03 : Feature Extraction
        self.send_info_event(2/3, message="Starting Stage Graph Embedder.")
        graph = GraphEmbedder(self, self.model_name, self.batch_size, self.device).extract_features(graph)
        self.send_info_event(3/3, message="Stage Graph Graph Embedder was completed.")

        ### Stage 04 : Save graph
        graph.save(self.graph_dir)     

        yield from []

    
    @classmethod
    def from_config(cls, config: Config) -> "FeatureExtractor":

        fe = FeatureExtractor(
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