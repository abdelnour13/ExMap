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
from src.shared.command_tools import BaseCommand
from src.shared.handlers import Event, ConsoleLogger
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
    def raw_path(self) -> str:
        return os.path.join(KGS_DIR, self.kg_id, self.stage, 'data.jsonl')
    
class Config(BaseModel):
    features_id : str

    @computed_field
    @property
    def features_dir(self) -> str:
        return os.path.join(FEATURES_DIR, self.features_id)
    
    @computed_field
    @property
    def config_path(self) -> str:
        return os.path.join(self.features_dir, 'config.json')
    
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
class FeatureExtractor(BaseCommand):

    def __init__(self, 
        reader : FileReader, 
        filter : FilterConfig,
        model_name : str,
        batch_size : int,
        device : str,
        graph_dir : str,
        domains : List[Domain]
    ) -> None:
        
        super().__init__()

        self.reader = reader
        self.filter = filter
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.graph_dir = graph_dir
        self.domains = domains

    def __iter__(self) -> Generator[Event, None, None]:

        ### Stage 01 : Graph Building
        graph = GraphBuilder(self, self.reader, self.domains).build()

        ### Stage 02 : Filtering
        graph = GraphFilter(self, self.filter).filter_graph(graph)

        ### Stage 03 : Feature Extraction
        graph = GraphEmbedder(self, self.model_name, self.batch_size, self.device).extract_features(graph)

        ### Stage 04 : Save graph
        graph.save(self.graph_dir)

        # Load & Test
        g = Graph.load(self.graph_dir)   

        print(g.get_node('Entity', 'model'))     

        yield from []

    
    @classmethod
    def from_config(cls, config: Config) -> "FeatureExtractor":

        fe = FeatureExtractor(
            reader=FileReader.from_filename(filename=config.conf.raw_path),
            filter=config.conf.filter,
            model_name=config.conf.model_name,
            batch_size=config.conf.batch_size,
            device=config.conf.device,
            graph_dir=config.features_dir,
            domains=config.conf.domains
        )

        fe.add_handler(ConsoleLogger("FeatureExtraction"))

        return fe