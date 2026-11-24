import os
from torch import Tensor
from typing import List, Self, TypeVar
from pydantic import computed_field
from neo4j import Session, Transaction, Driver
from instance import cli
from definitions import DATA_DIR
from src.shared.file_tools import Batched
from src.envirenment import env
from src.commands.feature_extraction.graph_utils import Graph, Edge, Node
from src.shared.command_tools import BaseCommand, add_handlers, ConnectionOption, OutputDirOption
from uuid import uuid4
from functools import cached_property

T = TypeVar('T')

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class Config(ConnectionOption, OutputDirOption):

    graph_id: str
    batch_size: int = 500
    neo4j_driver : str = "local"

    def output_dir(self):
        return os.path.join(DATA_DIR, "neo4j")

    @computed_field
    @property
    def expertise_dir(self) -> str:
        return os.path.join(DATA_DIR, *self.graph_id.split('/'))

    @property
    def driver(self) -> Driver:
        return env.neo4j[self.neo4j_driver].driver

    @computed_field
    @cached_property
    def log_filename(self) -> str:
        return os.path.join(self.output_dir(), f"{self.graph_id.replace('/', '::')}-{str(uuid4())}-log.jsonl")

# ---------------------------------------------------------------------------
# Uploader
# ---------------------------------------------------------------------------

@cli.register_stage(
    name="neo4j_upload",
    description="Uploads a graph JSON file (feature_extraction / expertise_map format) to Neo4j.",
    add_from_conf=False,
    add_save_to=False,
)
@add_handlers
class Neo4jUploader(BaseCommand):

    def __init__(self, 
        config: Config, 
        driver : Driver
    ) -> None:
        super().__init__()
        self.config = config
        self.driver = driver

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _run_batched(self, 
        session : Session, 
        query: str, 
        name : str,
        items: List[Edge] | List[Node], 
        graph_id : str
    ) -> None:

        total = len(items)
        uploaded = 0
        is_edge = isinstance(items[0], Edge)
        hint = 'edge' if is_edge else 'node'

        def run_tx(tx : Transaction, batch):
            tx.run(query, batch=batch, timeout=120)

        it = Batched(items, self.config.batch_size)
        it = self.track(it=it, task=f"[UPLOADING {hint}::{name} | {total} {hint.capitalize()+'s'}.]")

        for batch in it:

            batch = list(batch)

            for i in range(len(batch)):

                batch[i] = {
                    'graph_id' : graph_id,
                    **(
                        {
                            "src" : f"{graph_id}_{batch[i].src}",
                            "dst" : f"{graph_id}_{batch[i].dst}"
                        } if is_edge else {}
                    ),
                    **{
                        k : v
                        for k, v in batch[i].properties.items()
                        if not isinstance(v, Tensor)
                    },
                    'id' : f"{graph_id}_{batch[i].id}",
                }

            session.execute_write(run_tx, batch)
            uploaded += len(batch)

    # ------------------------------------------------------------------
    # Upload steps
    # ------------------------------------------------------------------

    def _clear_db(self, session : Session, graph_id : str) -> None:
        self.send_info_event(progress=0.0, message="[Neo4j] Clearing all nodes and relationships …")
        session.run(f"MATCH (n {{ graph_id: \"{graph_id}\" }}) DETACH DELETE n;")
        self.send_info_event(progress=0.0, message="[Neo4j] Database cleared")

    IGNORED_NODE_TYPES = {"Author", "Keyword"}

    def _upload_nodes(self, session : Session, graph : Graph, graph_id : str) -> None:

        for node_type, node_list in self.track(graph._nodes.items(), task="[NODES UPLOAD]"):

            if not node_list or node_type in self.IGNORED_NODE_TYPES:
                continue

            query = (
                f"UNWIND $batch AS node "
                f"CREATE (n:{node_type}) "
                f"SET n = node"
            )

            self._run_batched(
                session, 
                query, 
                node_type,
                node_list.nodes, 
                graph_id
            )

    def _create_indexes(self, session : Session, graph : Graph) -> None:

        self.send_info_event(progress=0.0, message="[Neo4j] Creating indexes on node ids …")

        for node_type in graph._nodes.keys():

            if node_type in self.IGNORED_NODE_TYPES:
                continue

            query = f"CREATE INDEX IF NOT EXISTS FOR (n:{node_type}) ON (n.id)"
            session.run(query)

            query = f"CREATE INDEX IF NOT EXISTS FOR (n:{node_type}) ON (n.graph_id)"
            session.run(query)

    def _upload_edges(self, session: Session, graph : Graph, graph_id : str) -> None:

        for edge_key, edge_list in self.track(graph._edges.items(), task="[EDGES UPLOAD]"):

            if not edge_list:
                continue

            src_type = edge_key.src
            dst_type = edge_key.dst
            rel_type = edge_key.name.upper()
            rel_type = rel_type.replace("-", "_").upper()

            if src_type in self.IGNORED_NODE_TYPES or dst_type in self.IGNORED_NODE_TYPES:
                continue

            query = (
                f"UNWIND $batch AS edge "
                f"MATCH (src:{src_type} {{id: edge.src}}) "
                f"MATCH (dst:{dst_type} {{id: edge.dst}}) "
                f"CREATE (src)-[r:{rel_type}]->(dst) "
                f"SET r = edge"
            )

            self._run_batched(session, query, edge_key, edge_list._edges, graph_id)

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------

    def _create_missing_relations(self, session : Session, graph_id : str) -> None:

        session.run("""
            WITH EXISTS {
                (:Entity {graph_id: $graph_id})-[:APPEARSIN]->(:Article {graph_id: $graph_id})
            } AS alreadyThere
            WHERE NOT alreadyThere
            MATCH (tr:Triple {graph_id: $graph_id})-[:BELONGSTO]->(a:Article {graph_id: $graph_id})
            CALL {
                WITH tr, a
                MATCH (tr)-[:HAS_HEAD|HAS_TAIL]->(e:Entity {graph_id: $graph_id})
                MERGE (e)-[:APPEARSIN]->(a)
            } IN TRANSACTIONS OF 5000 ROWS;
        """, graph_id=graph_id)

        session.run("""
            WITH EXISTS {
                (:Entity {graph_id: $graph_id})-[:HASRELATIONWITH]->(:Entity {graph_id: $graph_id})
            } AS alreadyThere
            WHERE NOT alreadyThere
            MATCH (tr:Triple {graph_id: $graph_id})
            CALL {
                WITH tr
                MATCH (tr)-[:HAS_HEAD]->(h:Entity {graph_id: $graph_id})
                MATCH (tr)-[:HAS_TAIL]->(t:Entity {graph_id: $graph_id})
                MATCH (tr)-[:HAS_PREDICATE]->(p:Predicate {graph_id: $graph_id})
                OPTIONAL MATCH (tr)-[:BELONGSTO]->(a:Article {graph_id: $graph_id})
                WITH h, t, p, collect(DISTINCT toString(a.id)) AS arts
                MERGE (h)-[r:HASRELATIONWITH {predicate: p.name}]->(t)
                SET r.articles = coalesce(r.articles, []) +
                                [x IN arts WHERE NOT x IN coalesce(r.articles, [])]
            } IN TRANSACTIONS OF 5000 ROWS;
        """, graph_id=graph_id)

    def __iter__(self):

        self.send_info_event(
            task="main",
            progress=0.0,
            message=f"[Neo4j] Loading graph from {self.config.expertise_dir}",
        )

        graph = Graph.load(self.config.expertise_dir)

        n_nodes = sum(graph.node_counts.values())
        n_edges = sum(graph.edge_counts.values())

        self.send_info_event(
            task="main",
            progress=0.0,
            message=f"[Neo4j] Graph loaded — {n_nodes:,} nodes, {n_edges:,} edges",
        )

        graph_id = next(iter(graph.get_nodes('ExpertiseMap').nodes_generator)).id

        with self.driver.session() as session:

            self._clear_db(session, graph_id)
            self.send_info_event(progress=0.2, message="[Neo4j] Database cleared")

            self._upload_nodes(session, graph, graph_id)
            self.send_info_event(progress=0.4, message="[Neo4j] Nodes Uploaded")

            self._create_indexes(session, graph)
            self.send_info_event(progress=0.6, message="[Neo4j] Indexes Created")

            self._upload_edges(session, graph, graph_id)
            self.send_info_event(progress=0.8, message="[Neo4j] Edges Uploaded")

            self._create_missing_relations(session, graph_id)
            self.send_info_event(progress=1.0, message="[Neo4j] Missing Relations Created.")

        # self.driver.close()

        self.send_info_event(
            progress=1.0,
            message="[Neo4j] Upload completed successfully",
        )

        yield from []

    @classmethod
    def from_config(cls, config: Config) -> Self:

        uploader = Neo4jUploader(
            config=config,
            driver=config.driver
        )

        return uploader