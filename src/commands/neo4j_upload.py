import os
import time
from torch import Tensor
from typing import Any, List, Self, TypeVar
from pydantic import computed_field
from neo4j import Session, Transaction, Driver, unit_of_work
from neo4j.exceptions import TransientError
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

    IGNORED_NODE_TYPES = {"Author", "Keyword"}

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

    @staticmethod
    def _run_auto_commit(
        session : Session,
        query : str,
        params : dict[str, Any] | None = None,
        retries : int = 5,
    ) -> None:
        """
        Run an auto-commit query (needed for CALL { ... } IN TRANSACTIONS),
        consume it so errors surface here, and retry transient errors.
        Only use with idempotent queries (MERGE, CREATE INDEX IF NOT EXISTS, ...).
        """
        for attempt in range(retries):
            try:
                session.run(query, parameters=params or {}).consume()
                return
            except TransientError:
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt)

    def _run_batched(self,
        session : Session,
        query: str,
        name : str,
        items: List[Edge] | List[Node],
        graph_id : str
    ) -> None:

        if not items:
            return

        total = len(items)
        is_edge = isinstance(items[0], Edge)
        hint = 'edge' if is_edge else 'node'

        # The timeout must be set on the transaction, not passed as a query parameter.
        @unit_of_work(timeout=120)
        def run_tx(tx : Transaction, batch):
            tx.run(query, batch=batch).consume()

        it = Batched(items, self.config.batch_size)
        it = self.track(it=it, task=f"[UPLOADING {hint}::{name} | {total} {hint.capitalize()+'s'}.]")

        for batch in it:

            rows = [
                {
                    'graph_id' : graph_id,
                    **(
                        {
                            "src" : f"{graph_id}_{item.src}",
                            "dst" : f"{graph_id}_{item.dst}"
                        } if is_edge else {}
                    ),
                    **{
                        k : v
                        for k, v in item.properties.items()
                        if not isinstance(v, Tensor)
                    },
                    'id' : f"{graph_id}_{item.id}",
                }
                for item in batch
            ]

            # execute_write retries TransientErrors automatically
            session.execute_write(run_tx, rows)

    # ------------------------------------------------------------------
    # Upload steps
    # ------------------------------------------------------------------

    def _clear_db(self, session : Session, graph_id : str) -> None:

        self.send_info_event(progress=0.0, message="[Neo4j] Clearing all nodes and relationships …")

        # Delete in chunks: one huge DETACH DELETE can exhaust transaction memory.
        # Each chunk is its own retried write transaction.
        def delete_chunk(tx : Transaction, gid : str) -> int:
            record = tx.run(
                """
                MATCH (n {graph_id: $graph_id})
                WITH n LIMIT 10000
                DETACH DELETE n
                RETURN count(*) AS deleted
                """,
                graph_id=gid,
            ).single()
            return record["deleted"]

        while session.execute_write(delete_chunk, graph_id) > 0:
            pass

        self.send_info_event(progress=0.0, message="[Neo4j] Database cleared")

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

            self._run_auto_commit(session, f"CREATE INDEX IF NOT EXISTS FOR (n:{node_type}) ON (n.id)")
            self._run_auto_commit(session, f"CREATE INDEX IF NOT EXISTS FOR (n:{node_type}) ON (n.graph_id)")

        # Indexes are populated in the background: wait until they are ONLINE,
        # otherwise the edge MATCHes run as full label scans.
        session.run("CALL db.awaitIndexes(600)").consume()

    def _upload_edges(self, session: Session, graph : Graph, graph_id : str) -> None:

        for edge_key, edge_list in self.track(graph._edges.items(), task="[EDGES UPLOAD]"):

            if not edge_list:
                continue

            src_type = edge_key.src
            dst_type = edge_key.dst
            rel_type = edge_key.name.replace("-", "_").upper()

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

    APPEARSIN_QUERY = """
        UNWIND $ids AS tid
        MATCH (tr:Triple {id: tid})-[:BELONGSTO]->(a:Article)
        MATCH (tr)-[:HAS_HEAD|HAS_TAIL]->(e:Entity)
        WITH DISTINCT e, a
        MERGE (e)-[:APPEARSIN]->(a)
    """

    HASRELATIONWITH_QUERY = """
        UNWIND $ids AS tid
        MATCH (tr:Triple {id: tid})-[:HAS_HEAD]->(h:Entity),
              (tr)-[:HAS_TAIL]->(t:Entity),
              (tr)-[:HAS_PREDICATE]->(p:Predicate)
        OPTIONAL MATCH (tr)-[:BELONGSTO]->(a:Article)
        WITH h, t, p.name AS pred, collect(DISTINCT toString(a.id)) AS arts
        MERGE (h)-[r:HASRELATIONWITH {predicate: pred}]->(t)
        SET r.articles = coalesce(r.articles, []) +
                         [x IN arts WHERE NOT x IN coalesce(r.articles, [])]
    """

    def _create_missing_relations(self, session : Session, graph_id : str, chunk_size : int = 2000) -> None:

        # Batched from Python: each chunk of triples is its own small write transaction.
        # - memory per transaction stays bounded (no huge collect())
        # - no long-running outer read that its own batches modify (no "Outdated" error)
        # - execute_write retries TransientErrors automatically
        # Both queries are idempotent (MERGE + de-duplicated article list), so a chunk
        # that is retried, or a relationship touched by several chunks, stays correct.

        triple_ids = [
            record["id"]
            for record in session.run(
                "MATCH (tr:Triple {graph_id: $graph_id}) RETURN tr.id AS id",
                graph_id=graph_id,
            )
        ]

        @unit_of_work(timeout=120)
        def run_chunk(tx : Transaction, query : str, ids : List[str]) -> None:
            tx.run(query, ids=ids).consume()

        for name, query in (
            ("APPEARSIN", self.APPEARSIN_QUERY),
            ("HASRELATIONWITH", self.HASRELATIONWITH_QUERY),
        ):
            chunks = Batched(triple_ids, chunk_size)
            chunks = self.track(it=chunks, task=f"[CREATING {name} | {len(triple_ids)} Triples.]")
            for chunk in chunks:
                session.execute_write(run_chunk, query, list(chunk))

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------

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

            # Indexes first: they're maintained during inserts and ready for the edge MATCHes.
            self._create_indexes(session, graph)
            self.send_info_event(progress=0.1, message="[Neo4j] Indexes Created")

            self._clear_db(session, graph_id)
            self.send_info_event(progress=0.2, message="[Neo4j] Database cleared")

            self._upload_nodes(session, graph, graph_id)
            self.send_info_event(progress=0.4, message="[Neo4j] Nodes Uploaded")

            self._upload_edges(session, graph, graph_id)
            self.send_info_event(progress=0.8, message="[Neo4j] Edges Uploaded")

            self._create_missing_relations(session, graph_id)
            self.send_info_event(progress=1.0, message="[Neo4j] Missing Relations Created.")

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