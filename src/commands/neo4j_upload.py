import os
import dotenv
from typing import Any, Dict, List, Self, Optional
from pydantic import BaseModel, computed_field
from neo4j import GraphDatabase, Session, Transaction
from definitions import ENV
from instance import cli
from src import helpers
from src.shared.command_tools import BaseCommand
from src.shared.handlers import ConsoleLogger

dotenv.load_dotenv(ENV)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class Config(BaseModel):

    graph_filename: str
    batch_size: int = 500

    @computed_field
    @property
    def neo4j_uri(self) -> str:
        return dotenv.get_key(ENV, "NEO4J_URI")

    @computed_field
    @property
    def neo4j_user(self) -> str:
        return dotenv.get_key(ENV, "NEO4J_USER")

    @computed_field
    @property
    def neo4j_pwd(self) -> str:
        return dotenv.get_key(ENV, "NEO4J_PWD")


# ---------------------------------------------------------------------------
# Uploader
# ---------------------------------------------------------------------------

@cli.register_stage(
    name="neo4j_upload",
    description="Uploads a graph JSON file (feature_extraction / expertise_map format) to Neo4j.",
    add_from_conf=False,
    add_save_to=False,
)
class Neo4jUploader(BaseCommand):

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _batches(items: List[Any], size: int):
        for i in range(0, len(items), size):
            yield items[i : i + size]

    def _run_batched(self, session : Session, query: str, items: List[Any], label: str, graph_id : str) -> None:

        total = len(items)
        uploaded = 0

        def run_tx(tx : Transaction, batch):
            tx.run(query, batch=batch)

        for batch in self._batches(items, self.config.batch_size):

            for i in range(len(batch)):

                batch[i]['graph_id'] = graph_id
                batch[i]['id'] = f"{graph_id}_{batch[i]['id']}"

                if 'src' in batch[i]:
                    batch[i]['src'] = f"{graph_id}_{batch[i]['src']}"
                    batch[i]['dst'] = f"{graph_id}_{batch[i]['dst']}"


            session.execute_write(run_tx, batch)
            uploaded += len(batch)

            self.send_debug_event(
                progress=uploaded / total,
                message=f"[Upload] {label}: {uploaded:,} / {total:,}",
            )

    # ------------------------------------------------------------------
    # Upload steps
    # ------------------------------------------------------------------

    def _clear_db(self, session : Session) -> None:
        self.send_info_event(progress=0.0, message="[Neo4j] Clearing all nodes and relationships …")
        session.run("MATCH (n) DETACH DELETE n")
        self.send_info_event(progress=0.0, message="[Neo4j] Database cleared")

    IGNORED_NODE_TYPES = {"Author", "Keyword"}

    def _upload_nodes(self, session : Session, nodes_by_type: Dict[str, List[Dict]], graph_id : str) -> None:

        for node_type, node_list in nodes_by_type.items():

            if not node_list or node_type in self.IGNORED_NODE_TYPES:
                continue

            self.send_info_event(
                progress=0.0,
                message=f"[Upload] Uploading {len(node_list):,} '{node_type}' nodes …",
            )

            query = (
                f"UNWIND $batch AS node "
                f"CREATE (n:{node_type}) "
                f"SET n = node"
            )

            self._run_batched(session, query, node_list, f"{node_type} nodes", graph_id)

    def _create_indexes(self, session : Session, nodes_by_type: Dict[str, List[Dict]]) -> None:

        self.send_info_event(progress=0.0, message="[Neo4j] Creating indexes on node ids …")

        for node_type in nodes_by_type.keys():

            if node_type in self.IGNORED_NODE_TYPES:
                continue

            query = f"CREATE INDEX IF NOT EXISTS FOR (n:{node_type}) ON (n.id)"
            session.run(query)

            query = f"CREATE INDEX IF NOT EXISTS FOR (n:{node_type}) ON (n.graph_id)"
            session.run(query)

    def _upload_edges(self, session: Session, edges_by_type: Dict[str, List[Dict]], graph_id : str) -> None:

        for edge_key, edge_list in edges_by_type.items():

            if not edge_list:
                continue

            parts    = edge_key.split("_")
            src_type = parts[0]
            dst_type = parts[-1]
            rel_type = "_".join(parts[1:-1]).upper()

            if src_type in self.IGNORED_NODE_TYPES or dst_type in self.IGNORED_NODE_TYPES:
                continue

            self.send_info_event(
                progress=0.0,
                message=(
                    f"[Upload] Uploading {len(edge_list):,} "
                    f"(:{src_type})-[:{rel_type}]->(:{dst_type}) edges …"
                ),
            )

            query = (
                f"UNWIND $batch AS edge "
                f"MATCH (src:{src_type} {{id: edge.src}}) "
                f"MATCH (dst:{dst_type} {{id: edge.dst}}) "
                f"CREATE (src)-[r:{rel_type}]->(dst) "
                f"SET r = edge"
            )

            self._run_batched(session, query, edge_list, f"{edge_key} edges", graph_id)

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------

    def __iter__(self):

        self.send_info_event(
            progress=0.0,
            message=f"[Neo4j] Loading graph from {self.config.graph_filename}",
        )

        raw_G: Dict[str, Any] = helpers.load_json_file(self.config.graph_filename)

        nodes_by_type: Dict[str, List[Dict]] = raw_G.get("nodes", {})
        edges_by_type: Dict[str, List[Dict]] = raw_G.get("edges", {})

        n_nodes = sum(len(v) for v in nodes_by_type.values())
        n_edges = sum(len(v) for v in edges_by_type.values())

        self.send_info_event(
            progress=0.0,
            message=f"[Neo4j] Graph loaded — {n_nodes:,} nodes, {n_edges:,} edges",
        )

        driver = GraphDatabase.driver(
            self.config.neo4j_uri,
            auth=(self.config.neo4j_user, self.config.neo4j_pwd),
        )

        graph_id = nodes_by_type['ExpertiseMap'][0]['id']

        with driver.session() as session:
            self._clear_db(session)
            self._upload_nodes(session, nodes_by_type, graph_id)
            self._create_indexes(session, nodes_by_type)
            self._upload_edges(session, edges_by_type, graph_id)

        driver.close()

        self.send_info_event(
            progress=1.0,
            message="[Neo4j] Upload completed successfully",
        )

        yield from []

    @classmethod
    def from_config(cls, config: Config) -> Self:
        uploader = cls(config=config)
        uploader.add_handler(ConsoleLogger("Neo4jUploader"))
        return uploader