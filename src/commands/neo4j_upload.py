import os
from torch import Tensor
from typing import List, Self, TypeVar
from pydantic import BaseModel, computed_field
from neo4j import Session, Transaction, Driver
from instance import cli
from definitions import DATA_DIR
from src.envirenment import env
from src.commands.feature_extraction.graph_utils import Graph, Edge, Node
from src.shared.command_tools import BaseCommand, add_handlers, ConnectionOption, OutputDirOption
from src.shared.handlers import ConsoleLogger

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

    def __init__(self, config: Config, driver : Driver) -> None:
        super().__init__()
        self.config = config
        self.driver = driver

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _batches(items: List[T], size: int):
        for i in range(0, len(items), size):
            yield items[i : i + size]

    def _run_batched(self, session : Session, query: str, items: List[Edge] | List[Node], label: str, graph_id : str) -> None:

        total = len(items)
        uploaded = 0

        def run_tx(tx : Transaction, batch):
            tx.run(query, batch=batch, timeout=120)

        for batch in self._batches(items, self.config.batch_size):

            for i in range(len(batch)):

                batch[i] = {
                    'graph_id' : graph_id,
                    **(
                        {
                            "src" : f"{graph_id}_{batch[i].src}",
                            "dst" : f"{graph_id}_{batch[i].dst}"
                        } if isinstance(batch[i], Edge) else {}
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

    def _upload_nodes(self, session : Session, graph : Graph, graph_id : str) -> None:

        for node_type, node_list in graph._nodes.items():

            if not node_list or node_type in self.IGNORED_NODE_TYPES:
                continue

            self.send_info_event(
                progress=0.0,
                message=f"[Upload] Uploading {len(node_list._nodes):,} '{node_type}' nodes …",
            )

            query = (
                f"UNWIND $batch AS node "
                f"CREATE (n:{node_type}) "
                f"SET n = node"
            )

            self._run_batched(session, query, node_list.nodes, f"{node_type} nodes", graph_id)

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

        for edge_key, edge_list in graph._edges.items():

            if not edge_list:
                continue

            src_type = edge_key.src
            dst_type = edge_key.dst
            rel_type = edge_key.name.upper()
            rel_type = rel_type.replace("-", "_").upper()

            if src_type in self.IGNORED_NODE_TYPES or dst_type in self.IGNORED_NODE_TYPES:
                continue

            self.send_info_event(
                progress=0.0,
                message=(
                    f"[Upload] Uploading {len(edge_list._edges):,} "
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

            self._run_batched(session, query, edge_list._edges, f"{edge_key} edges", graph_id)

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------

    def __iter__(self):

        self.send_info_event(
            progress=0.0,
            message=f"[Neo4j] Loading graph from {self.config.expertise_dir}",
        )

        graph = Graph.load(self.config.expertise_dir)

        n_nodes = sum(graph.node_counts.values())
        n_edges = sum(graph.edge_counts.values())

        self.send_info_event(
            progress=0.0,
            message=f"[Neo4j] Graph loaded — {n_nodes:,} nodes, {n_edges:,} edges",
        )

        graph_id = next(iter(graph.get_nodes('ExpertiseMap').nodes_generator)).id

        with self.driver.session() as session:
            self._clear_db(session)
            self._upload_nodes(session, graph, graph_id)
            self._create_indexes(session, graph)
            self._upload_edges(session, graph, graph_id)

        self.driver.close()

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

        # uploader.add_handler(ConsoleLogger("Neo4jUploader"))

        return uploader