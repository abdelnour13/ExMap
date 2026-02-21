import dotenv
from src.shared.command_tools import Stage, StageConfig, add_handlers
from src.shared.handlers import EventLevel, Event
from src.shared.file_tools import FileReader, FileWriter, TextfileCache
from typing import Dict, Any, Tuple, Union, Optional, List, Generator
from neo4j import GraphDatabase, Driver
from definitions import ENV
from pydantic import computed_field
from instance import cli

dotenv.load_dotenv(ENV)

class Config(StageConfig):

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

@cli.register_stage(
    name="insert2neo4j",
    description=(
        "Inserts the graph into a neo4j database using the provided credentials in the '.env' file."
    ),
    add_from_conf=False,
    add_save_to=False
) 
@add_handlers
class Neo4J(Stage):

    def __init__(self, 
        reader : FileReader, 
        writer : FileWriter, 
        driver : Driver
    ) -> None:
        super().__init__(reader, writer)

        self.entities = {}
        self.documents = {}
        self.driver = driver

    def format_value(self, value : Any) -> Any:

        if isinstance(value, str):
            return f'"{value}"'
        elif isinstance(value, bool):
            return "true" if value else "false"
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, (list, set)):
            return "[" + ", ".join(self.format_value(v) for v in value) + "]"
        elif value is None:
            return "null"
        else:
            raise TypeError(f"Unsupported property type: {type(value)} for value {value}")

    def node_query(self, properties: dict, type_ : str = "Entity") -> str:
        props_str = ", ".join(f"{k}: {self.format_value(v)}" for k, v in properties.items())
        query = f"MERGE (e:{type_} {{{props_str}}});"
        return query

    def relation_query(self,
        relation : Tuple[Union[int, str], Optional[str], Union[int, str]],
        types : Tuple[str, str, str] = ("Entity","Predicate","Entity")
    ) -> str:

        src, rel, dst = relation
        type_1, relation, type_2 = types

        if rel is not None:

            query = (
                f"""MATCH (s:{type_1} {{ id: {self.format_value(src)} }})"""
                f"""MATCH (o:{type_2} {{ id: {self.format_value(dst)} }})"""
                f"""MERGE (s)-[:{relation} {{ type: {self.format_value(rel)} }}]->(o);"""
            )

        else:

            query = (
                f"""MATCH (s:{type_1} {{ id: {self.format_value(src)} }}) """
                f"""MATCH (o:{type_2} {{ id: {self.format_value(dst)} }}) """
                f"""MERGE (s)-[:{relation}]->(o);"""
            )

        return query
    
    def transform(self, doc : Dict) -> List[str]:
        
        n1 = doc["n1"]
        n2 = doc["n2"]
        relation = doc["relation"]

        return [
            self.node_query(
                properties=n1["proprties"],
                type_=n1["type"]
            ),
            self.node_query(
                properties=n2["proprties"],
                type_=n2["type"]
            ),
            self.relation_query(
                relation=(n1["proprties"]["id"],relation['properties'].get("value", None),n2["proprties"]["id"]),
                types=(n1['type'], relation['type'],n2['type'])
            )
        ]
    
    def before_start(self) -> Generator[Event,None,None]:

        try:

            self.driver.verify_connectivity()

            yield Event(
                progress=0.0,
                level=EventLevel.DEBUG,
                message="Connection to Neo4j established successfully."
            )

            with self.driver.session() as session:

                session.run("""
                    MATCH(n)
                    CALL {
                        WITH n
                        DETACH DELETE n
                    } in TRANSACTIONS OF 100 ROWS
                """)

                yield Event(
                    progress=0.0,
                    level=EventLevel.DEBUG,
                    message="Deleted all rows in the database."
                )

                session.run("""
                    CREATE CONSTRAINT unique_entity_id IF NOT EXISTS
                    FOR (e:Entity)
                    REQUIRE e.id IS UNIQUE;
                """)

                session.run("""
                    CREATE CONSTRAINT unique_article_id IF NOT EXISTS
                    FOR (a:Article)
                    REQUIRE a.id IS UNIQUE;
                """)

                yield Event(
                    progress=0.0,
                    level=EventLevel.DEBUG,
                    message="Unique constraints created."
                )

        except Exception as e:

            yield Event(
                progress=0.0,
                level=EventLevel.ERROR,
                message="Connection to Neo4j failed."
            )

    def transform_chunk(self, chunk : List[str]) -> List[str]:

        chunk = super().transform_chunk(chunk)

        with self.driver.session() as session:

            with session.begin_transaction() as trx:
                for query in chunk:
                    trx.run(query=query)

        return chunk

    @classmethod
    def from_config(cls, config : Config) -> 'Neo4J':

        cache = TextfileCache(config.checkpoint_filename)

        driver = GraphDatabase.driver(
            uri=config.neo4j_uri,
            auth=(config.neo4j_user,config.neo4j_pwd)
        )

        return Neo4J(
            reader=FileReader
                .from_filename(config.input_filename, cache)
                .batched(config.batch_size),
            writer=FileWriter
                .from_filename(config.output_filename)
                .buffered(config.buffer_size, cache),
            driver=driver
        )