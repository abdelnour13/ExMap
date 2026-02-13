import dotenv
from src.shared.command_tools import Stage, BaseConfig, command
from src.shared.handlers import EventLevel, Event
from src.shared.file_tools import FileReader, FileWriter
from typing import Dict, Any, Tuple, Union, Optional, List, Generator
from neo4j import GraphDatabase, Driver
from pycli import Model
from definitions import ENV
from pydantic import computed_field
from instance import cli

dotenv.load_dotenv(ENV)

class Config(BaseConfig):

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

class Neo4J(Stage):

    def __init__(self, 
        reader : FileReader, 
        writer : FileWriter, 
        batch_size : int, 
        buffer_size : int,
        driver : Driver
    ) -> None:
        super().__init__(reader, writer, batch_size, buffer_size)

        self.entities = {}
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
        query = f"CREATE (e:{type_} {{{props_str}}});"
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
                f"""CREATE (s)-[:{relation} {{ type: {self.format_value(rel)} }}]->(o);"""
            )

        else:

            query = (
                f"""MATCH (s:{type_1} {{ id: {self.format_value(src)} }}) """
                f"""MATCH (o:{type_2} {{ id: {self.format_value(dst)} }}) """
                f"""CREATE (s)-[:{relation}]->(o);"""
            )

        return query
    
    def add_entity(self, entity : str) -> Tuple[int, bool]:

        exists = entity in self.entities

        if not exists:
            self.entities[entity] = len(self.entities)

        return self.entities[entity], exists
    
    def transform(self, doc : Dict) -> List[Dict]:
        
        ### Records To Return
        records = []

        ### Get head, predicate and tail
        head, predicate, tail = doc['head'], doc['predicate'], doc['tail']

        head = head.replace(" ","_")
        predicate = predicate.replace(" ","_")
        tail = tail.replace(" ","_")

        head_id,head_exists = self.add_entity(head)
        tail_id,tail_exists = self.add_entity(tail)

        if not head_exists:

            records.append({
                "id" : head_id,
                "query_type" : "node_isertion",
                "query" : self.node_query(properties={
                    "id" : head_id,
                    "name" : head
                })
            })

        if not tail_exists:

            records.append({
                "id" : tail_id,
                "query_type" : "node_isertion",
                "query" : self.node_query(properties={
                    "id" : tail_id,
                    "name" : tail
                })
            })

        records.append({
            "id": f"{head_id}_{predicate}_{tail_id}",
            "head_id" : head_id,
            "tail_id" : tail_id,
            "predicate" : predicate,
            "query_type" : "relation_insertion",
            "query" : self.relation_query(
                relation=(head_id,predicate,tail_id)
            )
        })

        return records
    
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

        except Exception as e:

            yield Event(
                progress=0.0,
                level=EventLevel.ERROR,
                message="Connection to Neo4j failed."
            )

    def transform_chunk(self, chunk : List[Dict]) -> List[Dict]:

        chunk = super().transform_chunk(chunk)

        with self.driver.session() as session:

            with session.begin_transaction() as trx:
                for record in chunk:
                    trx.run(query=record['query'])

        return chunk

@cli.register_command(
    name="insert2neo4j",
    description=(
        "Extract triples from sentences json file using an LLM-Based triples extractor componenet."
    ),
    model=Model(Config, spread=False),
    add_from_conf=False,
    add_save_to=False
) 
@command
def insert2neo4j(config : Config) -> 'Neo4J':

    driver = GraphDatabase.driver(
        uri=config.neo4j_uri,
        auth=(config.neo4j_user,config.neo4j_pwd)
    )

    return Neo4J(
        reader=FileReader.from_filename(config.input_filename),
        writer=FileWriter.from_filename(config.output_filename),
        buffer_size=config.buffer_size,
        batch_size=config.batch_size,
        driver=driver
    )