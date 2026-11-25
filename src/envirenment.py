import os
from pydantic import BaseModel, BeforeValidator
from typing import Dict, Annotated, List, Optional
from neo4j import GraphDatabase, Driver
from functools import cached_property
from src import helpers
from definitions import ENVIRONMENT, FIELDS_DIR, domain_path
from src.helpers.common import load_json_file

class Domain(BaseModel):
    sub_domain : str
    definition : Optional[str] = None
    domain : Optional[str] = None
    alt_names : Optional[List[str]] = None

class EntityType(BaseModel):
    type : str
    definition : str
    examples : List[str]

class FieldInfo(BaseModel):
    domains : List[Domain]

    @staticmethod
    def load(name : str) -> 'FieldInfo':

        domains_path = domain_path(name)
        domains = [Domain(**domain) for domain in load_json_file(domains_path)]

        return FieldInfo(domains=domains)

class EnvLLMConfig(BaseModel):
    platform : str
    api_key : str

class Neo4jConfig(BaseModel):
    uri : str
    user : str
    password : str

    @cached_property
    def driver(self) -> Driver:

        return GraphDatabase.driver(
            self.uri,
            auth=(self.user, self.password),
        )

NEO4J_LOCAL_CONFIG = Neo4jConfig(
    uri="bolt://neo4j:7687",
    user="neo4j",
    password="password1234"
)

def parse(data : List[Dict]) -> Dict[str, EnvLLMConfig]:

    data = [EnvLLMConfig(**item) for item in data]

    return {
        item.platform : item
        for item in data
    }

class Env(BaseModel):
    hf_token : str | None
    llms : Annotated[Dict[str, EnvLLMConfig], BeforeValidator(parse)]
    neo4j : Dict[str, Neo4jConfig]
    fields : Dict[str, FieldInfo]

    def get_api_key(self, platform : str) -> Optional[str]:

        if platform not in self.llms:
            return None

        return self.llms.get(platform).api_key

    @classmethod
    def load(self) -> 'Env':

        if not os.path.exists(ENVIRONMENT):

            env_data = {
                "hf_token" : None,
                "llms" : {},
                "neo4j" : {
                    "local" : NEO4J_LOCAL_CONFIG.model_dump(exclude_computed_fields=True)
                },
            }

            helpers.save_json_file(ENVIRONMENT, env_data)

        data = helpers.load_json_file(ENVIRONMENT)

        data['fields']  = {
            fname : FieldInfo.load(fname)
            for fname, _ in map(os.path.splitext, os.listdir(FIELDS_DIR))
        }

        return Env(**data)

env = Env.load()