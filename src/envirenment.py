from pydantic import BaseModel, BeforeValidator
from typing import Dict, Annotated, List, Optional
from neo4j import GraphDatabase, Driver
from functools import cached_property
from src import helpers
from definitions import ENVIRONMENT

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

def parse(data : List[Dict]) -> Dict[str, EnvLLMConfig]:

    data = [EnvLLMConfig(**item) for item in data]

    return {
        item.platform : item
        for item in data
    }

class Env(BaseModel):
    hf_token : str
    llms : Annotated[Dict[str, EnvLLMConfig], BeforeValidator(parse)]
    neo4j : Dict[str, Neo4jConfig]

    def get_api_key(self, platform : str) -> Optional[str]:

        if platform not in self.llms:
            return None

        return self.llms.get(platform).api_key

    @classmethod
    def load(self) -> 'Env':
        return Env(**helpers.load_json_file(ENVIRONMENT))

env = Env.load()