from dataclasses import dataclass
from definitions import ENV
from typing import Dict
from src.shared.llms import LLM as LLMInfo
from neo4j import GraphDatabase, Driver
from openai import Client, AsyncClient
from dotenv import dotenv_values

@dataclass
class LLM:
    info : LLMInfo
    client : Client
    async_client : AsyncClient

@dataclass
class Env:

    llms : Dict[str, LLM]
    driver : Driver
    hf_token : str
    cskg_dir : str

    @classmethod
    def load(cls, env : str) -> 'Env':

        raw = dotenv_values(env)

        data = {
            "hf_token" : raw['HF_TOKEN'] ,
            "cskg_dir" : raw['CSKG_DIR'],
            "llms" : {}
        }

        neo4j_connection_info = {}

        for key, value in raw.items():

            if key.endswith('_API_KEY'):

                llm_name = key.replace('_API_KEY', '').lower()
                info = LLMInfo.from_name(llm_name)

                client = Client(
                    api_key=value,
                    base_url=info.url
                )

                async_client = Client(
                    api_key=value,
                    base_url=info.url
                )

                data['llms'][llm_name] = LLM(
                    info=info,
                    client=client,
                    async_client=async_client
                )

            if key.startswith('NEO4J_'):
                key = key.replace('NEO4J_', '')
                neo4j_connection_info[key] = value

        data['driver'] = GraphDatabase.driver(
            neo4j_connection_info['URI'],
            auth=(neo4j_connection_info['USER'], neo4j_connection_info['PWD']),
        )

        return Env(**data)


env = Env.load(ENV)