import yaml
import json
import os
import wget
import subprocess
import dotenv
from src.shared.command_tools import BaseCommand
from src.shared.handlers import Event, ConsoleLogger
from typing import Optional, List, Generator
from pydantic import BaseModel
from instance import cli
from definitions import hf_model_dir, EXTERNAL_DIR, PLUGINS_DIR, ROOT_DIR, SCRIPTS_DIR, ENV
from huggingface_hub import snapshot_download
from pycli import Field

class Service(BaseModel):
    add : bool
    port : Optional[int] = None
    enable_gpu_access : bool = False

class Docker(BaseModel):
    neo4j : Service
    allennlp : Service
    openie6 : Service

class Manifest(BaseModel):
    docker : Docker
    models : List[str]
    neo4j_plugins : List[str]
    external_repos : List[str]

class Config(BaseModel):
    build : bool = Field(default=False, description="whether to run 'docker compose up --build' or not.")
    create_envs : bool = Field(default=False, description="whether to create conda environments or not.")
    download_models : bool = Field(default=True, description="whether to download huggingface models or not.")
    download_neo4j_plugins : bool = Field(default=True, description="whether to download neo4j plugins or not.")

@cli.register_stage(
    name="setup",
    description=(
        "used to setup the project based on 'manifest.json' file."
    ),
    add_help=True,
    add_from_conf=False,
    add_save_to=False
)
class Setup(BaseCommand):

    def __init__(self, config : Config) -> bool:
        super().__init__()

        self.config = config
        
        with open(os.path.join(ROOT_DIR, 'manifest.json'), 'r') as f:
            self.manifest = Manifest(**json.load(f))

    def dockerize(self) -> Generator[Event, None, None]:

        config = self.manifest.docker

        ### GPU capabilities
        deploy = {
            "resources" : {
                "reservations" :  {
                    "devices" : [
                        {
                            "driver" : "nvidia",
                            "count" : "all",
                            "capabilities" : ["gpu"]
                        }
                    ]
                }
            }
        }

        services = {}

        docker_compose = {
            "version" : "3.8",
            "services" : services
        }

        ### AllenNLP Service
        if config.allennlp.add:

            allennlp_service = {
                "build" : "./docker/allennlp",
                "ports" : [f"{config.allennlp.port}:{config.allennlp.port}"],
                "volumes" : [".:/allennlp"]
            }

            if config.allennlp.enable_gpu_access:
                allennlp_service["deploy"] = deploy

            services["allennlp"] = allennlp_service

        ### Openie6 Service
        if config.openie6:

            openie6_service = {
                "build" : "./docker/openie6",
                "ports" : [f"{config.openie6.port}:{config.openie6.port}"],
                "volumes" : [".:/openie6"],
            }

            if config.openie6.enable_gpu_access:
                openie6_service["deploy"] = deploy

            services["openie6"] = openie6_service

        ### Neo4j
        if config.neo4j.add:

            services['neo4j'] = {
                "image" : "neo4j:latest",
                "container_name" : "text2kg-neo4j",
                "ports" : [
                    "7474:7474",
                    "7687:7687"
                ],
                "volumes" : [
                    "./.neo4j/data:/data",
                    "./.neo4j/logs:/logs",
                    "./.neo4j/conf:/conf",
                    "./.neo4j/import:/var/lib/neo4j/import",
                    "./.neo4j/plugins:/plugins"
                ],
                "environment" : ["NEO4J_AUTH=neo4j/Abdelnour2003"],
                "restart" : "always"
            }

        ### Main
        text2kg_service = {
            "build" : "./docker/text2kg",
            "volumes" : [".:/text2kg"],
            "depends_on" : list(services.keys())
        }

        services["text2kg"] = text2kg_service

        ### Save docker compose
        with open(os.path.join(ROOT_DIR, 'docker-compose.yaml'), 'w') as file:
            yaml.dump(docker_compose, file, sort_keys=False, indent=2)

        yield self.make_info_event(
            progress=None,
            message="'docker-compose.yaml' file generated.",
            extra={}
        )

        if self.config.build:

            yield self.make_info_event(
                progress=None,
                message="running 'docker compose up --build'...",
                extra={}
            )

            subprocess.run(
                args=["docker", "compose", "up", "--build"],
                shell=True,
                cwd=ROOT_DIR
            )

    def download_hf_models(self) -> Generator[Event, None, None]:

        dotenv.load_dotenv(ENV)

        hf_token = dotenv.get_key(ENV, "HF_TOKEN")

        if hf_token is None:

            yield self.make_warn_event(
                progress=None,
                message='HF_TOKEN env variable is not set.',
                extra={}
            )

        yield self.make_info_event(
            progress=None,
            message=f'{len(self.manifest.models)} models to download...',
            extra={}
        )

        for model in self.manifest.models:

            yield self.make_info_event(
                progress=None,
                message=f'Starting downloading huggingface model : {model}...',
                extra={}
            )
            
            if not os.path.exists(hf_model_dir(model)):
                
                snapshot_download(
                    repo_id=model,
                    local_dir=hf_model_dir(model),
                    local_dir_use_symlinks=False,  # Set to False to actually download files instead of creating symlinks
                    resume_download=True,
                    token=hf_token
                )

    def download_external_repos(self):

        yield self.make_info_event(
            progress=None,
            message=f'{len(self.manifest.external_repos)} repos to download...',
            extra={}
        )

        for repo_url in self.manifest.external_repos:

            yield self.make_info_event(
                progress=None,
                message=f'Starting cloning github repo : {repo_url} ...',
                extra={}
            )
            
            name = repo_url.split('/')[-1].split('.')[0]
            path = os.path.join(EXTERNAL_DIR, name)
  
            if not os.path.exists(path):

                subprocess.run(
                    args=["git", "clone", repo_url],
                    shell=True,
                    cwd=EXTERNAL_DIR
                )

    def download_neo4j_plugins(self):

        yield self.make_info_event(
            progress=None,
            message=f'{len(self.manifest.neo4j_plugins)} plugins to download...',
            extra={}
        )

        for plugin in self.manifest.neo4j_plugins:

            yield self.make_info_event(
                progress=None,
                message=f'Starting downloading plugin : {plugin} ...',
                extra={}
            )

            plugin_name = plugin.split('/')[-1]

            if not os.path.exists(os.path.join(PLUGINS_DIR, plugin_name)):
                wget.download(plugin, PLUGINS_DIR)

    def create_envs(self) -> Generator[Event, None, None]:

        config = self.manifest.docker

        ### AllenNLP
        if config.allennlp.add:

            yield self.make_info_event(
                progress=None,
                message=f'setting up allennlp environment...',
                extra={}
            )

            subprocess.run(
                ["bash", "create-allennlp-env.sh"],
                cwd=SCRIPTS_DIR,
                check=True,
                text=True,
                capture_output=True,
            )

        ### Openi6
        if config.allennlp.add:

            yield self.make_info_event(
                progress=None,
                message=f'setting up openie6 environment...',
                extra={}
            )

            subprocess.run(
                ["bash", "create-openie6-env.sh"],
                cwd=SCRIPTS_DIR,
                check=True,
                text=True,
                capture_output=True,
            )

        ### Text2KG
        yield self.make_info_event(
            progress=None,
            message=f'setting up main environment...',
            extra={}
        )
        
        subprocess.run(
            ["bash", "create-text2kg-env.sh"],
            cwd=SCRIPTS_DIR,
            check=True,
            text=True,
            capture_output=True,
        )

    def __iter__(self) -> Generator[Event, None, None]:

        yield from self.dockerize()

        if self.config.create_envs:
            yield from self.create_envs()

        if self.config.download_models:
            yield from self.download_hf_models()

        yield from self.download_external_repos()

        if self.config.download_neo4j_plugins:
            yield from self.download_neo4j_plugins()

    @classmethod
    def from_config(cls, config : Config) -> 'Setup':
        return Setup(config=config) \
            .add_handler(ConsoleLogger(cls.__name__))