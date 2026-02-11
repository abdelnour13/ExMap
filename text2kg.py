import os
import yaml
import tomllib
import subprocess
from pydantic import BaseModel
from instance import cli

class RPCService(BaseModel):
    add : bool
    port : int
    enable_gpu_access : bool

class Config(BaseModel):
    allennlp : RPCService
    openie6 : RPCService

    @classmethod
    def load(cls) -> 'Config':
        with open('config.toml', 'rb') as f:
            return Config(**tomllib.load(f))

@cli.register_command(
    name="dockerize",
    description="Generate the 'docker-compose.yml' file according to the 'config.toml' file."
)
def dockerize():

    config = Config.load()

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

    ### Main
    text2kg_service = {
        "build" : "./docker/text2kg",
        "volumes" : [".:/text2kg"],
        "depends_on" : list(services.keys())
    }

    services["text2kg"] = text2kg_service

    ### Save docker compose
    with open('docker-compose.yaml', 'w') as file:
        yaml.dump(docker_compose, file, sort_keys=False, indent=2)


@cli.register_command(
    name="create-envs",
    description="Creates the conda envirenments according to the 'config.toml' project."
)
def create_envs() -> None:

    config = Config.load()

    ### AllenNLP
    if config.allennlp.add:

        subprocess.run(
            ["bash", "create-allennlp-env.sh"],
            cwd="scripts",
            check=True,
            text=True,
            capture_output=True,
        )

    ### Openi6
    if config.allennlp.add:

        subprocess.run(
            ["bash", "create-openie6-env.sh"],
            cwd="scripts",
            check=True,
            text=True,
            capture_output=True,
        )

    ### Text2KG
    subprocess.run(
        ["bash", "create-text2kg-env.sh"],
        cwd="scripts",
        check=True,
        text=True,
        capture_output=True,
    )
    
def main():
    cli.scan_dir(os.path.join(".", "src", "commands"), pattern_or_predicate="(.+).py")
    cli.run()

if __name__ == "__main__":
    main()