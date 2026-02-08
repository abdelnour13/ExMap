import yaml
from typing import Optional
from pydantic import BaseModel
from pycli import Cli, Parsable, toml_parser

cli = Cli(
    name="Project setup tools.",
    description="auto-generate docker compose"
)

class RPCService(BaseModel):
    port : int
    gpu_access : bool

@cli.register_command(
    name="generate-docker-compose",
    description="Automatically generates docker-compose.yml file."
)
def generate(
    allennlp : Parsable[RPCService, toml_parser],
    openie6 : Optional[Parsable[RPCService, toml_parser]] = None
):

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
    allennlp_service = {
        "build" : "./RPC/allennlp",
        "ports" : [f"{allennlp.port}:{allennlp.port}"],
        "volumes" : ["./RPC/allennlp/server.py:/allennlp/server.py"]
    }

    if allennlp.gpu_access:
        allennlp_service["deploy"] = deploy

    services["allennlp"] = allennlp_service

    ### Openie6 Service
    if openie6 is not None:

        openie6_service = {
            "build" : "./RPC/openie6",
            "ports" : [f"{openie6.port}:{openie6.port}"],
            "volumes" : ["./RPC/openie6/server.py:/openie6/server.py"],
        }

        if openie6.gpu_access:
            openie6_service["deploy"] = deploy

        services["openie6"] = openie6_service

    ### Save docker compose
    with open('docker-compose.yaml', 'w') as file:
        yaml.dump(docker_compose, file, sort_keys=False, indent=2)


if __name__ == "__main__":
    cli.run(name="generate-docker-compose")