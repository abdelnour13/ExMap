import time
import subprocess
from typing import List
from pycli import Example
from instance import cli
from src.commands.pipeline import Pipeline
from src.commands.download import DataDownloader
from src.commands.setup import Setup
from src.commands.feature_extraction import FeatureExtractor
from src.commands.train_v2 import Trainer

@cli.register_command(
    name="start_rpc",
    description="Start the rpc servers.",
    examples=[
        Example(
            args={ "servers" : ["openie6"] },
            description="Starts the openie6 rpc server."
        ),
        Example(
            args={ "servers" : ["openie6", "allennlp"] },
            description="Starts the openie6 and allennlp rpc servers."
        ),
        Example(
            args={ "servers" : ["all"] },
            description="Starts all servers."
        ),
    ],
    add_from_conf=False,
    add_save_to=False
)
def start_rpc(servers : List[str]):
    
    available_servers = ["allennlp", "openie6"]
    processes : list[subprocess.Popen] = []

    def cleanup():

        for process in processes:
            print(f"Stopping process {process.pid}.")
            process.terminate()

    if "all" in servers:
        servers = available_servers

    try:

        for server in servers:

            try:

                process = subprocess.Popen(
                    args = ["conda", "run", "-n", f"text2kg-{server}", "python3", "-m", f"src.rpc.servers.{server}"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1
                )

            except:
                cleanup()
                break
            
            print(f"Started server {server}...")
            processes.append(process)

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        cleanup()

    
def main():
    # cli.scan_dir(os.path.join(".", "src", "commands"), pattern_or_predicate="(.+).py")
    cli.run()

if __name__ == "__main__":
    main()