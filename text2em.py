import os
from instance import cli
from src.commands import download, text2kg, train, build_expertise_map, neo4j_upload, train_ssl

def main():

    commands_dir = os.path.join(".", "src", "commands")

    def predicate(path : str) -> bool:
        return os.path.dirname(path) == commands_dir and os.path.basename(path) not in ["train.py"]
    
    # cli.scan_dir(commands_dir, pattern_or_predicate=predicate)
    cli.run()

if __name__ == "__main__":
    main()