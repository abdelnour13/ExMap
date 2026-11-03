import os
from src.commands.data_download import main
from src.commands.knowledge_graph_construction import main
from src.commands.feature_extraction import main
from src.commands.graph_learning import main
# from src.commands.build_emap import main
from src.commands import build_expertise_map, train, neo4j_upload, feature_extraction_old, evaluate
from instance import cli

def main():
    commands_dir = os.path.join(".", "src", "commands")
    cli.run()

if __name__ == "__main__":
    main()