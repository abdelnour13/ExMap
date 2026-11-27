import os
from pathlib import Path
from pydantic import BaseModel

ROOT_DIR = os.path.dirname(__file__)
ROOT_DIR_ENVVAR_NAME = 'DATA_DIR_ROOT'
env_var_set = ROOT_DIR_ENVVAR_NAME in os.environ

class Dirs(BaseModel):
    root : Path

    ### NEO4J:
    @property
    def neo4j_dir(self) -> Path:
        return self.root / '.neo4j'

    @property
    def neo4j_plugins_dir(self) -> Path:
        return self.neo4j_dir / 'plugins'

    ### RESOURCES:
    @property
    def resources_dir(self) -> Path:
        return self.root / 'resources'

    @property
    def fields_dir(self) -> Path:
        return self.resources_dir / 'fields'

    @property
    def models_dir(self) -> Path:
        return self.root / 'models'

    ### DATA DIR:
    @property
    def data_dir(self) -> Path:
        return Path(os.environ[ROOT_DIR_ENVVAR_NAME]) if env_var_set else (self.root / 'data')

    @property
    def corpus_dir(self) -> Path:
        return self.data_dir / 'corpus'

    @property
    def kgs_dir(self) -> Path:
        return self.data_dir / 'knowledge_graphs'

    @property
    def features_dir(self) -> Path:
        return self.data_dir / 'features'

    @property
    def experiments_dir(self) -> Path:
        return self.data_dir / 'experiments'

    @property
    def maps_dir(self) -> Path:
        return self.data_dir / 'maps'

    @property
    def eval_dir(self) -> Path:
        return self.data_dir / 'eval'

    @property
    def neo4j_logs_dir(self) -> Path:
        return self.data_dir / 'neo4j_logs'

dirs = Dirs()