import os

ROOT_DIR = os.path.dirname(__file__)

DATA_DIR = os.path.join(ROOT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

CORPUS_DIR = os.path.join(DATA_DIR, "corpus")
os.makedirs(CORPUS_DIR, exist_ok=True)

KGS_DIR = os.path.join(DATA_DIR, "knowledge_graphs")
os.makedirs(KGS_DIR, exist_ok=True)

FEATURES_DIR = os.path.join(DATA_DIR, "features")
os.makedirs(FEATURES_DIR, exist_ok=True)

EMS_DIR = os.path.join(DATA_DIR, "expertise_maps")
os.makedirs(EMS_DIR, exist_ok=True)

EVAL_DIR = os.path.join(DATA_DIR, "eval")
os.makedirs(EVAL_DIR, exist_ok=True)


SCRIPTS_DIR = os.path.join(ROOT_DIR, "scripts")
EXTERNAL_DIR = os.path.join(ROOT_DIR, 'external')
EXPERIMENTS_DIR = os.path.join(ROOT_DIR, 'experiments')
RESOURCES_DIR = os.path.join(ROOT_DIR, "resources")
NEO4J_DIR = os.path.join(ROOT_DIR, '.neo4j')

MODELS_DIR = os.path.join(RESOURCES_DIR, "models")
FIELDS_DIR = os.path.join(RESOURCES_DIR, "fields")
PLUGINS_DIR = os.path.join(NEO4J_DIR, "plugins")

def rel_path(path : str) -> str:
    return os.path.relpath(path, ROOT_DIR)

ENV = os.path.join(ROOT_DIR, ".env")

def hf_model_dir(name : str) -> str:
    return os.path.join(MODELS_DIR, name.replace('/', '--'))

def nlp_dir(name : str) -> str:
    return os.path.join(FIELDS_DIR, name, 'nlp.json')

def domain_dir(name : str) -> str:
    return os.path.join(FIELDS_DIR, name, 'domains.json')