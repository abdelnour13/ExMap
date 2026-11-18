import os

ROOT_DIR = os.path.dirname(__file__)

EXPERIMENTS_DIR = os.path.join(ROOT_DIR, 'experiments')

DATA_DIR = os.path.join(ROOT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

CORPUS_DIR = os.path.join(DATA_DIR, "corpus")
os.makedirs(CORPUS_DIR, exist_ok=True)

ANNOTATION_DIR = os.path.join(DATA_DIR, "annotation")
os.makedirs(ANNOTATION_DIR, exist_ok=True)

KGS_DIR = os.path.join(DATA_DIR, "knowledge_graphs")
os.makedirs(KGS_DIR, exist_ok=True)

FEATURES_DIR = os.path.join(DATA_DIR, "features")
os.makedirs(FEATURES_DIR, exist_ok=True)

EMS_DIR = os.path.join(DATA_DIR, "expertise_maps")
os.makedirs(EMS_DIR, exist_ok=True)

EVAL_DIR = os.path.join(DATA_DIR, "eval")
os.makedirs(EVAL_DIR, exist_ok=True)

ENVIRONMENT = os.path.join(ROOT_DIR, "env.json")
MANIFEST = os.path.join(ROOT_DIR, "manifest.json")
SCRIPTS_DIR = os.path.join(ROOT_DIR, "scripts")

RESOURCES_DIR = os.path.join(ROOT_DIR, "resources")
MODELS_DIR = os.path.join(RESOURCES_DIR, "models")
FIELDS_DIR = os.path.join(RESOURCES_DIR, "fields")

NEO4J_DIR = os.path.join(ROOT_DIR, '.neo4j')
PLUGINS_DIR = os.path.join(NEO4J_DIR, "plugins")

def rel_path(path : str) -> str:
    return os.path.relpath(path, ROOT_DIR)

def hf_model_dir(name : str) -> str:
    return os.path.join(MODELS_DIR, name.replace('/', '--'))

def domain_dir(name : str) -> str:
    return os.path.join(FIELDS_DIR, f'{name}.json')