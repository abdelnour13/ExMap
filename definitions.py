import os

ROOT_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(ROOT_DIR, "data")
RESOURCES_DIR = os.path.join(ROOT_DIR, "resources")
MODELS_DIR = os.path.join(RESOURCES_DIR, "models")
FIELDS_DIR = os.path.join(RESOURCES_DIR, "fields")
ENV = os.path.join(ROOT_DIR, ".env")

def hf_model_dir(name : str) -> str:
    return os.path.join(MODELS_DIR, name.replace('/', '--'))

def nlp_dir(name : str) -> str:
    return os.path.join(FIELDS_DIR, name, 'nlp.json')

def domain_dir(name : str) -> str:
    return os.path.join(FIELDS_DIR, name, 'domains.json')