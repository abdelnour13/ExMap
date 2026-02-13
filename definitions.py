import os

ROOT_DIR = os.path.dirname(__file__)
RESOURCES_DIR = os.path.join(ROOT_DIR, "resources")
MODELS_DIR = os.path.join(RESOURCES_DIR, "models")
ENV = os.path.join(ROOT_DIR, ".env")

def hf_model_dir(name : str) -> str:
    return os.path.join(MODELS_DIR, name.replace('/', '--'))