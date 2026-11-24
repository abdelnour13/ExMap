import os
from definitions import API_DIR

SUBDOMAINS_LABEL_KEY = "subdomains"

GLOBAL_LOG_FILE = os.path.join(API_DIR, 'log.jsonl')

JOBS_DIR = os.path.join(API_DIR, 'jobs')
os.makedirs(JOBS_DIR, exist_ok=True)

VIS_DIR = os.path.join(API_DIR, "public", "visualizations")
os.makedirs(VIS_DIR, exist_ok=True)

TEMPLATES_DIR = os.path.join(API_DIR, 'templates')
API_URL = "http://localhost:8000/"