#!/bin/bash
set -e

# PyTorch & TorchVision
pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu118

# PyTorch Geometric + extensions
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.5.0+cu118.html

# General requirements
pip install --no-cache-dir -r requirements.txt

# pycli
pip install git+https://github.com/abdelnour13/pycli.git#egg=pycli

# spaCy model
python -m spacy download en_core_web_sm

# NLTK data
python -c "
import nltk
nltk.download('punkt_tab')
nltk.download('punkt')
# Package name changed in some index versions
try:
    nltk.download('averaged_perceptron_tagger_en', raise_on_error=True)
except Exception:
    nltk.download('averaged_perceptron_tagger')
"

# Hugging Face CLI
curl -LsSf https://hf.co/cli/install.sh | bash