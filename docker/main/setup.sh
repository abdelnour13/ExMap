#!/bin/bash
set -euo pipefail

# PyTorch + CUDA 11.8
python -m pip install \
    torch==2.6.0 torchvision \
    --index-url https://download.pytorch.org/whl/cu118

# PyTorch Geometric
python -m pip install torch_geometric

python -m pip install \
    pyg_lib torch_scatter torch_sparse \
    torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.6.0+cu118.html

# Numerical stack + GPU FAISS
# Install together through Conda to resolve dependencies.
python -m pip install --no-deps faiss-gpu==1.15.1

python -m pip install --no-deps \
    "nvidia-cuda-runtime-cu12>=12.6,<13" \
    "nvidia-cublas-cu12>=12.6,<13" \
    "nvidia-curand-cu12>=10.3.7,<11"

# General requirements
# requirements.txt must not pin NumPy/pandas to conflicting versions.
python -m pip install --no-cache-dir -r requirements.txt

# Check core imports and dependency metadata
python - <<'PY'
import numpy
import pandas
import torch
import faiss

print("NumPy:", numpy.__version__)
print("pandas:", pandas.__version__)
print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("FAISS:", faiss.__version__)
print("FAISS GPUs:", faiss.get_num_gpus())
PY

python -m pip check

# pycli
python -m pip install \
    git+https://github.com/abdelnour13/pycli.git

# spaCy model
python -m spacy download en_core_web_sm

# NLTK data
python - <<'PY'
import nltk

nltk.download("punkt_tab")
nltk.download("punkt")

try:
    nltk.download("averaged_perceptron_tagger_eng", raise_on_error=True)
except Exception:
    nltk.download("averaged_perceptron_tagger")
PY

# Hugging Face CLI
curl -LsSf https://hf.co/cli/install.sh | bash