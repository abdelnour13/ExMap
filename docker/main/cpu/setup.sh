#!/bin/bash
# CPU setup: FAISS CPU + PyTorch 2.6.0 (CPU) + PyG
set -euo pipefail

ENV_NAME=text2kg

# FAISS CPU (conda) — WITHOUT the conda `pytorch` package.
# faiss-cpu 1.8.0 is compiled against NumPy 1.x -> keep numpy 1.26.4 everywhere
conda install -n "$ENV_NAME" -y \
    faiss-cpu=1.8.0 \
    numpy=1.26.4 \
    -c conda-forge

# PyTorch 2.6.0 (CPU build), all pinned together
python -m pip install \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Lock torch + numpy for every later pip install
python -m pip freeze | grep -E '^(torch|torchvision|torchaudio)==' > /tmp/torch-constraints.txt
echo "numpy==1.26.4" >> /tmp/torch-constraints.txt
cat /tmp/torch-constraints.txt

# PyTorch Geometric
python -m pip install torch_geometric -c /tmp/torch-constraints.txt
python -m pip install \
    pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.6.0+cpu.html \
    -c /tmp/torch-constraints.txt

# Extras
python -m pip install -c /tmp/torch-constraints.txt \
    "git+https://github.com/abdelnour13/pycli.git@f7091f08fbb4dd201eb57a419cca5573bedd5660" \
    "fastapi[standard]" \
    "huggingface_hub[cli]"
