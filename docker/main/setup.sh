#!/bin/bash
# GPU setup: FAISS GPU + PyTorch 2.6.0 (CUDA 11.8) + PyG
# requirements.txt and install_resources.sh are run by the Dockerfile, not here.
set -euo pipefail

ENV_NAME=text2kg

# --------------------------------------------------
# FAISS GPU (conda) — FIRST, and WITHOUT the conda `pytorch` package
# (conda's pytorch channel stops at 2.5.1 and would downgrade torch)
# --------------------------------------------------
conda install -n "$ENV_NAME" -y \
    faiss-gpu=1.8.0 \
    pytorch-cuda=11.8 \
    -c pytorch -c nvidia -c conda-forge

# --------------------------------------------------
# PyTorch 2.6.0 + CUDA 11.8 (pip), all pinned together
# --------------------------------------------------
python -m pip install \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu118

# Lock torch versions for every later pip install
python -m pip freeze | grep -E '^(torch|torchvision|torchaudio)==' > /tmp/torch-constraints.txt

# --------------------------------------------------
# PyTorch Geometric
# --------------------------------------------------
python -m pip install torch_geometric -c /tmp/torch-constraints.txt
python -m pip install \
    pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.6.0+cu118.html \
    -c /tmp/torch-constraints.txt

# --------------------------------------------------
# Extras: PyCli, FastAPI, Hugging Face CLI
# --------------------------------------------------
python -m pip install -c /tmp/torch-constraints.txt \
    "git+https://github.com/abdelnour13/pycli.git@f7091f08fbb4dd201eb57a419cca5573bedd5660" \
    "fastapi[standard]" \
    "huggingface_hub[cli]"