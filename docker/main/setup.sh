#!/bin/bash
set -euo pipefail

# --------------------------------------------------
# PyTorch + CUDA 11.8 + FAISS GPU
# --------------------------------------------------

# Create and activate the environment first:
# conda create -n torch26 python=3.11 -y
# conda activate torch26

# --------------------------------------------------
# PyTorch 2.6.0 + CUDA 11.8
# --------------------------------------------------

python -m pip install \
    torch==2.6.0 \
    torchvision \
    --index-url https://download.pytorch.org/whl/cu118

# --------------------------------------------------
# FAISS GPU
# --------------------------------------------------

# Install FAISS through Conda.
# Do not install faiss-gpu through pip.
conda install -n text2kg -y \
    faiss-gpu=1.8.0 \
    pytorch \
    pytorch-cuda=11.8 \
    -c pytorch \
    -c nvidia \
    -c rapidsai \
    -c conda-forge

# --------------------------------------------------
# PyTorch Geometric
# --------------------------------------------------

python -m pip install torch_geometric

python -m pip install \
    pyg_lib torch_scatter torch_sparse \
    torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.6.0+cu118.html

# --------------------------------------------------
# General requirements
# --------------------------------------------------

# requirements.txt should NOT install or override:
# - torch
# - torchvision
# - faiss-gpu
# - CUDA 12 packages
#
# Avoid conflicting NumPy/pandas pins.

python -m pip install --no-cache-dir -r requirements.txt

python -m pip install --force-reinstall \
    numpy==2.2.6 \
    pandas==2.2.3

# --------------------------------------------------
# Check dependency metadata
# --------------------------------------------------

python -m pip check

# --------------------------------------------------
# PyCli
# -------------------------------------------------

python -m pip install \
    git+https://github.com/abdelnour13/pycli.git@f7091f08fbb4dd201eb57a419cca5573bedd5660

# Hugging Face CLI
curl -LsSf https://hf.co/cli/install.sh | bash

# Fastapi
curl -LsSf https://astral.sh/uv/0.13.5/install.sh | sh
pip install fastapi[standard]

# --------------------------------------------------
# NLP resources (generated from the manifest)
# --------------------------------------------------
bash ./install_resources.sh