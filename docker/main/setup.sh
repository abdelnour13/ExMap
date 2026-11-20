#!/bin/bash
set -euo pipefail

ENV_NAME=text2kg   # must be the env that `python` below points to

# --------------------------------------------------
# FAISS GPU (conda) — FIRST, and WITHOUT pytorch
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
# General requirements
# --------------------------------------------------
python -m pip install --no-cache-dir -r requirements.txt -c /tmp/torch-constraints.txt
python -m pip install --no-deps --force-reinstall numpy==2.2.6 pandas==2.2.3

# --------------------------------------------------
# Extras
# --------------------------------------------------
python -m pip install -c /tmp/torch-constraints.txt \
    "git+https://github.com/abdelnour13/pycli.git@f7091f08fbb4dd201eb57a419cca5573bedd5660" \
    "fastapi[standard]"

curl -LsSf https://hf.co/cli/install.sh | bash

bash ./install_resources.sh

# --------------------------------------------------
# Sanity checks (report, don't kill the build)
# --------------------------------------------------
python -m pip check || echo "WARNING: pip check reported conflicts (see above)"
python -c "import torch, torchvision, faiss; print(torch.__version__, torchvision.__version__, faiss.__version__, torch.cuda.is_available())"