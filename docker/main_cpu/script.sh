#!/bin/bash
set -euo pipefail

# --------------------------------------------------
# PyTorch 2.6.0 (CPU), all pinned together
# --------------------------------------------------
python -m pip install \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Lock torch versions for every later pip install
python -m pip freeze | grep -E '^(torch|torchvision|torchaudio)==' > /tmp/torch-constraints.txt

# --------------------------------------------------
# FAISS CPU (pip is fine for the CPU build)
# --------------------------------------------------
python -m pip install faiss-cpu -c /tmp/torch-constraints.txt

# --------------------------------------------------
# PyTorch Geometric (CPU wheels)
# --------------------------------------------------
python -m pip install torch_geometric -c /tmp/torch-constraints.txt
python -m pip install \
    pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.6.0+cpu.html \
    -c /tmp/torch-constraints.txt

# --------------------------------------------------
# Extras
# --------------------------------------------------
python -m pip install -c /tmp/torch-constraints.txt \
    "git+https://github.com/abdelnour13/pycli.git@f7091f08fbb4dd201eb57a419cca5573bedd5660" \
    "fastapi[standard]" \
    "huggingface_hub[cli]"

python -c "import torch, torchvision, faiss; print(torch.__version__, torchvision.__version__, faiss.__version__)"