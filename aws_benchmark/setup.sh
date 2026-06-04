#!/usr/bin/env bash
set -euo pipefail

#############################################
# Config
#############################################

PROJECT_ROOT="/home/ubuntu/"
SRC_DIR="$PROJECT_ROOT/src"
ENV_NAME="ase"
PYTHON_VERSION="3.11"

# Adjust if needed after checking `nvidia-smi`
# TORCH_CUDA_INDEX="cu132"
TORCH_CUDA_INDEX="cu121"

#############################################
# 1. Install Miniconda (if missing)
#############################################

if ! command -v conda &> /dev/null; then
    echo "[INFO] Conda not found. Installing Miniconda..."

    CONDA_INSTALLER="/tmp/miniconda.sh"
    curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o "$CONDA_INSTALLER"

    bash "$CONDA_INSTALLER" -b -p "$HOME/miniconda3"

    source "$HOME/miniconda3/etc/profile.d/conda.sh"

else
    echo "[INFO] Conda already installed."
    # eval "$(conda shell.bash hook)"
fi

source "$HOME/miniconda3/etc/profile.d/conda.sh"

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

#############################################
# 2. Create directory structure
#############################################

mkdir -p "$SRC_DIR"
cd "$SRC_DIR"

#############################################
# 3. Clone external dependencies
#############################################

if [ ! -d "$SRC_DIR/mace" ]; then
    git clone https://github.com/ACEsuit/mace.git
else
    echo "[INFO] mace already exists"
fi

if [ ! -d "$SRC_DIR/graph_electrostatics" ]; then
    git clone https://github.com/WillBaldwin0/graph_electrostatics.git
else
    echo "[INFO] graph_electrostatics already exists"
fi

#############################################
# 4. Create conda environment
#############################################

ENV_PATH="$HOME/miniconda3/envs/ase"

if [ -d "$ENV_PATH" ]; then
    echo "[INFO] Environment exists"
else
    echo "[INFO] Creating environment"
    conda create -n ase python=3.11 -y
fi

#############################################
# 5. Install PyTorch (CUDA build)
#############################################

conda run -n "$ENV_NAME" pip install --upgrade pip

if [ -n "$TORCH_CUDA_INDEX" ]; then
    echo "[INFO] Installing PyTorch with CUDA ($TORCH_CUDA_INDEX)"
    conda run -n "$ENV_NAME" pip install torch torchvision torchaudio \
        --index-url "https://download.pytorch.org/whl/$TORCH_CUDA_INDEX"
else
    echo "[INFO] Installing CPU-only PyTorch"
    conda run -n "$ENV_NAME" pip install torch torchvision torchaudio
fi

#############################################
# 6. Install local packages
#############################################

conda run -n "$ENV_NAME" pip install -e "$SRC_DIR/mace"
conda run -n "$ENV_NAME" pip install -e "$SRC_DIR/graph_electrostatics"

#############################################
# 7. Install remaining dependencies
#############################################

conda run -n "$ENV_NAME" pip install \
    ase \
    matplotlib \
    scipy \
    numpy \ 
    cuequivariance_torch

#############################################
# 8. Verify installation
#############################################

conda run -n "$ENV_NAME" python - << 'EOF'
from mace.calculators import mace_polar
import torch
import ase

print("mace_polar OK")
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("ase:", ase.__version__)
EOF

echo "[INFO] Setup complete."

source "$HOME/miniconda3/etc/profile.d/conda.sh"