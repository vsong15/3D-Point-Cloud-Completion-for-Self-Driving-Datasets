#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./setup_env.sh --cpu
#   ./setup_env.sh --cuda cu121     # common CUDA build (change as needed)
#
# This script:
#   1) Creates/updates the conda env from environment.yml
#   2) Installs PyTorch and PyG for CPU or CUDA
#   3) Prints final versions

ENV_NAME="pointcloud3d"
CUDA_VARIANT=""   # "cpu" or something like "cu121"

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 --cpu | --cuda cuXXX"
  exit 1
fi

if [[ "$1" == "--cpu" ]]; then
  CUDA_VARIANT="cpu"
elif [[ "$1" == "--cuda" ]]; then
  if [[ $# -lt 2 ]]; then
    echo "Please provide CUDA variant, e.g.: --cuda cu121"
    exit 1
  fi
  CUDA_VARIANT="$2"
else
  echo "Unknown option: $1"
  exit 1
fi

# 1) Create/Update environment
echo "[*] Creating/updating conda environment from environment.yml ..."
conda env update -f environment.yml --prune

# 2) Activate env (works in bash; for zsh use 'conda activate' in a new shell)
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "[*] Python: $(python --version)"
echo "[*] Pip:    $(pip --version)"

# 3) Install PyTorch + PyG
if [[ "$CUDA_VARIANT" == "cpu" ]]; then
  echo "[*] Installing PyTorch (CPU wheels) ..."
  pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

  echo "[*] Installing PyTorch Geometric (CPU wheels) ..."
  # pyg-lib/torch-scatter/torch-sparse/etc. provide prebuilt wheels via pyg.org
  TORCH_VERSION=$(python -c "import torch; print(torch.__version__.split('+')[0])")
  pip install --upgrade \
    pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric \
    -f https://data.pyg.org/whl/torch-${TORCH_VERSION}.html

else
  echo "[*] Installing PyTorch (CUDA wheels: $CUDA_VARIANT) ..."
  # Example for cu121; adjust if you use a different CUDA build available from pytorch.org
  pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/${CUDA_VARIANT}

  echo "[*] Installing PyTorch Geometric (CUDA wheels) ..."
  TORCH_VERSION=$(python -c "import torch; print(torch.__version__.split('+')[0])")
  # Most common case: cu121. If you use a different CUDA build, update the suffix below accordingly.
  PYG_CUDA_SUFFIX="${CUDA_VARIANT}"
  pip install --upgrade \
    pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric \
    -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+${PYG_CUDA_SUFFIX}.html
fi

echo "[*] torch version:        $(python -c 'import torch; print(torch.__version__)')"
echo "[*] torch_geometric ver.: $(python -c 'import torch_geometric; print(torch_geometric.__version__)')"

echo
echo "[*] Done. To activate later:  conda activate ${ENV_NAME}"
echo "[*] Next: python verify_env.py"
