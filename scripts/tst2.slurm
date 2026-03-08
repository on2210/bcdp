#!/bin/bash
#SBATCH --job-name=binding_test
#SBATCH --output=/home/ADV_2526a/omernaziri1/bcdp/slurm/data_test5.out
#SBATCH --error=/home/ADV_2526a/omernaziri1/bcdp/slurm/data_test5.err
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --time=1000
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=50000
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1

set -euo pipefail

# -----------------------------
# Paths
# -----------------------------
WORKDIR="/home/ADV_2526a/omernaziri1"
REPO_DIR="$WORKDIR/bcdp"   # <-- repo root (has scripts/ and bcdp/ package)
CONDA_SH="$WORKDIR/conda/bin/conda"
TOKEN_FILE_ADV="$WORKDIR/.huggingface/token"
TOKEN_FILE_HOME="$HOME/.huggingface/token"

mkdir -p "$REPO_DIR/slurm"

# -----------------------------
# Token (robust)
# -----------------------------
TOKEN_FILE=""
if [[ -f "$TOKEN_FILE_HOME" ]]; then
  TOKEN_FILE="$TOKEN_FILE_HOME"
elif [[ -f "$TOKEN_FILE_ADV" ]]; then
  TOKEN_FILE="$TOKEN_FILE_ADV"
else
  echo "ERROR: HuggingFace token file not found." >&2
  echo "Looked in:" >&2
  echo "  $TOKEN_FILE_HOME" >&2
  echo "  $TOKEN_FILE_ADV" >&2
  exit 1
fi

export HF_TOKEN="$(tr -d '\r\n' < "$TOKEN_FILE")"
if [[ -z "$HF_TOKEN" ]]; then
  echo "ERROR: Token file exists but is empty: $TOKEN_FILE" >&2
  exit 1
fi
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
echo "DEBUG: token_file=$TOKEN_FILE"
echo "DEBUG: token_len=${#HF_TOKEN}"

# -----------------------------
# Cache (writable)
# -----------------------------
if [[ -d "/scratch" ]]; then
  CACHE_ROOT="/scratch/$USER/hf_cache"
else
  CACHE_ROOT="/tmp/$USER/hf_cache"
fi

mkdir -p "$CACHE_ROOT"
export HF_HOME="$CACHE_ROOT"
export HF_HUB_CACHE="$CACHE_ROOT/hub"
export TRANSFORMERS_CACHE="$CACHE_ROOT/transformers"
mkdir -p "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE"
echo "DEBUG: HF_HOME=$HF_HOME"

# -----------------------------
# Activate env
# -----------------------------
cd "$WORKDIR"
eval "$($CONDA_SH shell.bash hook)"
conda activate bcdp

# Optional: prove auth works
python - << 'EOF'
import os
from huggingface_hub import HfApi
tok = os.environ.get("HUGGINGFACE_HUB_TOKEN")
print("PY_DEBUG: token_present", tok is not None, "len", len(tok) if tok else None)
print("PY_DEBUG: whoami", HfApi().whoami(token=tok))
EOF

# -----------------------------
# Run sanity
# -----------------------------
cd "$REPO_DIR"

# Make sure imports work regardless of how python is invoked
export PYTHONPATH="/home/ADV_2526a/omernaziri1/bcdp/src:${PYTHONPATH:-}" 
python3 -m scripts.test_dbcm
