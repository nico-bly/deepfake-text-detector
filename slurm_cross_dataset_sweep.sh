#!/bin/bash
#SBATCH --job-name=cross_dataset_qwen
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=P100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

set -euo pipefail

PYTHON="$HOME/miniconda3/envs/env_esa/bin/python"
if [ ! -x "$PYTHON" ]; then
  echo "Python not found at: $PYTHON" >&2
  exit 1
fi

echo "Node: $(hostname)"
"$PYTHON" --version

# Example: delete saved models after Mercor evaluation to save space
DELETE_AFTER_EVAL=1 \
PYTHON="$PYTHON" \
MEM_EFF="--memory_efficient" \
DEVICE=cuda:0 \
bash scripts/run_cross_dataset_sweep.sh
