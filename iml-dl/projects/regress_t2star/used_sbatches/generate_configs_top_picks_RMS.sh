#!/bin/bash

# ── self-locating paths ─────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PROJECT_DIR/../.." && pwd)"
PHIMO_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
cd "$REPO_ROOT"
# ──────────────────────────────────────────────────────────────────────────────


source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cupa_T2_star

# ensure we can import project modules if your script needs it
export PYTHONPATH="$REPO_ROOT"

# disable wandb no matter what
export WANDB_MODE=disabled
export WANDB_DISABLED=true

python $PROJECT_DIR/generate_final_configs_RMS.py \
    --csv $PROJECT_DIR/tissue_combined_results/Cholesky/Acc4/top_runs.csv \
    --top_k 3 \
    --acc_rates 4 \