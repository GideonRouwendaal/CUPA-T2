#!/bin/bash


# ── self-locating paths ─────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PROJECT_DIR/../.." && pwd)"
CUPA_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
cd "$REPO_ROOT"
# ──────────────────────────────────────────────────────────────────────────────


source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cupa_T2_star

# ensure we can import project modules if your script needs it
export PYTHONPATH="$REPO_ROOT"

# disable wandb
export WANDB_MODE=disabled
export WANDB_DISABLED=true

python $PROJECT_DIR/regression_evaluator_exp_paths.py \
    --evaluator_py $PROJECT_DIR/regression_evaluator.py \
    --test_data_base $CUPA_ROOT/data/T2_param_data/With_Unc/ \
    --out_root $CUPA_ROOT/evaluations/ \
    --device cuda:0