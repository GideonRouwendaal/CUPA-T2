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

python $PROJECT_DIR/generate_qResults_plot.py \
    --config $PROJECT_DIR/configs/regress/example_test_config.yaml \
    --puq_ckpt       $REPO_ROOT/results/regression/<PATH_TO_PUQ_CKPT> \
    --hetero_ckpt    $REPO_ROOT/results/regression/<PATH_TO_HET_CKPT> \
    --cupa_chol_ckpt $REPO_ROOT/results/regression/<PATH_TO_CUPA_CKPT> \
    --subject sub-22 \
    --slice_num 16 \
    --out $REPO_ROOT/results/comparison.png