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

python -u $PROJECT_DIR/regression_evaluator_statistical_tests_final.py \
    --log_dir $PROJECT_DIR/RMS_results/Cholesky/Acc4 \
    --puq_eval_dir  $PROJECT_DIR/evaluations/Final/Acc4/BASELINE_PUQ \
    --hetero_eval_dir $PROJECT_DIR/evaluations/Final/Acc4/BASELINE_Hetero \
    --sweep_eval_dir $PROJECT_DIR/evaluations/Final/Acc4/SWEEP_TOP1 \
    --out_csv $PROJECT_DIR/evaluations/Final/Acc4/wilcoxon_acc4.csv \
    --metric nrmse --metric ssim \
    --agg mean \
    --tissue wm --tissue gm --tissue csf --tissue overall \
    --unit subject \
    --min_subjects 1 