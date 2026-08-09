#!/bin/bash

# ── self-locating paths ─────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
REPO_ROOT="$(cd "$PROJECT_DIR/../.." && pwd)"
PHIMO_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
cd "$REPO_ROOT"
# ──────────────────────────────────────────────────────────────────────────────


source $(conda info --base)/etc/profile.d/conda.sh
conda activate cupa_T2_star


export PYTHONPATH="$REPO_ROOT"

python $PROJECT_DIR/run_group.py \
    --config_path $PROJECT_DIR/configs/Ablations/configs_2
