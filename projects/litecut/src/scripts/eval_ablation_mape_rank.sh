#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export PYTHONPATH="$PROJECT_ROOT/src"

python "$PROJECT_ROOT/src/validate/eval_ablation_mape_rank.py" \
    --dataset_path /data/zhoulingfeng/data_cutLib/cutLibData2 \
    --picks_csv "$PROJECT_ROOT/docs/main_table_picks.csv" \
    --result_root "$PROJECT_ROOT/results/ablation_main_table" \
    --experiments E0_empty E1_context8 E2_distill_context8 E3_distill_context8_rank \
    --output_csv "$PROJECT_ROOT/results/ablation_main_table/eval_mape_rank.csv" \
    --device cpu
