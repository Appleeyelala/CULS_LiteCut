#!/bin/bash
set -euo pipefail

dataset_path='/home/zhoulingfeng/data/open-sourceDataset'
abc_bin='./abc/abc'
lib_path='./lib/asap7.lib'
picks_csv='./docs/main_table_picks.csv'
output_csv='./log/resyn2_picks.csv'

PYTHONPATH=src python3 -u src/validate/measure_resyn2_picks.py \
    --dataset_path "$dataset_path" \
    --abc_bin "$abc_bin" \
    --lib_path "$lib_path" \
    --picks_csv "$picks_csv" \
    --output_csv "$output_csv"
