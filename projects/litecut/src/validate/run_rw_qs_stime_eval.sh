#! /bin/bash
set -euo pipefail

# Paths
dataset_path='/home/zhoulingfeng/data/open-sourceDataset'
abc_bin='./abc/abc'
lib_path='./lib/asap7.lib'
pred_dir='./log/rw_qs_stime_pred'
summary_path='./log/rw_qs_stime_rw_summary.csv'
log_path='./log/rw_qs_stime_rw_eval.log'
jobs=16
flush_every=8

env PYTHONPATH=src nohup python3 -u src/validate/run_rw_qs_stime_eval.py \
    --dataset_path "$dataset_path" \
    --abc_bin "$abc_bin" \
    --all \
    --lib_path "$lib_path" \
    --pred_dir "$pred_dir" \
    --summary_path "$summary_path" \
    --log_path "$log_path" \
    --jobs "$jobs" \
    --flush-every "$flush_every" \
    > ./log/rw_eval.log 2>&1 &
