#!/bin/bash
set -euo pipefail

checkpoint='./checkpoints/extract_cutlib_student_group_fusion/best.pt'
dataset_path='/data/zhoulingfeng/data_cutLib/cutLibData2'
export_path='./log/global_bucket_specs.json'

PYTHONPATH=src python3 -u src/entry_script/plot_student_moe_specialization.py \
    --checkpoint "$checkpoint" \
    --dataset_path "$dataset_path" \
    --output_dir ./log \
    --export_buckets "$export_path" \
    --num_workers 8
