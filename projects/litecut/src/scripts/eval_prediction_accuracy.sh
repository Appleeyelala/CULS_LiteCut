#!/bin/bash
set -euo pipefail

dataset_path='/data/zhoulingfeng/data_cutLib/cutLibData2'
picks_csv='./docs/main_table_picks.csv'
teacher_ckpt='./checkpoints/extract_cutlib_teacher/best.pt'
student_ckpt='./checkpoints/extract_cutlib_student_group_fusion/best.pt'
output_csv='./log/prediction_accuracy_picks.csv'

student_nodistill_ckpt='./results/ablation_main_table/E-1_nodistill/checkpoints/best.pt'

if [ -n "$student_nodistill_ckpt" ] && [ -f "$student_nodistill_ckpt" ]; then
    PYTHONPATH=src python3 -u src/validate/eval_prediction_accuracy.py \
        --dataset_path "$dataset_path" \
        --picks_csv "$picks_csv" \
        --teacher_ckpt "$teacher_ckpt" \
        --student_ckpt "$student_ckpt" \
        --student_nodistill_ckpt "$student_nodistill_ckpt" \
        --output_csv "$output_csv"
else
    PYTHONPATH=src python3 -u src/validate/eval_prediction_accuracy.py \
        --dataset_path "$dataset_path" \
        --picks_csv "$picks_csv" \
        --teacher_ckpt "$teacher_ckpt" \
        --student_ckpt "$student_ckpt" \
        --output_csv "$output_csv"
fi
