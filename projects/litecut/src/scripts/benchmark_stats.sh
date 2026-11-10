#!/bin/bash

dataset_path='/data/zhoulingfeng/data_cutLib/cutLibData2'
train_ratio=0.9
output_csv='./log/benchmark_stats.csv'
output_detail_csv='./log/benchmark_stats_detail.csv'
n_workers=16
label_outlier_filter='mad'
label_outlier_threshold=10.0
forgeeda_sample_size=50
forgeeda_n_bins=10
random_seed=42
log_file='./log/benchmark_stats.log'

mkdir -p ./log

nohup env PYTHONPATH=src python -u src/entry_script/benchmark_stats.py \
    --dataset_path "$dataset_path" \
    --train_ratio "$train_ratio" \
    --output_csv "$output_csv" \
    --output_detail_csv "$output_detail_csv" \
    --n_workers "$n_workers" \
    --label_outlier_filter "$label_outlier_filter" \
    --label_outlier_threshold "$label_outlier_threshold" \
    --forgeeda_sample_size "$forgeeda_sample_size" \
    --forgeeda_n_bins "$forgeeda_n_bins" \
    --random_seed "$random_seed" \
    > "$log_file" 2>&1 &

echo "Started PID $!, log: $log_file"
echo "Follow: tail -f $log_file"
