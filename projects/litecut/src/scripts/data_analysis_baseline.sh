#! /bin/bash

# Paths
dataset_path='/data/zhoulingfeng/data_cutLib/cutLibData2'
output_path='./log/cutlib_baseline_neighbor_analysis.json'
log_path='./log/cutlib_baseline_neighbor_analysis.log'
stderr_log='./log/cutlib_baseline_neighbor_analysis_error.log'

# Data Scope
case_dirs=(
    'save_json_OpenABC'
    'save_json_ACE'
    'save_json_iwls2022'
    'save_json_iwls2024'
    'save_json_forgeEDA'
)

num_workers=16

# # Query Setup
# query_indices=(
#     0
#     10
#     20
# )
query_indices=()
query_count=1200
random_seed=42
neighbor_k=40
metric='cosine' # euclidean or cosine
delay_bins=8
top_count=10
max_samples=''

# Feature Search
standardize=true
include_self=false

standardize_flag=""
if [ "$standardize" = "true" ]; then
    standardize_flag="--standardize"
fi

include_self_flag=""
if [ "$include_self" = "true" ]; then
    include_self_flag="--include_self"
fi

max_samples_flag=""
if [ -n "$max_samples" ]; then
    max_samples_flag="--max_samples $max_samples"
fi

query_indices_flag=""
if [ ${#query_indices[@]} -gt 0 ]; then
    query_indices_flag="--query_indices ${query_indices[*]}"
fi

nohup env PYTHONPATH=src python3 -u src/entry_script/data_analysis_baseline.py \
    --dataset_path $dataset_path \
    --case_dirs "${case_dirs[@]}" \
    --num_workers $num_workers \
    $query_indices_flag \
    --query_count $query_count \
    --random_seed $random_seed \
    --neighbor_k $neighbor_k \
    --metric $metric \
    --delay_bins $delay_bins \
    --top_count $top_count \
    --output_path $output_path \
    $standardize_flag \
    $include_self_flag \
    $max_samples_flag \
    > $log_path 2> $stderr_log &
