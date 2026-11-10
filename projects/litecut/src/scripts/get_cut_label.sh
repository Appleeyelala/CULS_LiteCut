#! /bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2

# Paths
dataset_path='/data/zhoulingfeng/data_cutLib/cutLibData'
checkpoint_dir='./checkpoints/extract_cutlib_teacher'
resume_checkpoint=''
stderr_log='./log/cutLib_extraction_error.log'

# Runtime
device='auto'
random_seed=42
include_graph=true
cpu_worker=16
cpu_graph_worker=64
label_outlier_filter='none' # none,zscore,mad,maximum
label_outlier_threshold=1000.0
graph_tensorize_chunk_size=128

# Training
train_ratio=0.9
batch_size=64
eval_batch_size=128
num_epochs=20
learning_rate=1e-3
weight_decay=1e-4
scheduler_t_max=20
scheduler_eta_min=1e-6
dataloader_num_workers=0
log_every=50
save_every=1

# Feature Branch
feature_group_hidden_dim=32
feature_fusion_hidden_dim=128

# Graph Branch
gnn_hidden_dim=64
gnn_num_layers=3
readout_pos_gate_hidden_dim=16
readout_content_hidden_dim=32
p_inv_init='gaussian' # zero or gaussian
p_inv_init_std=0.02

# Regression Head
regression_hidden_dim1=128
regression_hidden_dim2=64
regression_dropout=0.1

graph_flag="--include_graph"
if [ "$include_graph" != "true" ]; then
    graph_flag="--no-include_graph"
fi

resume_flag=""
if [ -n "$resume_checkpoint" ]; then
    resume_flag="--resume_checkpoint $resume_checkpoint"
fi

nohup python3 -u src/entry_script/data_analysis.py \
    --dataset_path $dataset_path \
    --device $device \
    --random_seed $random_seed \
    $graph_flag \
    --label_outlier_filter $label_outlier_filter \
    --label_outlier_threshold $label_outlier_threshold \
    --cpu_worker $cpu_worker \
    --cpu_graph_worker $cpu_graph_worker \
    --train_ratio $train_ratio \
    --batch_size $batch_size \
    --graph_tensorize_chunk_size $graph_tensorize_chunk_size \
    --eval_batch_size $eval_batch_size \
    --num_epochs $num_epochs \
    --learning_rate $learning_rate \
    --weight_decay $weight_decay \
    --scheduler_t_max $scheduler_t_max \
    --scheduler_eta_min $scheduler_eta_min \
    --dataloader_num_workers $dataloader_num_workers \
    --log_every $log_every \
    --save_every $save_every \
    --checkpoint_dir $checkpoint_dir \
    $resume_flag \
    --feature_group_hidden_dim $feature_group_hidden_dim \
    --feature_fusion_hidden_dim $feature_fusion_hidden_dim \
    --gnn_hidden_dim $gnn_hidden_dim \
    --gnn_num_layers $gnn_num_layers \
    --readout_pos_gate_hidden_dim $readout_pos_gate_hidden_dim \
    --readout_content_hidden_dim $readout_content_hidden_dim \
    --p_inv_init $p_inv_init \
    --p_inv_init_std $p_inv_init_std \
    --regression_hidden_dim1 $regression_hidden_dim1 \
    --regression_hidden_dim2 $regression_hidden_dim2 \
    --regression_dropout $regression_dropout \
    > $stderr_log 2>&1 &
