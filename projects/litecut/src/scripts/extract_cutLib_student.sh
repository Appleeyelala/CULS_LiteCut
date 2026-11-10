#! /bin/bash
export CUDA_VISIBLE_DEVICES=0

# Paths
dataset_path='/data/zhoulingfeng/data_cutLib/cutLibData2'
teacher_checkpoint='./checkpoints/extract_cutlib_teacher/best.pt'
result_root='./results/ablation_main_table'
experiment_name='E-1_nodistill'
checkpoint_dir=''
resume_checkpoint=''
stderr_log=''
prepared_cache='/data/zhoulingfeng/data_cutLib/prepared_cache_student'
log_path=''

# Runtime
device='auto'
random_seed=42
cpu_worker=16
teacher_batch_size=1024
label_outlier_filter='mad' # none,zscore,mad,maximum
label_outlier_threshold=10.0
with_context8=true
with_distill=false
with_rank_loss=true
with_moe=true

# Student Model: group encoder + soft fusion + expert soft MoE regression
shared_hidden_dim=64
teacher_pls_dim=48
num_experts=4
gate_temperature=1.0

# Distillation Loss
lambda_hard=1.0
lambda_soft=0.8
lambda_align=0.8
rank_loss_weight=0.3

# Training
train_ratio=0.9
batch_size=128
eval_batch_size=128
num_epochs=10
learning_rate=8e-4
weight_decay=1e-4
scheduler_t_max=10
scheduler_t_mult=2
scheduler_eta_min=1e-6
dataloader_num_workers=8
log_every=500
save_every=1

context_flag="--with_context8"
if [ "$with_context8" != "true" ]; then
    context_flag="--no-with_context8"
fi

distill_flag="--with_distill"
if [ "$with_distill" != "true" ]; then
    distill_flag="--no-with_distill"
fi

rank_flag="--with_rank_loss"
if [ "$with_rank_loss" != "true" ]; then
    rank_flag="--no-with_rank_loss"
fi

moe_flag="--with_moe"
if [ "$with_moe" != "true" ]; then
    moe_flag="--no-with_moe"
fi

resume_flag=""
if [ -n "$resume_checkpoint" ]; then
    resume_flag="--resume_checkpoint $resume_checkpoint"
fi

stderr_log="$result_root/$experiment_name/logs/stderr.log"
mkdir -p "$(dirname "$stderr_log")"

nohup env PYTHONPATH=src python3 -u src/entry_script/extract_cutLib_student.py \
    --dataset_path $dataset_path \
    --experiment_name $experiment_name \
    --result_root $result_root \
    --teacher_checkpoint $teacher_checkpoint \
    --teacher_batch_size $teacher_batch_size \
    --prepared_cache_dir $prepared_cache \
    --reuse_prepared_cache \
    \
    --device $device \
    --random_seed $random_seed \
    --cpu_worker $cpu_worker \
    --label_outlier_filter $label_outlier_filter \
    --label_outlier_threshold $label_outlier_threshold \
    $context_flag \
    $distill_flag \
    $rank_flag \
    $moe_flag \
    \
    --shared_hidden_dim $shared_hidden_dim \
    --teacher_pls_dim $teacher_pls_dim \
    --num_experts $num_experts \
    --gate_temperature $gate_temperature \
    \
    --lambda_hard $lambda_hard \
    --lambda_soft $lambda_soft \
    --lambda_align $lambda_align \
    --rank_loss_weight $rank_loss_weight \
    \
    --train_ratio $train_ratio \
    --batch_size $batch_size \
    --eval_batch_size $eval_batch_size \
    --num_epochs $num_epochs \
    --learning_rate $learning_rate \
    --weight_decay $weight_decay \
    --scheduler_t_max $scheduler_t_max \
    --scheduler_t_mult $scheduler_t_mult \
    --scheduler_eta_min $scheduler_eta_min \
    --dataloader_num_workers $dataloader_num_workers \
    --log_every $log_every \
    --save_every $save_every \
    $resume_flag \
    > $stderr_log 2>&1 &
