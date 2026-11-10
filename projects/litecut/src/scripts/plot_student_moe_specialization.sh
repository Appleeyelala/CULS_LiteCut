#! /bin/bash
export CUDA_VISIBLE_DEVICES=0

# Paths
checkpoint='./checkpoints/extract_cutlib_student_group_fusion/best.pt'
dataset_path='/data/zhoulingfeng/data_cutLib/cutLibData2'
output_dir='./log/student_moe'
bycase_output_dir='./moe_bycase'

# Runtime
device='auto'
batch_size=256
num_workers=8
scope='val'
bycase=true

bycase_flag=""
if [ "$bycase" = "true" ]; then
    bycase_flag="--bycase --bycase_output_dir $bycase_output_dir"
fi

mkdir -p "$output_dir"

nohup env PYTHONPATH=src python3 -u src/entry_script/plot_student_moe_specialization.py \
    --checkpoint $checkpoint \
    --dataset_path $dataset_path \
    --output_dir $output_dir \
    --scope $scope \
    --device $device \
    --batch_size $batch_size \
    --num_workers $num_workers \
    $bycase_flag \
    > $output_dir/plot_moe.log 2>&1 &
