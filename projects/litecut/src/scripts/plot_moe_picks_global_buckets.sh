#!/bin/bash
set -euo pipefail

checkpoint='./checkpoints/extract_cutlib_student_group_fusion/best.pt'
dataset_path='/data/zhoulingfeng/data_cutLib/cutLibData2'
bucket_file='./log/global_bucket_specs.json'
output_dir='./moe_global'

# 14 cases from main_table_picks.csv: (split_dir, case_dir)
cases=(
    "save_json_forgeEDA hevc_md_top"
    "save_json_forgeEDA sc_cpu"
    "save_json_forgeEDA cabac_ucontext"
    "save_json_forgeEDA dma_rc_logic"
    "save_json_forgeEDA intra_pred_PE"
    "save_json_ACE b22_comb"
    "save_json_ACE b17_1_comb"
    "save_json_ACE systemcaes_comb"
    "save_json_ACE b18_1_comb"
    "save_json_OpenABC jpeg_orig"
    "save_json_OpenABC ac97_ctrl_orig"
    "save_json_OpenABC c5315_orig"
    "save_json_OpenABC pci_orig"
    "save_json_OpenABC bc0_orig"
)

for entry in "${cases[@]}"; do
    split_dir=$(echo "$entry" | awk '{print $1}')
    case_dir=$(echo "$entry" | awk '{print $2}')
    echo "=== ${split_dir}/${case_dir} ==="
    PYTHONPATH=src python3 -u src/entry_script/plot_student_moe_specialization.py \
        --checkpoint "$checkpoint" \
        --dataset_path "$dataset_path" \
        --output_dir "$output_dir" \
        --single_case \
        --dataset "$split_dir" \
        --case_name "$case_dir" \
        --bucket_file "$bucket_file" \
        --device auto \
        --batch_size 256
done

echo "Done. Output: $output_dir"
