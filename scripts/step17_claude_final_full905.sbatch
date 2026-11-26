#!/bin/bash
#SBATCH --job-name=rw4b-s2-905
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -euo pipefail
mkdir -p logs output build cap0/output cap16/output

echo "=== Node: $(hostname) | Job: $SLURM_JOB_ID ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

cmake -S src/projects/culs -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 32 --target gpuls
ls -lh build/gpuls

export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

echo "=================================================================="
echo "=== RUN 1: Full 905 circuits: -r 3 --adp-w 2.0 --kappa 2.0 --fo-cap 0 ==="
echo "=================================================================="
/home/share/process/anaconda3/bin/python -u src/scripts/step8_gpu_struct_full905.py \
    --run "$(pwd)/cap0" \
    --gpuls "$(pwd)/build/gpuls" \
    --flags "-r 3 --adp-w 2.0 --kappa 2.0 --fo-cap 0 -v 1" \
    --workers 32

echo "=================================================================="
echo "=== RUN 2: Full 905 circuits: -r 3 --adp-w 2.0 --kappa 2.0 --fo-cap 16 ==="
echo "=================================================================="
/home/share/process/anaconda3/bin/python -u src/scripts/step8_gpu_struct_full905.py \
    --run "$(pwd)/cap16" \
    --gpuls "$(pwd)/build/gpuls" \
    --flags "-r 3 --adp-w 2.0 --kappa 2.0 --fo-cap 16 -v 1" \
    --workers 32

echo "DONE step17_claude_final_full905"
