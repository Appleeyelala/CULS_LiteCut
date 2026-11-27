#!/bin/bash
#SBATCH --job-name=rw4b-b2-905
#SBATCH --nodelist=eda
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -euo pipefail
mkdir -p logs output build

echo "=== Node: $(hostname) | Job: $SLURM_JOB_ID ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

cmake -S src/projects/culs -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 32 --target gpuls
ls -lh build/gpuls

export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
/home/share/process/anaconda3/bin/python -u src/scripts/step8_gpu_struct_full905.py \
    --run "$(pwd)" \
    --gpuls "$(pwd)/build/gpuls" \
    --flags "-r 3 --joint-grad --steps 25 --lr 0.20 --disc-lam 0.30 --fo-inc-lam 0.0 -v 1" \
    --workers 32

echo "DONE step21_stageb_full905"
