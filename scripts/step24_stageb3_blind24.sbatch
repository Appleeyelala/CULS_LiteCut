#!/bin/bash
#SBATCH --job-name=stageb3-blind24
#SBATCH --partition=Compute
#SBATCH --nodelist=eda
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -euo pipefail
mkdir -p logs output

echo "=== Node: $(hostname) | Job: $SLURM_JOB_ID ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

cmake -S src/projects/culs -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 24 --target gpuls
ls -lh build/gpuls

export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
/home/share/process/anaconda3/bin/python -u src/scripts/step24_stageb3_blind24.py \
    --run "$(pwd)" \
    --gpuls "$(pwd)/build/gpuls" \
    --workers 24
