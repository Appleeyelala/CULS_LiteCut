#!/bin/bash
#SBATCH --job-name=rw4b-b2-unlk
#SBATCH --nodelist=eda
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -euo pipefail
mkdir -p logs output build

echo "=== Node: $(hostname) | Job: $SLURM_JOB_ID ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

cmake -S src/projects/culs -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 16 --target gpuls
ls -lh build/gpuls

export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
/home/share/process/anaconda3/bin/python -u src/scripts/step22_stageb_unlock_verify.py \
    --run "$(pwd)" \
    --gpuls "$(pwd)/build/gpuls" \
    --workers 16
