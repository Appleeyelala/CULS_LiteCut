#!/bin/bash
#SBATCH --job-name=paper-4suites
#SBATCH --partition=Compute
#SBATCH --nodelist=eda
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -euo pipefail
mkdir -p logs output

echo "=== Node: $(hostname) | Job: $SLURM_JOB_ID ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

cmake -S src/projects/culs -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 32 --target gpuls
ls -lh build/gpuls

export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
/home/share/process/anaconda3/bin/python -u src/scripts/step23_paper_eval_suite.py \
    --run "$(pwd)" \
    --gpuls "$(pwd)/build/gpuls" \
    --suites "ACE,OpenABC,iwls2022,iwls2024" \
    --include-panel \
    --workers 32
