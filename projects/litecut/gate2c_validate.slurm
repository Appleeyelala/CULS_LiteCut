#!/usr/bin/env bash
#SBATCH --job-name=gate2c-smoke
#SBATCH --output=/data/wushangge/gate2c_runs/gate2c_validation_20260917/logs/slurm-%j.out
#SBATCH --error=/data/wushangge/gate2c_runs/gate2c_validation_20260917/logs/slurm-%j.err
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --chdir=/data/wushangge/gate2c_runs/gate2c_validation_20260917
#SBATCH --export=NONE

set -Eeuo pipefail

RUN_ROOT=/data/wushangge/gate2c_runs/gate2c_validation_20260917
export TMPDIR="$RUN_ROOT/tmp"
export XDG_CACHE_HOME="$RUN_ROOT/cache"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$RUN_ROOT/logs"

exec /home/share/process/anaconda3/bin/python "$RUN_ROOT/src/gate2c_runner.py" \
  --manifest "$RUN_ROOT/input/sampling_manifest.json" \
  --run-root "$RUN_ROOT" \
  --original "$RUN_ROOT/input/normalized_original.aig" \
  --rewrite-library "$RUN_ROOT/input/rewrite_library.inc" \
  --timing-library "$RUN_ROOT/input/asap7.lib" \
  --checkpoint "$RUN_ROOT/input/best.pt" \
  --gate1-engine "$RUN_ROOT/bin/abcg" \
  --abc /data/public/AIGRewrite/abc/abc \
  --litecut-repo "$RUN_ROOT/litecut_repo" \
  --decision-limit 1
