#!/usr/bin/env bash
# Bounded c1355 single-candidate pilot; all work/results remain on /data.
#SBATCH --job-name=culs-path-pilot
#SBATCH --partition=Compute
#SBATCH --output=/data/wushangge/runs/CULS-path-delay-model-20260929-v3/logs/slurm-%j.out
#SBATCH --error=/data/wushangge/runs/CULS-path-delay-model-20260929-v3/logs/slurm-%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --chdir=/data/wushangge/runs/CULS-path-delay-model-20260929-v3
#SBATCH --export=NONE

set -Eeuo pipefail
RUN_ROOT=/data/wushangge/runs/CULS-path-delay-model-20260929-v3
if [[ -e "$RUN_ROOT/output/run_manifest.json" ]]; then
  echo "Refusing to overwrite existing run output: $RUN_ROOT/output" >&2
  exit 2
fi
export TMPDIR="$RUN_ROOT/tmp"
export OMP_NUM_THREADS=1
mkdir -p "$TMPDIR" "$RUN_ROOT/output"

/home/share/process/anaconda3/bin/python3 "$RUN_ROOT/src/run_pilot.py" \
  --aig "$RUN_ROOT/input/c1355_orig.aig" \
  --gate3a "$RUN_ROOT/input/gate3a" \
  --gate3b "$RUN_ROOT/input/gate3b" \
  --gate3c "$RUN_ROOT/input/gate3c" \
  --gate3e "$RUN_ROOT/input/gate3e" \
  --cutlib "$RUN_ROOT/cutlib" \
  --checkpoint "$RUN_ROOT/cutlib/best.pt" \
  --rewrite-library "$RUN_ROOT/input/rewrite_library.inc" \
  --region-map "$RUN_ROOT/input/node_to_strict_mffc_region.csv" \
  --gpuls "$RUN_ROOT/bin/gpuls" \
  --abc "$RUN_ROOT/bin/abc" \
  --library "$RUN_ROOT/input/asap7.lib" \
  --tools-dir "$RUN_ROOT/src" \
  --out-dir "$RUN_ROOT/output" \
  --max-candidates 15 \
  --min-candidates 10 \
  --timing-precision 0.01
