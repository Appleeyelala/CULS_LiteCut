#!/usr/bin/env bash
# Full single-candidate calibration on the fixed c1355 PO11 endpoint.
# All replay, mapping, timing, logs, and outputs are under /data/wushangge.
#SBATCH --job-name=culs-impact-cal
#SBATCH --partition=Compute
#SBATCH --output=/data/wushangge/runs/CULS-path-impact-calibration-20260929-v1/logs/slurm-%j.out
#SBATCH --error=/data/wushangge/runs/CULS-path-impact-calibration-20260929-v1/logs/slurm-%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --chdir=/data/wushangge/runs/CULS-path-impact-calibration-20260929-v1
#SBATCH --export=NONE

set -Eeuo pipefail
RUN_ROOT=/data/wushangge/runs/CULS-path-impact-calibration-20260929-v1
PRIOR_RUN=/data/wushangge/runs/CULS-path-delay-model-20260929-v3
ABC_RUN=/data/wushangge/runs/CULS-gate3f-20260922-gate2babc

if [[ -e "$RUN_ROOT/output/results/calibration_run_summary.json" ]]; then
  echo "Refusing to overwrite completed calibration results in $RUN_ROOT/output" >&2
  exit 2
fi

export TMPDIR="$RUN_ROOT/tmp"
export OMP_NUM_THREADS=1
mkdir -p "$TMPDIR" "$RUN_ROOT/output"

/home/share/process/anaconda3/bin/python3 "$RUN_ROOT/src/run_calibration.py" \
  --mode single \
  --aig "$RUN_ROOT/input/c1355_orig.aig" \
  --region-map "$RUN_ROOT/input/node_to_strict_mffc_region.csv" \
  --gpuls "$PRIOR_RUN/bin/gpuls" \
  --abc "$ABC_RUN/bin/abc" \
  --library "$ABC_RUN/input/asap7.lib" \
  --path-catalog "$RUN_ROOT/input/path_catalog.csv" \
  --delta-csv "$RUN_ROOT/input/litecut_candidate_delta.csv" \
  --pilot-csv "$RUN_ROOT/input/single_candidate_validation.csv" \
  --pilot-replay-root "$PRIOR_RUN/output/work/replay" \
  --out-dir "$RUN_ROOT/output" \
  --pi-node 4 --po 11 --expected-baseline 194.4 \
  --epsilon 0.01 --timing-precision 0.1 --seed 20260929
