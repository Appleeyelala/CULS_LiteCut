#!/usr/bin/env bash
# Follow-on shared-region endpoint audit, submitted only after single calibration passes.
#SBATCH --job-name=culs-impact-paths
#SBATCH --partition=Compute
#SBATCH --output=/data/wushangge/runs/CULS-path-impact-calibration-20260929-v1/logs/multipath-%j.out
#SBATCH --error=/data/wushangge/runs/CULS-path-impact-calibration-20260929-v1/logs/multipath-%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --chdir=/data/wushangge/runs/CULS-path-impact-calibration-20260929-v1
#SBATCH --export=NONE

set -Eeuo pipefail
RUN_ROOT=/data/wushangge/runs/CULS-path-impact-calibration-20260929-v1
ABC_RUN=/data/wushangge/runs/CULS-gate3f-20260922-gate2babc
export TMPDIR="$RUN_ROOT/tmp"
export OMP_NUM_THREADS=1
mkdir -p "$TMPDIR"

if [[ ! -e "$RUN_ROOT/output/results/calibration_run_summary.json" ]]; then
  echo "Single-candidate calibration has not completed; refusing multipath stage" >&2
  exit 2
fi

/home/share/process/anaconda3/bin/python3 "$RUN_ROOT/src/run_calibration.py" \
  --mode multipath \
  --aig "$RUN_ROOT/input/c1355_orig.aig" \
  --region-map "$RUN_ROOT/input/node_to_strict_mffc_region.csv" \
  --gpuls /data/wushangge/runs/CULS-path-delay-model-20260929-v3/bin/gpuls \
  --abc "$ABC_RUN/bin/abc" \
  --library "$ABC_RUN/input/asap7.lib" \
  --path-catalog "$RUN_ROOT/input/path_catalog.csv" \
  --delta-csv "$RUN_ROOT/input/litecut_candidate_delta.csv" \
  --single-csv "$RUN_ROOT/output/results/single_candidate_calibration.csv" \
  --out-dir "$RUN_ROOT/output" \
  --epsilon 0.01 --timing-precision 0.1 --seed 20260929
