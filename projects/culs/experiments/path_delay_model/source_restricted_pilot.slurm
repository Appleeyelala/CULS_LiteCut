#!/usr/bin/env bash
#SBATCH --job-name=abc-pair-pilot
#SBATCH --partition=Compute
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --export=NONE

set -Eeuo pipefail
RUN_ROOT="$(pwd)"
export TMPDIR="$RUN_ROOT/tmp"
export XDG_CACHE_HOME="$RUN_ROOT/cache"
export OMP_NUM_THREADS=2
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$RUN_ROOT/output"
python3 "$RUN_ROOT/run_source_restricted.py" \
  --abc "$RUN_ROOT/bin/abc_pair_timing" \
  --lib "$RUN_ROOT/input/asap7.lib" \
  --aig "$RUN_ROOT/input/c1355_orig.aig" \
  --catalog "$RUN_ROOT/input/path_catalog.csv" \
  --calibration "$RUN_ROOT/input/single_candidate_calibration.csv" \
  --fixture "$RUN_ROOT/input/disconnected_pairs.blif" \
  --out "$RUN_ROOT/output"
