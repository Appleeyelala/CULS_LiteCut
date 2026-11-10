#!/usr/bin/env bash
#SBATCH --job-name=mapper-internal
#SBATCH --partition=Compute
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --export=NONE
set -Eeuo pipefail
RUN_ROOT="$(pwd)"
export TMPDIR="$RUN_ROOT/tmp"
export XDG_CACHE_HOME="$RUN_ROOT/cache"
export OMP_NUM_THREADS=8
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
make -C "$RUN_ROOT/abc" -j8
cp "$RUN_ROOT/abc/abc" "$RUN_ROOT/bin/abc_pair_timing"
python3 "$RUN_ROOT/run_mapper_internal.py" "$RUN_ROOT"
