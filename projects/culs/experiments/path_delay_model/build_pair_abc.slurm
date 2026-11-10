#!/usr/bin/env bash
#SBATCH --job-name=abc-pair-build
#SBATCH --partition=Compute
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --export=NONE

set -Eeuo pipefail
RUN_ROOT="$(pwd)"
export TMPDIR="$RUN_ROOT/tmp"
export XDG_CACHE_HOME="$RUN_ROOT/cache"
export OMP_NUM_THREADS=8
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$RUN_ROOT/bin"
cd "$RUN_ROOT/abc"
make -j8
cp abc "$RUN_ROOT/bin/abc_pair_timing"
sha256sum "$RUN_ROOT/bin/abc_pair_timing"
