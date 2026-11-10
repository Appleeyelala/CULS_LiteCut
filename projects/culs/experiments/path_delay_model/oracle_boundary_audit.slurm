#!/usr/bin/env bash
#SBATCH --job-name=oracle-boundary
#SBATCH --partition=Compute
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --export=NONE

set -Eeuo pipefail
RUN_ROOT="$(pwd)"
export TMPDIR="$RUN_ROOT/tmp"
export XDG_CACHE_HOME="$RUN_ROOT/cache"
export ABC_DUMP_AIGER_MAP=1
export ABC_DUMP_ORIG=1
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
"$RUN_ROOT/bin/abc_pair_timing" -c "read \"$RUN_ROOT/input/asap7.lib\"; read \"$RUN_ROOT/input/c1355_orig.aig\"; map; print_pair_delay po11 pi03; topo; print_pair_delay po11 pi03"
"$RUN_ROOT/bin/abc_pair_timing" -c "read \"$RUN_ROOT/input/asap7.lib\"; read \"$RUN_ROOT/input/S002.aig\"; map; print_pair_delay po11 pi03; topo; print_pair_delay po11 pi03"
