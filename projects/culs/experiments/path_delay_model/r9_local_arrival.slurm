#!/usr/bin/env bash
#SBATCH --job-name=r9-local-arrival
#SBATCH --partition=Compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --export=NONE
set -Eeuo pipefail
# Stage the independent ABC source with pair/mapper patches plus
# r9_local_arrival.patch, these two Python runners and gate3b_analyze.py.
# Submit with --chdir=<unique /data run>; never use the original ABC binary.
RUN_ROOT="$(pwd)"
case "$RUN_ROOT" in /data/wushangge/runs/*) ;; *) exit 2;; esac
export TMPDIR="$RUN_ROOT/tmp"
export XDG_CACHE_HOME="$RUN_ROOT/cache"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
PRIOR_RUN=/data/wushangge/runs/CULS-mapper-internal-20261003-34U6OZ
python3 run_r9_local_frontier.py "$RUN_ROOT"
make -C "$RUN_ROOT/abc" -j4 > "$RUN_ROOT/build.log" 2>&1
AIGREWRITE_R9_LOCAL=1 AIGREWRITE_MAP_DEBUG_NODE=213 "$RUN_ROOT/abc/abc" -c \
 "read $PRIOR_RUN/input/asap7.lib; read $PRIOR_RUN/input/c1355_orig.aig; map; write_blif $RUN_ROOT/baseline_observed.blif; local_r9_arrival $PRIOR_RUN/input/c1355_orig.aig $PRIOR_RUN/input/S002.aig" \
 > "$RUN_ROOT/local_prediction.log"
python3 finalize_r9_local_arrival.py "$RUN_ROOT"
