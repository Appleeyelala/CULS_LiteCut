#!/usr/bin/env bash
#SBATCH --job-name=gate2c-sm86-smoke
#SBATCH --output=/data/wushangge/gate2c_runs/gate2c_sm86_build/logs/smoke-%j.out
#SBATCH --error=/data/wushangge/gate2c_runs/gate2c_sm86_build/logs/smoke-%j.err
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --chdir=/data/wushangge/gate2c_runs/gate2c_sm86_build
#SBATCH --export=ALL

set -Eeuo pipefail

RUN_ROOT=/data/wushangge/gate2c_runs/gate2c_sm86_build
ENGINE="$RUN_ROOT/engine/abcg_sm86"
INPUT=/data/public/AIGRewrite/case/c1355.aig
OUTPUT="$RUN_ROOT/smoke_replayed.aig"
PROVENANCE="$RUN_ROOT/smoke_provenance.json"
LOG="$RUN_ROOT/logs/smoke_engine.log"
CEC_LOG="$RUN_ROOT/logs/smoke_cec.log"

rm -f "$OUTPUT" "$PROVENANCE" "$LOG" "$CEC_LOG"
export GATE1A_ROOT=146
export GATE1A_CUT=0
export GATE1A_SUB=0
export GATE1A_PROVENANCE="$PROVENANCE"
unset GATE1C_SPEC GATE1B_AUDIT GATE1B_DESIGN GATE1B_ROOT
export LD_LIBRARY_PATH="$RUN_ROOT/engine"

"$ENGINE" -c "gread $INPUT; grw; gwrite $OUTPUT" > "$LOG" 2>&1
test -s "$OUTPUT"
test -s "$PROVENANCE"
/data/public/AIGRewrite/abc/abc -c "cec $INPUT $OUTPUT" > "$CEC_LOG" 2>&1
grep -q 'Networks are equivalent' "$CEC_LOG"

sha256sum "$ENGINE" "$RUN_ROOT/engine/libabc.so" "$INPUT" "$OUTPUT" "$PROVENANCE" > "$RUN_ROOT/smoke_sha256.txt"
