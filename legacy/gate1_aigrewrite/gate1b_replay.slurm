#!/usr/bin/env bash
#SBATCH --job-name=gate1b-replay
#SBATCH --output=/data/wushangge/gate1a_AIGRewrite/gate1b_out/replay-slurm-%j.out
#SBATCH --error=/data/wushangge/gate1a_AIGRewrite/gate1b_out/replay-slurm-%j.err
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --chdir=/data/wushangge/gate1a_AIGRewrite

set -Eeuo pipefail
RUN_ROOT=/data/wushangge/gate1a_AIGRewrite
OUT="$RUN_ROOT/gate1b_out"
BUILD="$RUN_ROOT/build_gate1b"
INPUT=/data/public/AIGRewrite/test_designs/dft_orig.aig
ABC=/data/public/AIGRewrite/abc/abc
cmake --build "$BUILD" --target abcg -j2
for sub in 0 2; do
  unset GATE1B_AUDIT GATE1B_DESIGN GATE1B_ROOT
  export GATE1A_ROOT=80078 GATE1A_CUT=0 GATE1A_SUB="$sub"
  export GATE1A_PROVENANCE="$OUT/sub${sub}_replay.json"
  output="$OUT/root80078_cut0_sub${sub}.aig"
  LD_LIBRARY_PATH="$BUILD" "$BUILD/abcg" -c "gread $INPUT; grw; gwrite $output" > "$OUT/sub${sub}_replay.log" 2>&1
  "$ABC" -c "cec $INPUT $output" > "$OUT/sub${sub}_cec.log" 2>&1
  grep -q "Networks are equivalent" "$OUT/sub${sub}_cec.log"
done
jq -s '{design:"dft_orig",root:80078,cut_slot:0,
  alignment:(.[0].records[]|select(.root==80078 and .cut_slot==0)),candidates:[.[1],.[2]]}' \
  "$OUT/dft_root80078_alignment.json" "$OUT/sub0_replay.json" "$OUT/sub2_replay.json" > "$OUT/gate1b_alignment.json"
