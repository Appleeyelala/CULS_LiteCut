#!/usr/bin/env bash
#SBATCH --job-name=gate1b-target
#SBATCH --output=/data/wushangge/gate1a_AIGRewrite/gate1b_out/target-slurm-%j.out
#SBATCH --error=/data/wushangge/gate1a_AIGRewrite/gate1b_out/target-slurm-%j.err
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
mkdir -p "$OUT" "$BUILD"
cmake --build "$BUILD" --target abcg -j2
export GATE1B_AUDIT="$OUT/dft_root80078_alignment.json"
export GATE1B_DESIGN=dft_orig GATE1B_ROOT=80078
unset GATE1A_ROOT GATE1A_CUT GATE1A_SUB GATE1A_PROVENANCE
LD_LIBRARY_PATH="$BUILD" "$BUILD/abcg" -c "gread $INPUT; grw" > "$OUT/dft_root80078_audit.log" 2>&1

jq -s '{design:"dft_orig",root:80078,cut_slot:0,
  alignment:(.[0].records[]|select(.root==80078 and .cut_slot==0)),
  candidates:[.[1],.[2]]}' \
  "$OUT/dft_root80078_alignment.json" "$OUT/sub0_replay.json" "$OUT/sub2_replay.json" \
  > "$OUT/gate1b_alignment.json"
