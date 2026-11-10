#!/usr/bin/env bash
#SBATCH --job-name=gate1cr-c
#SBATCH --output=/data/wushangge/gate1a_AIGRewrite/gate1c_r_out/stage_c-%j.out
#SBATCH --error=/data/wushangge/gate1a_AIGRewrite/gate1c_r_out/stage_c-%j.err
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --chdir=/data/wushangge/gate1a_AIGRewrite
set -Eeuo pipefail
ROOT=/data/wushangge/gate1a_AIGRewrite
OUT="$ROOT/gate1c_r_out"
BUILD="$ROOT/build_gate1c"
ABC=/data/public/AIGRewrite/abc/abc
DFT=/data/public/AIGRewrite/test_designs/dft_orig.aig
C1355=/data/public/AIGRewrite/case/c1355.aig
mkdir -p "$OUT"
cmake --build "$BUILD" --target abcg -j2

run_one() {
  local tag=$1 input=$2 root=$3 cut=$4 sub=$5
  export GATE1A_ROOT=$root GATE1A_CUT=$cut GATE1A_SUB=$sub
  unset GATE1C_SPEC
  export GATE1A_PROVENANCE="$OUT/$tag.json"
  LD_LIBRARY_PATH="$BUILD" "$BUILD/abcg" -c "gread $input; grw; gwrite $OUT/$tag.aig" > "$OUT/$tag.rewrite.log" 2>&1
  "$ABC" -c "cec $input $OUT/$tag.aig" > "$OUT/${tag}_cec.log" 2>&1
  grep -q "Networks are equivalent" "$OUT/${tag}_cec.log"
}

run_one gate1a_sub0_regression "$DFT" 80078 0 0
run_one gate1a_sub2_regression "$DFT" 80078 0 2

# Enumerate all available candidates reported by the Gate 1B audit.
for sub in 0 1; do run_one d1_sub${sub} "$C1355" 146 0 "$sub"; done
for sub in 0 1 2 3 4; do run_one d2_sub${sub} "$C1355" 226 0 "$sub"; done
