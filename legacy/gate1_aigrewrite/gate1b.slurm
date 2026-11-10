#!/usr/bin/env bash
#SBATCH --job-name=gate1b-align
#SBATCH --output=/data/wushangge/gate1a_AIGRewrite/gate1b_out/slurm-%j.out
#SBATCH --error=/data/wushangge/gate1a_AIGRewrite/gate1b_out/slurm-%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --chdir=/data/wushangge/gate1a_AIGRewrite

set -Eeuo pipefail
RUN_ROOT=/data/wushangge/gate1a_AIGRewrite
OUT="$RUN_ROOT/gate1b_out"
BUILD="$RUN_ROOT/build_gate1b"
ABC=/data/public/AIGRewrite/abc/abc
mkdir -p "$OUT" "$OUT/tmp" "$BUILD"
export TMPDIR="$OUT/tmp"

cmake -S "$RUN_ROOT" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD" --target abcg -j4

designs=(dft_orig idft_orig vga_lcd_orig)
for design in "${designs[@]}"; do
  input="/data/public/AIGRewrite/test_designs/${design}.aig"
  export GATE1B_AUDIT="$OUT/${design}_alignment.json"
  export GATE1B_DESIGN="$design"
  unset GATE1A_ROOT GATE1A_CUT GATE1A_SUB GATE1A_PROVENANCE
  LD_LIBRARY_PATH="$BUILD" "$BUILD/abcg" -c "gread $input; grw" \
    > "$OUT/${design}_audit.log" 2>&1
done

input=/data/public/AIGRewrite/test_designs/dft_orig.aig
for sub in 0 2; do
  unset GATE1B_AUDIT GATE1B_DESIGN
  export GATE1A_ROOT=80078 GATE1A_CUT=0 GATE1A_SUB="$sub"
  export GATE1A_PROVENANCE="$OUT/sub${sub}_replay.json"
  output="$OUT/root80078_cut0_sub${sub}.aig"
  LD_LIBRARY_PATH="$BUILD" "$BUILD/abcg" -c "gread $input; grw; gwrite $output" \
    > "$OUT/sub${sub}_replay.log" 2>&1
  "$ABC" -c "cec $input $output" > "$OUT/sub${sub}_cec.log" 2>&1
  grep -q "Networks are equivalent" "$OUT/sub${sub}_cec.log"
done

jq -s '{design:"dft_orig",root:80078,cut_slot:0,
  alignment:(.[0].records[]|select(.root==80078 and .cut_slot==0)),
  candidates:[.[1],.[2]]}' \
  "$OUT/dft_orig_alignment.json" "$OUT/sub0_replay.json" "$OUT/sub2_replay.json" \
  > "$OUT/gate1b_alignment.json"

jq -s '{records:map(.records[])|sort_by(.design,.root,.cut_slot), records_seen:(map(.records_seen)|add)}' \
  "$OUT/dft_orig_alignment.json" "$OUT/idft_orig_alignment.json" "$OUT/vga_lcd_orig_alignment.json" \
  > "$OUT/gate1b_sample_records.json"
