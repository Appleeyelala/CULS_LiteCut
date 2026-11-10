#!/usr/bin/env bash
#SBATCH --job-name=gate1cr-b
#SBATCH --output=/data/wushangge/gate1a_AIGRewrite/gate1c_r_out/stage_b-%j.out
#SBATCH --error=/data/wushangge/gate1a_AIGRewrite/gate1c_r_out/stage_b-%j.err
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --chdir=/data/wushangge/gate1a_AIGRewrite
set -Eeuo pipefail
ROOT=/data/wushangge/gate1a_AIGRewrite
OUT="$ROOT/gate1c_r_out"
BUILD="$ROOT/build_gate1c"
RAW=/data/public/AIGRewrite/case/c1355.aig
ABC=/data/public/AIGRewrite/abc/abc
mkdir -p "$OUT"
cmake --build "$BUILD" --target abcg -j2
unset GATE1C_SPEC GATE1A_ROOT GATE1A_CUT GATE1A_SUB GATE1A_PROVENANCE
LD_LIBRARY_PATH="$BUILD" "$BUILD/abcg" -c "gread $RAW; gwrite $OUT/no_rewrite_roundtrip.aig" > "$OUT/no_rewrite_roundtrip.log" 2>&1
for item in raw_c1355:$RAW no_rewrite_roundtrip:$OUT/no_rewrite_roundtrip.aig gate1a_sub0:$ROOT/gate1a_out/root80078_cut0_sub0.aig gate1a_sub2:$ROOT/gate1a_out/root80078_cut0_sub2.aig; do
  tag=${item%%:*}; file=${item#*:}
  python3 "$ROOT/gate1c_r_io_signature.py" "$file" > "$OUT/io_signature_${tag}.json"
  "$ABC" -c "read $file; print_stats -p; print_io" > "$OUT/io_stats_${tag}.log" 2>&1
done
"$ABC" -c "cec $RAW $OUT/no_rewrite_roundtrip.aig" > "$OUT/no_rewrite_roundtrip_cec.log" 2>&1
grep -q "Networks are equivalent" "$OUT/no_rewrite_roundtrip_cec.log"
