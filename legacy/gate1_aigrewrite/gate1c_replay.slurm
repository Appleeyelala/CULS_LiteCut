#!/usr/bin/env bash
#SBATCH --job-name=gate1c-replay
#SBATCH --output=/data/wushangge/gate1a_AIGRewrite/gate1c_out/replay-%j.out
#SBATCH --error=/data/wushangge/gate1a_AIGRewrite/gate1c_out/replay-%j.err
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --chdir=/data/wushangge/gate1a_AIGRewrite
set -Eeuo pipefail
ROOT=/data/wushangge/gate1a_AIGRewrite; OUT="$ROOT/gate1c_out"; BUILD="$ROOT/build_gate1c"
INPUT=/data/public/AIGRewrite/case/c1355.aig; ABC=/data/public/AIGRewrite/abc/abc
mkdir -p "$OUT"; cmake --build "$BUILD" --target abcg -j2
for x in 0 1; do for y in 0 1; do
  tag="d1${x}_d2${y}"; export GATE1C_SPEC="146:0:${x},226:0:${y}" GATE1A_ROOT=146 GATE1A_CUT=0 GATE1A_SUB="$x" GATE1A_PROVENANCE="$OUT/${tag}.json"
  LD_LIBRARY_PATH="$BUILD" "$BUILD/abcg" -c "gread $INPUT; grw; gwrite $OUT/${tag}.aig" > "$OUT/${tag}.rewrite.log" 2>&1
  "$ABC" -c "cec $INPUT $OUT/${tag}.aig" > "$OUT/${tag}.cec.log" 2>&1; grep -q "Networks are equivalent" "$OUT/${tag}.cec.log"
  "$ABC" -c "read $OUT/${tag}.aig; print_stats -p" > "$OUT/${tag}.stats.log" 2>&1
done; done
sha256sum "$OUT"/d1*.aig > "$OUT/output_hashes.txt"
