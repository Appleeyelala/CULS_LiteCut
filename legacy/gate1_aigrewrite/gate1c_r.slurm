#!/usr/bin/env bash
#SBATCH --job-name=gate1c-r
#SBATCH --output=/data/wushangge/gate1a_AIGRewrite/gate1c_r_out/slurm-%j.out
#SBATCH --error=/data/wushangge/gate1a_AIGRewrite/gate1c_r_out/slurm-%j.err
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --chdir=/data/wushangge/gate1a_AIGRewrite
set -Eeuo pipefail
ROOT=/data/wushangge/gate1a_AIGRewrite
OUT="$ROOT/gate1c_r_out"
BUILD="$ROOT/build_gate1c"
INPUT=/data/public/AIGRewrite/case/c1355.aig
ABC=/data/public/AIGRewrite/abc/abc
mkdir -p "$OUT"
cmake --build "$BUILD" --target abcg -j2
LD_LIBRARY_PATH="$BUILD" "$BUILD/abcg" -c "gread $INPUT; gwrite $OUT/normalized_original.aig" > "$OUT/normalize.log" 2>&1
for c in AC AD BC BD; do
  case "$c" in
    AC) x=0; y=0;; AD) x=0; y=1;; BC) x=1; y=0;; BD) x=1; y=1;;
  esac
  export GATE1C_SPEC="146:0:${x},226:0:${y}"
  export GATE1A_ROOT=146 GATE1A_CUT=0 GATE1A_SUB="$x"
  export GATE1A_PROVENANCE="$OUT/$c.json"
  LD_LIBRARY_PATH="$BUILD" "$BUILD/abcg" -c "gread $INPUT; grw; gwrite $OUT/$c.aig" > "$OUT/$c.rewrite.log" 2>&1
  "$ABC" -c "cec $OUT/normalized_original.aig $OUT/$c.aig" > "$OUT/${c}_cec.log" 2>&1
  "$ABC" -c "read $OUT/$c.aig; print_stats -p" > "$OUT/$c.stats.log" 2>&1
done
sha256sum "$OUT/normalized_original.aig" "$OUT"/{AC,AD,BC,BD}.aig > "$OUT/sha256.txt"
