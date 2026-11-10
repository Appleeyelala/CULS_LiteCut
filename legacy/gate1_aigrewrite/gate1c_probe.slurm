#!/usr/bin/env bash
#SBATCH --job-name=gate1c-probe
#SBATCH --output=/data/wushangge/gate1a_AIGRewrite/gate1c_out/probe-%j.out
#SBATCH --error=/data/wushangge/gate1a_AIGRewrite/gate1c_out/probe-%j.err
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --chdir=/data/wushangge/gate1a_AIGRewrite

set -Eeuo pipefail
ROOT=/data/wushangge/gate1a_AIGRewrite
OUT="$ROOT/gate1c_out"
BUILD="$ROOT/build_gate1c"
mkdir -p "$OUT" "$BUILD"
cmake -S "$ROOT" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD" --target abcg -j2
for design in c1355 cavlc c2670; do
  export GATE1B_AUDIT="$OUT/${design}_probe.json" GATE1B_DESIGN="$design"
  unset GATE1B_ROOT GATE1A_ROOT GATE1A_CUT GATE1A_SUB GATE1A_PROVENANCE
  LD_LIBRARY_PATH="$BUILD" "$BUILD/abcg" -c "gread /data/public/AIGRewrite/case/${design}.aig; grw" > "$OUT/${design}_probe.log" 2>&1
done
