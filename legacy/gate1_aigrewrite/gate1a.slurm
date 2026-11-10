#!/usr/bin/env bash
#SBATCH --job-name=gate1a-force
#SBATCH --output=/data/wushangge/gate1a_AIGRewrite/gate1a_out/slurm-%j.out
#SBATCH --error=/data/wushangge/gate1a_AIGRewrite/gate1a_out/slurm-%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --chdir=/data/wushangge/gate1a_AIGRewrite

set -Eeuo pipefail
RUN_ROOT=/data/wushangge/gate1a_AIGRewrite
OUT="$RUN_ROOT/gate1a_out"
BUILD="$RUN_ROOT/build_gate1a"
INPUT=/data/public/AIGRewrite/test_designs/dft_orig.aig
ABC=/data/public/AIGRewrite/abc/abc
export TMPDIR="$OUT/tmp"
mkdir -p "$OUT" "$TMPDIR" "$BUILD"

cmake -S "$RUN_ROOT" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD" --target abcg -j4

sha256sum "$INPUT" "$RUN_ROOT/src/algorithms/rewrite_library.inc" > "$OUT/input_hashes.txt"

for sub in 0 2; do
  export GATE1A_ROOT=80078
  export GATE1A_CUT=0
  export GATE1A_SUB="$sub"
  export GATE1A_PROVENANCE="$OUT/sub${sub}.json"
  LD_LIBRARY_PATH="$BUILD" "$BUILD/abcg" \
    -c "gread $INPUT; grw; gwrite $OUT/root80078_cut0_sub${sub}.aig" \
    > "$OUT/sub${sub}_rewrite.log" 2>&1
  "$ABC" -c "cec $INPUT $OUT/root80078_cut0_sub${sub}.aig" \
    > "$OUT/sub${sub}_cec.log" 2>&1
  grep -q "Networks are equivalent" "$OUT/sub${sub}_cec.log"
  "$ABC" -c "read $OUT/root80078_cut0_sub${sub}.aig; print_stats -p" \
    > "$OUT/sub${sub}_stats.log" 2>&1
  design_sha=$(sha256sum "$INPUT" | awk '{print $1}')
  library_sha=$(sha256sum "$RUN_ROOT/src/algorithms/rewrite_library.inc" | awk '{print $1}')
  output_sha=$(sha256sum "$OUT/root80078_cut0_sub${sub}.aig" | awk '{print $1}')
  jq --arg design_sha256 "$design_sha" --arg rewrite_library_sha256 "$library_sha" \
     --arg output_aig "$OUT/root80078_cut0_sub${sub}.aig" --arg output_sha256 "$output_sha" \
     '. + {design_sha256:$design_sha256,rewrite_library_sha256:$rewrite_library_sha256,output_aig:$output_aig,output_sha256:$output_sha256,cec:"PASS"}' \
     "$OUT/sub${sub}.json" > "$OUT/sub${sub}.json.tmp"
  mv "$OUT/sub${sub}.json.tmp" "$OUT/sub${sub}.json"
done

sha256sum "$OUT"/root80078_cut0_sub*.aig > "$OUT/output_hashes.txt"
