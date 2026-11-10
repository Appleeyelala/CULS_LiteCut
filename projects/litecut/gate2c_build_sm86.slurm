#!/usr/bin/env bash
#SBATCH --job-name=gate2c-sm86-build
#SBATCH --output=/data/wushangge/gate2c_runs/gate2c_sm86_build/logs/slurm-%j.out
#SBATCH --error=/data/wushangge/gate2c_runs/gate2c_sm86_build/logs/slurm-%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=eda
#SBATCH --chdir=/data/wushangge/gate2c_runs/gate2c_sm86_build
#SBATCH --export=ALL

set -Eeuo pipefail

export PATH=/usr/local/bin:/usr/bin:/bin:/usr/local/cuda-11.8/bin:$PATH

SOURCE=/data/wushangge/gate1a_AIGRewrite
RUN_ROOT=/data/wushangge/gate2c_runs/gate2c_sm86_build
BUILD="$RUN_ROOT/build"
WRAPPER="$RUN_ROOT/nvcc_sm86_wrapper.sh"
mkdir -p "$RUN_ROOT/logs" "$BUILD" "$RUN_ROOT/engine"

cmake -S "$SOURCE" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="$WRAPPER" \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++ \
  > "$RUN_ROOT/logs/cmake_configure.log" 2>&1
cmake --build "$BUILD" --target abcg -j4 \
  > "$RUN_ROOT/logs/cmake_build.log" 2>&1

cp "$BUILD/abcg" "$RUN_ROOT/engine/abcg"
cp "$BUILD/libabc.so" "$RUN_ROOT/engine/libabc.so"
sha256sum "$SOURCE/CMakeLists.txt" "$SOURCE/src/algorithms/rewrite.cu" \
  "$RUN_ROOT/engine/abcg" "$RUN_ROOT/engine/libabc.so" "$WRAPPER" \
  > "$RUN_ROOT/engine/sha256.txt"
{
  printf 'source=%s\n' "$SOURCE"
  printf 'build=%s\n' "$BUILD"
  printf 'requested_original_cuda_flags=-arch=compute_89 -code=compute_89,sm_89\n'
  printf 'effective_cuda_flags=-arch=compute_86 -code=compute_86,sm_86\n'
  printf 'source_CMakeLists_sha256='; sha256sum "$SOURCE/CMakeLists.txt" | awk '{print $1}'
  printf 'source_rewrite_sha256='; sha256sum "$SOURCE/src/algorithms/rewrite.cu" | awk '{print $1}'
  cat "$RUN_ROOT/engine/sha256.txt"
} > "$RUN_ROOT/engine/build_manifest.txt"
