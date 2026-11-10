#!/usr/bin/env bash
#SBATCH --job-name=gate2c-sm86-stage
#SBATCH --output=/data/wushangge/gate2c_runs/gate2c_sm86_build/logs/staged-%j.out
#SBATCH --error=/data/wushangge/gate2c_runs/gate2c_sm86_build/logs/staged-%j.err
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --chdir=/data/wushangge/gate2c_runs/gate2c_sm86_build
#SBATCH --export=NONE

set -Eeuo pipefail

export PATH=/usr/bin:/bin
RUN_ROOT=/data/wushangge/gate2c_runs/gate2c_sm86_build
SOURCE=/data/wushangge/gate1a_AIGRewrite
ORIGINAL_BUILD="$SOURCE/build_gate1c"
BUILD="$RUN_ROOT/build_sm86"
TOOL_ROOT="$RUN_ROOT/toolchain"
MAKE="$TOOL_ROOT/bin/make"
GCC_ROOT="$TOOL_ROOT"
GXX_WRAPPER="$RUN_ROOT/gxx_wrapper.sh"
GCC_WRAPPER="$RUN_ROOT/gcc_wrapper.sh"
NVCC_WRAPPER="$RUN_ROOT/nvcc_sm86_wrapper.sh"
GXX_SOURCE="$RUN_ROOT/gxx_wrapper_source.sh"
GCC_SOURCE="$RUN_ROOT/gcc_wrapper_source.sh"
NVCC_SOURCE="$RUN_ROOT/nvcc_sm86_wrapper_source.sh"
export GATE2C_TOOL_ROOT="$TOOL_ROOT"
export GATE2C_GCC_ROOT="$GCC_ROOT"
export GATE2C_GXX_WRAPPER="$GXX_WRAPPER"

mkdir -p "$RUN_ROOT/logs" "$BUILD"
if [[ ! -f "$BUILD/Makefile" ]]; then
  cp -a "$ORIGINAL_BUILD/." "$BUILD/"
fi
cp "$GXX_SOURCE" "$GXX_WRAPPER"
cp "$GCC_SOURCE" "$GCC_WRAPPER"
cp "$NVCC_SOURCE" "$NVCC_WRAPPER"
chmod +x "$GXX_WRAPPER" "$GCC_WRAPPER" "$NVCC_WRAPPER"

# The copied CMake-generated build tree is data-local.  Only its generated
# tool paths and CUDA code-generation flags are changed; source and libabc
# objects remain the Gate 1C-R artifacts.
sed -i \
  -e "s#-arch=compute_89 -code=compute_89,sm_89#-arch=compute_86 -code=compute_86,sm_86#g" \
  "$BUILD/CMakeFiles/abcg.dir/flags.make"
sed -i \
  -e "s#/usr/local/cuda-11.8/bin/nvcc#$NVCC_WRAPPER#g" \
  -e "s#/usr/local/cuda-11.8#$TOOL_ROOT/cuda-11.8#g" \
  -e "s#/usr/local/bin/cmake#$TOOL_ROOT/bin/cmake#g" \
  -e "s#/usr/local/bin/c++#$GXX_WRAPPER#g" \
  -e "s#/usr/local/bin/cc#$GCC_WRAPPER#g" \
  -e "s#/usr/local/bin/gcc#$GCC_WRAPPER#g" \
  "$BUILD/CMakeFiles/abcg.dir/build.make" \
  "$BUILD/CMakeFiles/abcg.dir/link.txt"

"$MAKE" -C "$BUILD" -f CMakeFiles/abcg.dir/build.make \
  CMakeFiles/abcg.dir/build -j4 > "$RUN_ROOT/logs/make_sm86.log" 2>&1
cp "$BUILD/abcg" "$RUN_ROOT/engine/abcg_sm86"
cp "$BUILD/libabc.so" "$RUN_ROOT/engine/libabc.so"
sha256sum "$SOURCE/CMakeLists.txt" "$SOURCE/src/algorithms/rewrite.cu" \
  "$RUN_ROOT/engine/abcg_sm86" "$RUN_ROOT/engine/libabc.so" \
  "$GXX_WRAPPER" "$NVCC_WRAPPER" > "$RUN_ROOT/engine/sm86_sha256.txt"
{
  printf 'source=%s\n' "$SOURCE"
  printf 'original_build=%s\n' "$ORIGINAL_BUILD"
  printf 'copied_build=%s\n' "$BUILD"
  printf 'requested_original_cuda_flags=-arch=compute_89 -code=compute_89,sm_89\n'
  printf 'effective_cuda_flags=-arch=compute_86 -code=compute_86,sm_86\n'
  printf 'source_and_candidate_semantics_unchanged=true\n'
  cat "$RUN_ROOT/engine/sm86_sha256.txt"
} > "$RUN_ROOT/engine/sm86_build_manifest.txt"
