#!/usr/bin/env bash
set -Eeuo pipefail

real_nvcc=/usr/local/cuda-11.8/bin/nvcc
rewritten=()
for argument in "$@"; do
  case "$argument" in
    -arch=compute_89)
      rewritten+=("-arch=compute_86")
      ;;
    -code=compute_89,sm_89)
      rewritten+=("-code=compute_86,sm_86")
      ;;
    *)
      rewritten+=("$argument")
      ;;
  esac
done
exec "$real_nvcc" "${rewritten[@]}"
