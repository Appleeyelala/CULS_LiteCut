#!/usr/bin/env bash
set -Eeuo pipefail
GCC_ROOT=${GATE2C_GCC_ROOT:?GATE2C_GCC_ROOT is required}
export GCC_EXEC_PREFIX="$GCC_ROOT/usr/local/"
export COMPILER_PATH="$GCC_ROOT/usr/local/libexec/gcc/x86_64-pc-linux-gnu/11.4.0:$GCC_ROOT/usr/local/lib/gcc/x86_64-pc-linux-gnu/11.4.0${COMPILER_PATH:+:$COMPILER_PATH}"
export LIBRARY_PATH="$GCC_ROOT/usr/local/lib64:$GCC_ROOT/usr/local/lib/gcc/x86_64-pc-linux-gnu/11.4.0${LIBRARY_PATH:+:$LIBRARY_PATH}"
exec "$GCC_ROOT/usr/local/bin/gcc" -B"$GCC_ROOT/usr/local/" "$@"
