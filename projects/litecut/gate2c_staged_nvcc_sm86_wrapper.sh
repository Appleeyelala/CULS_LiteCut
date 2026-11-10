#!/usr/bin/env bash
set -Eeuo pipefail
TOOL_ROOT=${GATE2C_TOOL_ROOT:?GATE2C_TOOL_ROOT is required}
GXX_WRAPPER=${GATE2C_GXX_WRAPPER:?GATE2C_GXX_WRAPPER is required}
exec "$TOOL_ROOT/cuda-11.8/bin/nvcc" -ccbin "$GXX_WRAPPER" "$@"
