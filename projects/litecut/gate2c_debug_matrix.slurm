#!/usr/bin/env bash
#SBATCH --job-name=gate2c-debug
#SBATCH --output=/data/wushangge/gate2c_runs/gate2c_debug_matrix/logs/slurm-%j.out
#SBATCH --error=/data/wushangge/gate2c_runs/gate2c_debug_matrix/logs/slurm-%j.err
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --chdir=/data/wushangge/gate2c_runs/gate2c_debug_matrix
#SBATCH --export=ALL

set -u

ROOT=/data/wushangge/gate1a_AIGRewrite
BUILD="$ROOT/build_gate1c"
STAGED=/data/wushangge/gate2c_runs/gate2c_validation_v2_20260917/bin
RAW=/data/public/AIGRewrite/case/c1355.aig
NORMALIZED=/data/wushangge/gate2c_runs/gate2c_validation_v2_20260917/input/normalized_original.aig
OUT=/data/wushangge/gate2c_runs/gate2c_debug_matrix/output
mkdir -p "$OUT" "$OUT/logs" "$OUT/work"

run_case() {
  local tag=$1 engine=$2 cwd=$3 input=$4 output=$5 mode=$6
  local prov="$OUT/${tag}.json" log="$OUT/logs/${tag}.log"
  rm -f "$output" "$prov" "$log"
  mkdir -p "$cwd" "$(dirname "$output")"
  (cd "$cwd" && env -u GATE1C_SPEC -u GATE1B_AUDIT -u GATE1B_DESIGN -u GATE1B_ROOT \
    GATE1A_ROOT=146 GATE1A_CUT=0 GATE1A_SUB=0 GATE1A_PROVENANCE="$prov" \
    LD_LIBRARY_PATH="$engine_dir" \
    "$engine" -c "gread $mode; grw; gwrite $output") >"$log" 2>&1
  local rc=$?
  printf '%s\trc=%s\tengine=%s\tcwd=%s\tinput=%s\toutput=%s\tprov=%s\n' \
    "$tag" "$rc" "$engine" "$cwd" "$input" "$output" "$prov" >> "$OUT/results.tsv"
  printf 'case=%s rc=%s output=%s prov=%s\n' "$tag" "$rc" \
    "$(test -f "$output" && echo yes || echo no)" \
    "$(test -f "$prov" && echo yes || echo no)" >> "$OUT/results.tsv"
}

: > "$OUT/results.tsv"
printf 'hostname=%s\n' "$(hostname)" > "$OUT/job_env.txt"
env | sort >> "$OUT/job_env.txt"

# The first four cases differ only in input/cwd/output location and use the
# original Gate 1 executable.  The next two use the staged executable.
engine_dir="$BUILD"
run_case original_root_raw "$BUILD/abcg" "$ROOT" "$RAW" "$OUT/original_root_raw.aig" "$RAW"
run_case original_root_normalized "$BUILD/abcg" "$ROOT" "$NORMALIZED" "$OUT/original_root_normalized.aig" "$NORMALIZED"
run_case original_work_raw "$BUILD/abcg" "$OUT/work/original_raw" "$RAW" "$OUT/original_work_raw.aig" "$RAW"
run_case original_work_normalized "$BUILD/abcg" "$OUT/work/original_normalized" "$NORMALIZED" "$OUT/original_work_normalized.aig" "$NORMALIZED"
engine_dir="$STAGED"
run_case staged_work_normalized "$STAGED/abcg" "$OUT/work/staged_normalized" "$NORMALIZED" "$OUT/staged_work_normalized.aig" "$NORMALIZED"
run_case staged_work_nested "$STAGED/abcg" "$OUT/work/staged_nested" "$NORMALIZED" "$OUT/replayed_aigs/c1355_r000146_c000_n000_sub00.aig" "$NORMALIZED"

# Approximate Slurm --export=NONE while preserving the GPU assignment and the
# runtime loader variables needed by the executable.
tag=staged_cleanenv
engine="$STAGED/abcg"
cwd="$OUT/work/cleanenv"
output="$OUT/staged_cleanenv.aig"
prov="$OUT/${tag}.json"
log="$OUT/logs/${tag}.log"
mkdir -p "$cwd"
(cd "$cwd" && env -i PATH=/usr/bin:/bin HOME=/tmp CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  GATE1A_ROOT=146 GATE1A_CUT=0 GATE1A_SUB=0 GATE1A_PROVENANCE="$prov" \
  LD_LIBRARY_PATH="$STAGED" \
  "$engine" -c "gread $NORMALIZED; grw; gwrite $output") >"$log" 2>&1
rc=$?
printf '%s\trc=%s\tengine=%s\tcwd=%s\tinput=%s\toutput=%s\tprov=%s\n' \
  "$tag" "$rc" "$engine" "$cwd" "$NORMALIZED" "$output" "$prov" >> "$OUT/results.tsv"
printf 'case=%s rc=%s output=%s prov=%s\n' "$tag" "$rc" \
  "$(test -f "$output" && echo yes || echo no)" \
  "$(test -f "$prov" && echo yes || echo no)" >> "$OUT/results.tsv"

printf '%s\n' '--- results.tsv ---'
cat "$OUT/results.tsv"
