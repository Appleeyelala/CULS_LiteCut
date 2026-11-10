#!/usr/bin/env bash
#SBATCH --job-name=gate2c-gpu-probe
#SBATCH --output=/data/wushangge/gate2c_runs/gate2c_gpu_probe/logs/slurm-%j.out
#SBATCH --error=/data/wushangge/gate2c_runs/gate2c_gpu_probe/logs/slurm-%j.err
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=eda3
#SBATCH --chdir=/data/wushangge/gate2c_runs/gate2c_gpu_probe
#SBATCH --export=ALL

set -u
ROOT=/data/wushangge/gate1a_AIGRewrite
BUILD="$ROOT/build_gate1c"
INPUT=/data/public/AIGRewrite/case/c1355.aig
OUT=/data/wushangge/gate2c_runs/gate2c_gpu_probe
mkdir -p "$OUT/logs"
{
  hostname
  nvidia-smi -L
  nvidia-smi
  env | sort | rg 'CUDA|SLURM|LD_LIBRARY|PATH'
} > "$OUT/environment.txt" 2>&1

cd "$OUT"
set +e
env -u GATE1C_SPEC -u GATE1B_AUDIT -u GATE1B_DESIGN -u GATE1B_ROOT \
  GATE1A_ROOT=146 GATE1A_CUT=0 GATE1A_SUB=0 GATE1A_PROVENANCE="$OUT/provenance.json" \
  LD_LIBRARY_PATH="$BUILD" \
  "$BUILD/abcg" -c "gread $INPUT; grw; gwrite $OUT/replayed.aig" > "$OUT/replay.log" 2>&1
printf 'abcg_rc=%s\n' "$?" >> "$OUT/environment.txt"
set -e
ls -l "$OUT" >> "$OUT/environment.txt"
