#!/usr/bin/env bash
# Gate 3F timing validation.  The job workspace and all intermediate files
# remain under /data; the final artifact is copied to the repository by the
# post-processing command after the job finishes.
#SBATCH --job-name=culs-gate3f
#SBATCH --partition=Compute
#SBATCH --output=/data/wushangge/runs/CULS-gate3f-20260922-gate2babc/logs/slurm-%j.out
#SBATCH --error=/data/wushangge/runs/CULS-gate3f-20260922-gate2babc/logs/slurm-%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --chdir=/data/wushangge/runs/CULS-gate3f-20260922-gate2babc
#SBATCH --export=NONE

set -Eeuo pipefail
RUN_ROOT=/data/wushangge/runs/CULS-gate3f-20260922-gate2babc
export OMP_NUM_THREADS=1
export TMPDIR="$RUN_ROOT/tmp"
mkdir -p "$TMPDIR" "$RUN_ROOT/output"

/home/share/process/anaconda3/bin/python3 "$RUN_ROOT/src/gate3f_run.py" \
  --original-aig "$RUN_ROOT/input/c1355_orig.aig" \
  --library "$RUN_ROOT/input/asap7.lib" \
  --gpuls "$RUN_ROOT/bin/gpuls" \
  --region-map "$RUN_ROOT/input/node_to_strict_mffc_region.csv" \
  --gate3e "$RUN_ROOT/input/gate3e" \
  --abc "$RUN_ROOT/bin/abc" \
  --out-dir "$RUN_ROOT/output"
