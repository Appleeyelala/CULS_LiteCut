#!/usr/bin/env bash
#SBATCH --job-name=gate2c-litecut-dx
#SBATCH --output=/data/wushangge/gate2c_runs/gate2c_validation_sm86_20260918/logs/litecut-dx-%j.out
#SBATCH --error=/data/wushangge/gate2c_runs/gate2c_validation_sm86_20260918/logs/litecut-dx-%j.err
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --chdir=/data/wushangge/gate2c_runs/gate2c_validation_sm86_20260918
#SBATCH --export=NONE

set -Eeuo pipefail
RUN_ROOT=/data/wushangge/gate2c_runs/gate2c_validation_sm86_20260918
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH="$RUN_ROOT/litecut_repo"

exec /home/share/process/anaconda3/bin/python - <<'PY'
import json
from pathlib import Path
import numpy as np
import gate2a_candidate_inference as bridge

root = Path('/data/wushangge/gate2c_runs/gate2c_validation_sm86_20260918')
record = root / 'output/candidate_records/c1355_r000146_c000_n000_sub00.json'
original = root / 'input/normalized_original.aig'
library = root / 'input/rewrite_library.inc'
checkpoint = root / 'input/best.pt'
a = bridge.infer_one(record, original, library, checkpoint)
b = bridge.infer_one(record, original, library, checkpoint)
fa = np.asarray(a['raw_features_25'], dtype=np.float64)
fb = np.asarray(b['raw_features_25'], dtype=np.float64)
ca = np.asarray(a['continuous_features_19'], dtype=np.float64)
cb = np.asarray(b['continuous_features_19'], dtype=np.float64)
out = {
    'raw_features_exact': a['raw_features_25'] == b['raw_features_25'],
    'continuous_features_exact': a['continuous_features_19'] == b['continuous_features_19'],
    'raw_features_max_abs_diff': float(np.max(np.abs(fa - fb))),
    'raw_features_diff_indices': np.flatnonzero(fa != fb).tolist(),
    'continuous_max_abs_diff': float(np.max(np.abs(ca - cb))),
    'replacement_dag_exact': a['replacement_dag'] == b['replacement_dag'],
    'prediction_a': a['inference']['predicted_raw_abc_delay'],
    'prediction_b': b['inference']['predicted_raw_abc_delay'],
    'prediction_abs_diff': abs(a['inference']['predicted_raw_abc_delay'] - b['inference']['predicted_raw_abc_delay']),
    'model_output_a': a['inference']['raw_model_output'],
    'model_output_b': b['inference']['raw_model_output'],
    'feature_a': a['raw_features_25'],
    'feature_b': b['raw_features_25'],
}
print(json.dumps(out, indent=2))
PY
