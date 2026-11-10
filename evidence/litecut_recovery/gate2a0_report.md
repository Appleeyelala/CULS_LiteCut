# Gate 2A-0: Canonical LiteCut Artifact Recovery

## Status: PARTIAL

The canonical source repository was recovered at `/home/wushangge/cutLibOrigin`, commit `077aa1e49b8b97ed6d76013a7272c06bfacabf23`. It contains the old inference path and the newer teacher/student LiteCut implementation, but no usable trained teacher/student checkpoint was found under `/home/wushangge`, `/data/wushangge`, or `/data/public`.

## Recovered implementation

- Feature extractor: `src/data_preprocess/feat_extraction_v3.py` (`CompleteCutFeatureExtractor`).
- Teacher preprocessing/model: `src/extract_cutlib_teacher/data_preprocess.py`, `feature_branch.py`, `fusion.py`, `trainer.py`.
- Student preprocessing/model: `src/extract_cutlib_student/data_preprocess.py`, `modeling.py`, `trainer.py`.
- Inference/denormalization reference: `src/validate/eval_prediction_accuracy.py`.
- Historical old entry points: `src/abc_pipe/py_pipeline.py` (`inference_buildData`, `inference_model`) and `src/models/CellPool_wo_lib.py`.

## Feature and transform audit

The source-defined export order is the 25-column vector listed in `gate2a0_litecut_bundle.json`: 19 continuous features followed by `WL_Hash`. Teacher input is 19 continuous values plus the separate WL hash tensor. Student input is 19 values with context or 11 values without context. Source preprocessing casts model arrays to `float32` and uses fitted `standardize_features` parameters.

The new target transform is exactly `raw abc_delay -> log1p -> standardize`; inverse is `expm1(y_scaled / y_inv_sigma + y_mu)` after the implementation's `[-10,10]` log clamp. These parameters are expected in a trained checkpoint (`x_mu`, `x_inv_sigma`, `y_mu`, `y_inv_sigma`) and were not found.

`_precompute_global_context()` operates on the supplied whole-AIG graph: it computes forward depth `L`, reverse depth `R`, global depth `D`, and `SlackProxy = D-(L+R)`. Therefore context cannot be reconstructed from a candidate-local cut without the original whole-AIG snapshot.

## Artifact inventory

`/data/public/checkpoints/DeepCutDelay_*.pth` contains 205 old DeepCut checkpoints. Inspection shows the payload keys are `checkpoint`, `epoch`, and `optimizer`; the files do not contain the new normalization arrays or teacher/student model configuration. They are classified `ORIGINAL / OLD MODEL` and are not inference-ready LiteCut artifacts.

The repository scripts reference expected paths such as `./checkpoints/extract_cutlib_teacher/best.pt` and `./checkpoints/extract_cutlib_student_group_fusion/best.pt`, but those files are absent from all audited user-readable roots. The only local `.pth` in the recovered repository is `deepgate/pretrained/model.pth`, which is a DeepGate encoder prerequisite, not a LiteCut delay predictor checkpoint.

No historical raw-feature/`abc_delay` sample bundle was found in the audited roots. Consequently no numeric inference can be run without inventing a checkpoint, scaler, or sample, which is explicitly disallowed.

## Reproduction result

Historical LiteCut delay inference: **not reproducible in this environment** (`FAIL_NOT_RUN_MISSING_BUNDLE`). No prediction is reported. Gate 2A must remain blocked at `PARTIAL`; this run does not modify Gate 1C-R and does not train or substitute a model.
