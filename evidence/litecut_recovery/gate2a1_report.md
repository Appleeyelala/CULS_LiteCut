# Gate 2A-1 Artifact Provenance

## Conclusion: NOT_REPRODUCIBLE (student checkpoint recovered; exact historical reproduction still blocked)

The source proves that canonical teacher and student checkpoints were designed to be generated. Teacher writes `latest.pt` and `best.pt` under `ExtractCutLibConfig.checkpoint_dir`, defaulting to `./checkpoints/extract_cutlib_teacher`. Student writes `last.pt` and `best.pt` under `ExtractCutLibStudentConfig.checkpoint_dir`, defaulting to `./checkpoints/extract_cutlib_student`; `--result_root` plus `--experiment_name` resolves a grouped `<result_root>/<experiment_name>/checkpoints` directory. Normalization is embedded in checkpoints and may also be written into prepared-cache `.npy` files.

The supplied `/home/wushangge/cutLibOrigin/best.pt` is a genuine student checkpoint. It contains the student state dict, config (`epoch=150`, `global_step=7866150`, `seed=42`, MAD filter 10, input_dim 19), complete `x_mu`, `x_inv_sigma`, `y_mu`, `y_inv_sigma`, and an export payload. SHA256: `b3f499d2973bdb36fb42690794e9838d4c8359f05d6e141aa8cbe2b9fe3ae64d`. `/data/public/best.pt` is byte-identical. This establishes that at least the student checkpoint was generated and recovered.

The repository's shell scripts identify the most likely historical paths as:

```text
./checkpoints/extract_cutlib_teacher/best.pt
./checkpoints/extract_cutlib_student_group_fusion/best.pt
```

They also identify `/data/zhoulingfeng/data_cutLib/cutLibData2` as the training dataset root. That exact root, its generated `save_json_*` layout, prepared caches, and teacher checkpoint are absent. A related legacy dataset is present at `/data/public/src/data/save_json` (47 designs, with `abc_delay` in `data_2.json`), but it contains graph/cell records rather than the canonical 25-column LiteCut feature records and is not evidence of the exact teacher/student training population. Shell history and available Slurm/script logs contained no LiteCut training invocation or output path.

## Normalization and split

`standardize_features` computes `mu` and `inv_sigma` directly from the in-memory training preparation arrays. Labels are filtered according to the CLI option, transformed with `log1p`, then standardized. The data loader concatenates the five split directories (`save_json_OpenABC`, `save_json_ACE`, `save_json_iwls2022`, `save_json_iwls2024`, `save_json_forgeEDA`), filters labels, and permutes the population. The trainer then creates a second permutation for the train/validation split. Source defaults are seed `42` and train ratio `0.9`, but historical overrides and the exact generated JSON population are unavailable. Therefore normalization and exact split membership cannot be reconstructed exactly.

## Recovery matrix

| Item | Status |
|---|---|
| Canonical source and feature semantics | RECOVERED |
| Model architecture and loss definition | RECOVERED |
| Checkpoint write path semantics | RECOVERED |
| Historical checkpoint | MISSING |
| Exact generated training dataset | MISSING |
| Exact train/validation split | MISSING |
| Historical normalization arrays | MISSING |
| Historical raw-feature + `abc_delay` sample | MISSING |
| Historical CLI overrides/random seed | UNKNOWN |

An independently retrained model would therefore be a new reproduction model, not the original historical checkpoint. No training was started, and no LiteCut prediction was fabricated.

The detailed machine-readable evidence is in `gate2a1_artifact_provenance.json`.
