"""Evaluate MAPE and RankAcc for ablation Student models (E0-E3) on the 14 selected cases."""

from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime
from pathlib import Path

import re
import numpy as np
import torch

FEATURE_INPUT_DIM = 19  # 8 + 6 + 5


def _now() -> str:
    return datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAPE (%) on raw delay values."""
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (np.abs(y_true) > 1e-8)
    if mask.sum() == 0:
        return float("nan")
    denom = np.maximum(np.abs(y_true[mask]), 1e-8)
    return float(np.mean(np.abs(y_pred[mask] - y_true[mask]) / denom) * 100.0)


def compute_rank_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pairwise ranking accuracy (%) over all pairs with y_true[i] != y_true[j]."""
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    n = len(y_true)
    if n < 2:
        return float("nan")

    MAX_PAIRS = 500_000
    n_all_pairs = n * (n - 1) // 2

    if n_all_pairs <= MAX_PAIRS:
        idx_i, idx_j = np.triu_indices(n, k=1)
        true_diff = y_true[idx_i] - y_true[idx_j]
        pred_diff = y_pred[idx_i] - y_pred[idx_j]
    else:
        rng = np.random.default_rng(42)
        idx_i = rng.integers(0, n, size=MAX_PAIRS)
        idx_j = rng.integers(0, n, size=MAX_PAIRS)
        same = idx_i == idx_j
        idx_j[same] = (idx_j[same] + 1) % n
        true_diff = y_true[idx_i] - y_true[idx_j]
        pred_diff = y_pred[idx_i] - y_pred[idx_j]

    nonzero = np.abs(true_diff) > 1e-12
    if nonzero.sum() == 0:
        return float("nan")
    concordant = np.sign(true_diff[nonzero]) == np.sign(pred_diff[nonzero])
    return float(concordant.mean() * 100.0)


def denormalize_y(y_scaled: np.ndarray, y_mu: np.ndarray, y_inv_sigma: np.ndarray) -> np.ndarray:
    """Reverse standardize + log1p transform: y_raw = expm1(y_scaled / inv_sigma + mu)."""
    y_log = y_scaled / y_inv_sigma + y_mu
    y_log = np.clip(y_log, -10, 10)
    return np.expm1(y_log)


def upgrade_legacy_student_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Translate old checkpoint key names to current code names."""
    upgraded: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        for old, new in (
            ("group1_encoder.", "group_encoders.0."),
            ("group2_encoder.", "group_encoders.1."),
            ("group3_encoder.", "group_encoders.2."),
        ):
            if new_key.startswith(old):
                new_key = new + new_key[len(old):]
                break
        new_key = re.sub(r"^experts\.(\d+)\.0\.", r"experts.\1.head.0.", new_key)
        new_key = re.sub(r"^experts\.(\d+)\.2\.", r"experts.\1.head.2.", new_key)
        if new_key.startswith("teacher_align_adapter."):
            continue
        upgraded[new_key] = value
    return upgraded


def load_student_model(checkpoint_path: str, device: str = "cpu") -> tuple[torch.nn.Module, dict]:
    """Load Student model from checkpoint."""
    from extract_cutlib_student.config import ExtractCutLibStudentConfig
    from extract_cutlib_student.modeling import StudentModel

    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt["config"]
    if isinstance(cfg, dict):
        cfg = ExtractCutLibStudentConfig(**{k: v for k, v in cfg.items() if k in ExtractCutLibStudentConfig.__dataclass_fields__})

    model = StudentModel(
        input_dim=cfg.input_dim,
        shared_hidden_dim=cfg.shared_hidden_dim,
        with_context8=cfg.with_context8,
        with_moe=cfg.with_moe,
        num_experts=cfg.num_experts,
        gate_temperature=cfg.gate_temperature,
        teacher_hidden_dim=cfg.teacher_hidden_dim,
    )
    sd = ckpt["model_state_dict"]
    try:
        model.load_state_dict(sd, strict=True)
    except RuntimeError:
        model.load_state_dict(upgrade_legacy_student_state_dict(sd), strict=True)
    model.to(device)
    model.eval()
    return model, ckpt


def load_case_data(dataset_path: str, dataset: str, case: str) -> tuple[np.ndarray, np.ndarray]:
    """Load raw features (x, 19-dim) and labels (y) for a single case."""
    from subutils.extraction_utils import _load_single_json_case

    if dataset == "OpenABC":
        json_dir = f"save_json_{dataset}"
        case_dir = f"{case}_orig"
    else:
        json_dir = f"save_json_{dataset}"
        case_dir = case

    json_path = os.path.join(dataset_path, json_dir, case_dir, "data_2.json")
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"data_2.json not found: {json_path}")

    _, x_batch, y_batch, _, n_cuts = _load_single_json_case(json_path, include_graph=False)
    if x_batch is None or n_cuts == 0:
        raise ValueError(f"No valid cuts in {json_path}")

    x = x_batch[:, :FEATURE_INPUT_DIM]
    return x, y_batch


def infer_student(model: torch.nn.Module, x_scaled: np.ndarray,
                  y_mu: np.ndarray, y_inv_sigma: np.ndarray,
                  with_context8: bool,
                  device: str = "cpu", batch_size: int = 2048) -> np.ndarray:
    """Run Student inference and return raw predictions."""
    if not with_context8:
        x_scaled = x_scaled[:, 8:]

    preds = []
    with torch.inference_mode():
        for start in range(0, len(x_scaled), batch_size):
            end = min(start + batch_size, len(x_scaled))
            x_t = torch.from_numpy(x_scaled[start:end]).float().to(device)
            output = model(x_t)
            preds.append(output.prediction.cpu().numpy())
    y_pred_scaled = np.concatenate(preds, axis=0)
    return denormalize_y(y_pred_scaled, y_mu, y_inv_sigma)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate MAPE & RankAcc for ablation Student models on selected cases")
    p.add_argument("--dataset_path", required=True, help="Root of data_2.json files")
    p.add_argument("--picks_csv", required=True, help="main_table_picks.csv with dataset,case columns")
    p.add_argument("--result_root", required=True, help="Root of ablation results (e.g. results/ablation_main_table)")
    p.add_argument("--experiments", nargs="+", required=True,
                   help="Experiment entries: either dir name under result_root, or name:ckpt_path")
    p.add_argument("--output_csv", required=True, help="Output CSV path")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main() -> int:
    print(f"CurPID: {os.getpid()}")
    args = parse_args()

    # Load picks
    with open(args.picks_csv, encoding="utf-8") as f:
        picks = list(csv.DictReader(f))
    print(f"{_now()} Loaded {len(picks)} cases from {args.picks_csv}")

    # Load all models
    models = {}
    ckpt_data = {}
    exp_names = []
    for entry in args.experiments:
        if ":" in entry:
            exp_name, ckpt_path = entry.split(":", 1)
        else:
            exp_name = entry
            ckpt_path = os.path.join(args.result_root, exp_name, "checkpoints", "best.pt")
        exp_names.append(exp_name)
        if not os.path.isfile(ckpt_path):
            print(f"{_now()} WARNING: {ckpt_path} not found, skipping {exp_name}")
            continue
        print(f"{_now()} Loading {exp_name} from {ckpt_path}")
        model, ckpt = load_student_model(ckpt_path, args.device)
        models[exp_name] = model
        ckpt_data[exp_name] = ckpt
        cfg = ckpt["config"]
        with_ctx = cfg.get("with_context8", True) if isinstance(cfg, dict) else getattr(cfg, "with_context8", True)
        input_dim = cfg["input_dim"] if isinstance(cfg, dict) else cfg.input_dim
        print(f"{_now()}   with_context8={with_ctx} input_dim={input_dim}")

    if not models:
        print(f"{_now()} ERROR: No models loaded")
        return 1

    # Use first model's y normalization params (all identical)
    first_ckpt = next(iter(ckpt_data.values()))
    y_mu = np.asarray(first_ckpt["y_mu"], dtype=np.float32)
    y_inv_sigma = np.asarray(first_ckpt["y_inv_sigma"], dtype=np.float32)

    # Evaluate each case
    results = []
    for i, row in enumerate(picks):
        dataset = row["dataset"]
        case = row["case"]
        print(f"{_now()} [{i+1}/{len(picks)}] {dataset}/{case}")

        try:
            x_raw, y_raw = load_case_data(args.dataset_path, dataset, case)
        except (FileNotFoundError, ValueError) as e:
            print(f"{_now()}   ERROR: {e}")
            continue

        n_cuts = len(y_raw)
        print(f"{_now()}   {n_cuts} cuts loaded")

        result_row = {"dataset": dataset, "case": case, "n_cuts": n_cuts}

        for exp_name, model in models.items():
            ckpt = ckpt_data[exp_name]
            cfg = ckpt["config"]
            with_ctx = cfg.get("with_context8", True) if isinstance(cfg, dict) else getattr(cfg, "with_context8", True)

            x_mu = np.asarray(ckpt["x_mu"], dtype=np.float32)
            x_inv_sigma = np.asarray(ckpt["x_inv_sigma"], dtype=np.float32)

            if with_ctx:
                x_scaled = (x_raw - x_mu) * x_inv_sigma
            else:
                x_scaled_11 = (x_raw[:, 8:] - x_mu) * x_inv_sigma
                x_scaled = np.zeros_like(x_raw)
                x_scaled[:, 8:] = x_scaled_11

            pred = infer_student(model, x_scaled, y_mu, y_inv_sigma, with_ctx, args.device)
            mape = compute_mape(y_raw, pred)
            rank_acc = compute_rank_accuracy(y_raw.reshape(-1), pred.reshape(-1))
            print(f"{_now()}   {exp_name}: MAPE={mape:.2f}% RankAcc={rank_acc:.2f}%")

            result_row[f"{exp_name}_mape"] = f"{mape:.4f}"
            result_row[f"{exp_name}_rank_acc"] = f"{rank_acc:.4f}"

        results.append(result_row)

    # Build fieldnames
    fieldnames = ["dataset", "case", "n_cuts"]
    for exp_name in exp_names:
        if exp_name in models:
            fieldnames.append(f"{exp_name}_mape")
            fieldnames.append(f"{exp_name}_rank_acc")

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"{_now()} Saved {len(results)} rows to {args.output_csv}")

    # Print summary table
    print(f"\n{'='*120}")
    header = f"{'case':>25s}"
    for exp_name in exp_names:
        if exp_name in models:
            header += f" | {exp_name}_MAPE {exp_name}_Rank"
    print(header)
    print("-" * 120)
    for r in results:
        line = f"{r['dataset']+'/'+r['case']:>25s}"
        for exp_name in exp_names:
            if exp_name in models:
                m = r.get(f"{exp_name}_mape", "N/A")
                ra = r.get(f"{exp_name}_rank_acc", "N/A")
                line += f" | {m:>16s} {ra:>16s}"
        print(line)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
