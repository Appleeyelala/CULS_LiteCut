"""Evaluate prediction accuracy (MAPE, Ranking Accuracy) of Teacher / Student / Student-no-distill
on the 14 selected cases from main_table_picks.csv."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime
from pathlib import Path

import re

import numpy as np
import torch

FEATURE_INPUT_DIM = 19  # 8 + 6 + 5


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
        # Skip teacher_align_adapter (not used in inference)
        if new_key.startswith("teacher_align_adapter."):
            continue
        upgraded[new_key] = value
    return upgraded


def _now() -> str:
    return datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")


def _log1p_transform_y(y: np.ndarray) -> np.ndarray:
    return np.log1p(y)


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

    # For large N, sampling pairs to avoid O(N^2) memory
    MAX_PAIRS = 500_000
    n_all_pairs = n * (n - 1) // 2

    if n_all_pairs <= MAX_PAIRS:
        # Full pairwise
        idx_i, idx_j = np.triu_indices(n, k=1)
        true_diff = y_true[idx_i] - y_true[idx_j]
        pred_diff = y_pred[idx_i] - y_pred[idx_j]
    else:
        # Random sampling
        rng = np.random.default_rng(42)
        idx_i = rng.integers(0, n, size=MAX_PAIRS)
        idx_j = rng.integers(0, n, size=MAX_PAIRS)
        same = idx_i == idx_j
        idx_j[same] = (idx_j[same] + 1) % n
        true_diff = y_true[idx_i] - y_true[idx_j]
        pred_diff = y_pred[idx_i] - y_pred[idx_j]

    # Exclude ties in ground truth
    nonzero = np.abs(true_diff) > 1e-12
    if nonzero.sum() == 0:
        return float("nan")
    concordant = np.sign(true_diff[nonzero]) == np.sign(pred_diff[nonzero])
    return float(concordant.mean() * 100.0)


def load_case_data(dataset_path: str, dataset: str, case: str) -> tuple[np.ndarray, np.ndarray]:
    """Load raw features (x) and labels (y) for a single case from data_2.json.

    Returns (x, y) where x is [N, D] and y is [N, 1].
    """
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

    # Take first FEATURE_INPUT_DIM columns (19 continuous features)
    x = x_batch[:, :FEATURE_INPUT_DIM]
    return x, y_batch


def denormalize_y(y_scaled: np.ndarray, y_mu: np.ndarray, y_inv_sigma: np.ndarray) -> np.ndarray:
    """Reverse standardize + log1p transform: y_raw = expm1(y_scaled / inv_sigma + mu)."""
    y_log = y_scaled / y_inv_sigma + y_mu
    y_log = np.clip(y_log, -10, 10)
    return np.expm1(y_log)


def standardize_with_params(x: np.ndarray, mu: np.ndarray, inv_sigma: np.ndarray) -> np.ndarray:
    return (x - mu) * inv_sigma


def load_teacher_model(checkpoint_path: str, device: str = "cpu") -> tuple[torch.nn.Module, dict]:
    """Load Teacher model from checkpoint."""
    from extract_cutlib_teacher.config import ExtractCutLibConfig
    from extract_cutlib_teacher.fusion import TeacherTwoBranchModel

    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt["config"]
    if isinstance(cfg, dict):
        cfg = ExtractCutLibConfig(**{k: v for k, v in cfg.items() if k in ExtractCutLibConfig.__dataclass_fields__})

    model = TeacherTwoBranchModel(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt


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


def infer_teacher(model: torch.nn.Module, x_scaled: np.ndarray, wl_hash: np.ndarray,
                  y_mu: np.ndarray, y_inv_sigma: np.ndarray,
                  device: str = "cpu", batch_size: int = 2048) -> np.ndarray:
    """Run Teacher inference and return raw predictions."""
    preds = []
    with torch.inference_mode():
        for start in range(0, len(x_scaled), batch_size):
            end = min(start + batch_size, len(x_scaled))
            x_t = torch.from_numpy(x_scaled[start:end]).float().to(device)
            wl_t = torch.from_numpy(wl_hash[start:end]).long().to(device)
            y_pred, _, _ = model(x_t, wl_t, None)
            preds.append(y_pred.cpu().numpy())
    y_pred_scaled = np.concatenate(preds, axis=0)
    return denormalize_y(y_pred_scaled, y_mu, y_inv_sigma)


def infer_student(model: torch.nn.Module, x_scaled: np.ndarray,
                  y_mu: np.ndarray, y_inv_sigma: np.ndarray,
                  device: str = "cpu", batch_size: int = 2048) -> np.ndarray:
    """Run Student inference and return raw predictions."""
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
    p = argparse.ArgumentParser(description="Evaluate prediction accuracy for Teacher/Student on selected cases")
    p.add_argument("--dataset_path", required=True, help="Root of data_2.json files")
    p.add_argument("--picks_csv", required=True, help="main_table_picks.csv with dataset,case columns")
    p.add_argument("--teacher_ckpt", required=True, help="Teacher checkpoint path")
    p.add_argument("--student_ckpt", required=True, help="Student (distilled) checkpoint path")
    p.add_argument("--student_nodistill_ckpt", default=None, help="Student (no distill) checkpoint path (optional)")
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

    # Load normalization params from Student checkpoint (shared with Teacher)
    student_ckpt_data = torch.load(args.student_ckpt, map_location="cpu")
    x_mu = np.asarray(student_ckpt_data["x_mu"], dtype=np.float32)
    x_inv_sigma = np.asarray(student_ckpt_data["x_inv_sigma"], dtype=np.float32)
    y_mu = np.asarray(student_ckpt_data["y_mu"], dtype=np.float32)
    y_inv_sigma = np.asarray(student_ckpt_data["y_inv_sigma"], dtype=np.float32)
    print(f"{_now()} Loaded normalization params from Student checkpoint")

    # Load models
    print(f"{_now()} Loading Teacher model...")
    teacher_model, _ = load_teacher_model(args.teacher_ckpt, args.device)

    print(f"{_now()} Loading Student (distilled) model...")
    student_model, _ = load_student_model(args.student_ckpt, args.device)

    student_nodistill_model = None
    if args.student_nodistill_ckpt and os.path.isfile(args.student_nodistill_ckpt):
        print(f"{_now()} Loading Student (no distill) model...")
        student_nodistill_model, _ = load_student_model(args.student_nodistill_ckpt, args.device)
    else:
        print(f"{_now()} Student (no distill) checkpoint not provided, skipping")

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

        # Standardize features using training-set params
        x_scaled = standardize_with_params(x_raw, x_mu, x_inv_sigma)
        # WL hash is last column of full feature vector (col index 24 in original 25-dim)
        # But we only loaded first 19 dims. Need to reload for wl_hash.
        wl_hash = np.zeros(n_cuts, dtype=np.int64)  # placeholder, Teacher needs it
        # Actually load wl_hash from the raw data
        try:
            from subutils.extraction_utils import _load_single_json_case
            if dataset == "OpenABC":
                json_path = os.path.join(args.dataset_path, f"save_json_{dataset}", f"{case}_orig", "data_2.json")
            else:
                json_path = os.path.join(args.dataset_path, f"save_json_{dataset}", case, "data_2.json")
            _, x_full, _, _, _ = _load_single_json_case(json_path, include_graph=False)
            if x_full is not None and x_full.shape[1] > FEATURE_INPUT_DIM:
                wl_hash = x_full[:, -1].astype(np.int64)
        except Exception:
            pass  # use zeros if can't load

        # Teacher inference
        teacher_pred = infer_teacher(teacher_model, x_scaled, wl_hash, y_mu, y_inv_sigma, args.device)
        teacher_mape = compute_mape(y_raw, teacher_pred)
        teacher_rank = compute_rank_accuracy(y_raw.reshape(-1), teacher_pred.reshape(-1))
        print(f"{_now()}   Teacher: MAPE={teacher_mape:.2f}% RankAcc={teacher_rank:.2f}%")

        # Student (distilled) inference
        student_pred = infer_student(student_model, x_scaled, y_mu, y_inv_sigma, args.device)
        student_mape = compute_mape(y_raw, student_pred)
        student_rank = compute_rank_accuracy(y_raw.reshape(-1), student_pred.reshape(-1))
        print(f"{_now()}   Student: MAPE={student_mape:.2f}% RankAcc={student_rank:.2f}%")

        # Student (no distill) inference
        nodistill_mape = ""
        nodistill_rank = ""
        if student_nodistill_model is not None:
            nodistill_pred = infer_student(student_nodistill_model, x_scaled, y_mu, y_inv_sigma, args.device)
            nodistill_mape = f"{compute_mape(y_raw, nodistill_pred):.4f}"
            nodistill_rank = f"{compute_rank_accuracy(y_raw.reshape(-1), nodistill_pred.reshape(-1)):.4f}"
            print(f"{_now()}   Student(no-distill): MAPE={nodistill_mape}% RankAcc={nodistill_rank}%")

        results.append({
            "dataset": dataset,
            "case": case,
            "n_cuts": n_cuts,
            "teacher_mape": f"{teacher_mape:.4f}",
            "teacher_rank_acc": f"{teacher_rank:.4f}",
            "student_mape": f"{student_mape:.4f}",
            "student_rank_acc": f"{student_rank:.4f}",
            "student_nodistill_mape": nodistill_mape,
            "student_nodistill_rank_acc": nodistill_rank,
        })

    # Write output
    fieldnames = ["dataset", "case", "n_cuts",
                  "teacher_mape", "teacher_rank_acc",
                  "student_mape", "student_rank_acc",
                  "student_nodistill_mape", "student_nodistill_rank_acc"]
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"{_now()} Saved {len(results)} rows to {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
