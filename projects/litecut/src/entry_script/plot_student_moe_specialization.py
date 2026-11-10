from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
from dataclasses import replace
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)
if not logging.root.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from extract_cutlib_student.config import ExtractCutLibStudentConfig
from extract_cutlib_student.data_preprocess import (
    CONTEXT_FEATURE_DIM,
    StudentPreparedData,
    StudentDataset,
    _build_label_outlier_mask,
    adapt_prepared_data_for_config,
    collate_student_batch,
    load_student_data,
    load_student_data_cache,
)
from extract_cutlib_student.modeling import StudentModel
from extract_cutlib_student.trainer import StudentTrainer
from extract_cutlib_teacher.feature_branch import FEATURE_INPUT_DIM
from extract_cutlib_teacher.preprocess import seed_all
import subutils.extraction_utils as ext_utils


def _set_font_rcparams() -> None:
    plt.rcParams.update(
        {
            "font.size": 12,
            "font.weight": "bold",
            "font.family": "times new roman",
            "axes.labelsize": 13,
            "axes.titlesize": 15,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
        }
    )
    plt.rcParams["text.parse_math"] = False


def apply_publication_style() -> None:
    import scienceplots  # noqa: F401 — raises ImportError if not installed
    plt.style.use(["science", "ieee", "no-latex"])
    _set_font_rcparams()
    # scienceplots leaves text.parse_math=True and mathtext.fontset=cm,
    # which causes colorbar ticks to render as $\mathdefault{0.0}$.
    plt.rcParams["text.usetex"] = False
    plt.rcParams["text.parse_math"] = False
    plt.rcParams["mathtext.default"] = "regular"
    plt.rcParams["mathtext.fontset"] = "dejavusans"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Student MoE specialization heatmaps and per-expert summaries")
    parser.add_argument("--checkpoint", required=True, help="Path to Student checkpoint")
    parser.add_argument("--scope", choices=["all", "val"], default="val", help="Whether to use all samples or val split")
    parser.add_argument("--output_dir", required=True, help="Directory for output CSV/figure/summary artifacts")
    parser.add_argument(
        "--dataset_path",
        default=None,
        help="Optional dataset root override. Defaults to the dataset_path stored in the checkpoint config.",
    )
    parser.add_argument("--device", default="auto", help="Inference device: auto/cpu/cuda/cuda:0")
    parser.add_argument("--batch_size", type=int, default=256, help="Inference batch size")
    parser.add_argument(
        "--bycase",
        action="store_true",
        default=False,
        help="Generate per-case artifacts (excluding forgeEDA) under --bycase_output_dir",
    )
    parser.add_argument(
        "--bycase_output_dir",
        default="./moe_bycase",
        help="Root directory for per-case output when --bycase is set (default: ./moe_bycase)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) // 2),
        help="Parallel workers for JSON loading and artifact generation",
    )
    parser.add_argument(
        "--single_case",
        action="store_true",
        default=False,
        help="Generate artifacts for exactly one case specified by --dataset and --case_name",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Split directory name for --single_case mode (e.g. save_json_OpenABC)",
    )
    parser.add_argument(
        "--case_name",
        default=None,
        help="Case subdirectory name for --single_case mode (e.g. aes)",
    )
    parser.add_argument(
        "--export_buckets",
        default=None,
        help="Export global quantile bucket edges to this JSON file (uses training set of all 5 splits)",
    )
    parser.add_argument(
        "--bucket_file",
        default=None,
        help="Load pre-computed bucket edges from this JSON file instead of computing per-case",
    )
    return parser.parse_args(argv)


def resolve_scope_indices(num_samples: int, train_ratio: float, random_seed: int, scope: str) -> np.ndarray:
    if scope == "all":
        return np.arange(num_samples, dtype=np.int64)
    if scope != "val":
        raise ValueError(f"Unsupported scope: {scope}")

    rng = np.random.RandomState(random_seed)
    # load_student_data() already shuffles once before materializing prepared-data coordinates.
    # Replaying the trainer split therefore means applying the split permutation on top of that
    # prepared ordering rather than collapsing back to a single permutation in raw-dataset space.
    rng.permutation(num_samples)
    perm = rng.permutation(num_samples)
    train_size = max(1, int(num_samples * train_ratio))
    if num_samples > 1:
        train_size = min(train_size, num_samples - 1)
    val_indices = perm[train_size:]
    if val_indices.size == 0:
        val_indices = perm[:1]
    return val_indices.astype(np.int64, copy=False)


def _bucket_mask(y_raw: np.ndarray, lower: float, upper: float, include_lower: bool) -> np.ndarray:
    y_raw = np.asarray(y_raw, dtype=np.float64).reshape(-1)
    if np.isneginf(lower):
        return y_raw < upper if not include_lower else y_raw <= upper
    if include_lower:
        return (y_raw >= lower) & (y_raw <= upper)
    return (y_raw > lower) & (y_raw <= upper)


def compute_quantile_bucket_specs(
    y_raw: np.ndarray,
    n_quantiles: int = 4,
) -> list[tuple[str, float, float, bool]]:
    """Return bucket specs based on quantile boundaries of y_raw.

    Returns a list of n_quantiles tuples in the same format as LABEL_BUCKET_SPECS:
        (label, lower, upper, include_lower)
    The first bucket is closed on the left ([lower, upper]).
    All others are open on the left ((lower, upper]).
    """
    y = np.asarray(y_raw, dtype=np.float64).reshape(-1)
    percentiles = np.linspace(0.0, 100.0, n_quantiles + 1)
    edges = np.nanpercentile(y, percentiles)
    specs: list[tuple[str, float, float, bool]] = []
    for i in range(n_quantiles):
        lo = float(edges[i])
        hi = float(edges[i + 1])
        include_lower = i == 0
        if include_lower:
            label = f"Q{i + 1} [{lo:.1f}, {hi:.1f}]"
        else:
            label = f"Q{i + 1} ({lo:.1f}, {hi:.1f}]"
        specs.append((label, lo, hi, include_lower))
    return specs


def compute_heatmap_rows(
    y_raw: np.ndarray,
    expert_probs: np.ndarray,
    bucket_specs: list[tuple[str, float, float, bool]],
) -> list[dict[str, float | int | str]]:
    y_raw = np.asarray(y_raw, dtype=np.float64).reshape(-1)
    expert_probs = np.asarray(expert_probs, dtype=np.float64)
    if expert_probs.ndim != 2:
        raise ValueError("expert_probs must be a 2-D array")
    if y_raw.size != expert_probs.shape[0]:
        raise ValueError("y_raw and expert_probs must have the same number of samples")

    rows: list[dict[str, float | int | str]] = []
    num_experts = int(expert_probs.shape[1])
    for label, lower, upper, include_lower in bucket_specs:
        mask = _bucket_mask(y_raw, lower, upper, include_lower)
        row: dict[str, float | int | str] = {"bucket": label, "count": int(mask.sum())}
        if np.any(mask):
            mean_probs = expert_probs[mask].mean(axis=0)
        else:
            mean_probs = np.zeros(num_experts, dtype=np.float64)
        for idx, value in enumerate(mean_probs):
            row[f"expert_{idx}"] = float(value)
        rows.append(row)
    return rows


def compute_distribution_rows(
    y_raw: np.ndarray,
    selected_expert: np.ndarray,
    num_experts: int,
) -> list[dict[str, float | int]]:
    y_raw = np.asarray(y_raw, dtype=np.float64).reshape(-1)
    selected_expert = np.asarray(selected_expert, dtype=np.int64).reshape(-1)

    rows: list[dict[str, float | int]] = []
    for expert_id in range(num_experts):
        expert_delays = y_raw[selected_expert == expert_id]
        rows.append(
            {
                "expert_id": expert_id,
                "count": int(expert_delays.size),
                "delay_min": float(expert_delays.min()) if expert_delays.size else 0.0,
                "delay_max": float(expert_delays.max()) if expert_delays.size else 0.0,
                "delay_median": float(np.median(expert_delays)) if expert_delays.size else 0.0,
                "delay_mean": float(expert_delays.mean()) if expert_delays.size else 0.0,
            }
        )
    return rows


def collect_distribution_series(
    y_raw: np.ndarray,
    selected_expert: np.ndarray,
    num_experts: int,
) -> list[np.ndarray]:
    y_raw = np.asarray(y_raw, dtype=np.float64).reshape(-1)
    selected_expert = np.asarray(selected_expert, dtype=np.int64).reshape(-1)
    if y_raw.size != selected_expert.size:
        raise ValueError("y_raw and selected_expert must have the same length")
    return [y_raw[selected_expert == expert_id].copy() for expert_id in range(num_experts)]


def build_summary(
    checkpoint_path: str,
    scope: str,
    y_raw: np.ndarray,
    selected_expert: np.ndarray,
    num_experts: int,
    bucket_specs: list[tuple[str, float, float, bool]],
) -> dict[str, object]:
    y_raw = np.asarray(y_raw, dtype=np.float64).reshape(-1)
    selected_expert = np.asarray(selected_expert, dtype=np.int64).reshape(-1)
    if y_raw.size != selected_expert.size:
        raise ValueError("y_raw and selected_expert must have the same length")
    if np.any(selected_expert < 0) or np.any(selected_expert >= num_experts):
        raise ValueError("selected_expert values must be in [0, num_experts)")

    return {
        "checkpoint_path": checkpoint_path,
        "scope": scope,
        "num_samples": int(y_raw.size),
        "num_experts": int(num_experts),
        "bucket_labels": [label for label, *_rest in bucket_specs],
        "bucket_edges": [(float(lo), float(hi)) for _, lo, hi, _ in bucket_specs],
        "delay_range": {
            "min": float(np.min(y_raw)) if y_raw.size else 0.0,
            "max": float(np.max(y_raw)) if y_raw.size else 0.0,
        },
        "per_expert_sample_counts": {
            f"expert_{expert_id}": int(np.sum(selected_expert == expert_id))
            for expert_id in range(num_experts)
        },
    }


def write_csv_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("rows must not be empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_output_paths(output_dir: Path, scope: str) -> dict[str, Path]:
    return {
        "heatmap_csv": output_dir / f"student_moe_heatmap_{scope}.csv",
        "heatmap_plot": output_dir / f"student_moe_heatmap_{scope}",
        "distribution_csv": output_dir / f"student_moe_delay_distribution_{scope}.csv",
        "distribution_plot": output_dir / f"student_moe_delay_distribution_{scope}",
        "summary_json": output_dir / f"student_moe_summary_{scope}.json",
    }


def _save_figure(fig: plt.Figure, output_prefix: Path) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=400, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), dpi=400, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap_rows(rows: list[dict[str, float | int | str]], output_prefix: Path) -> None:
    rows = [row for row in rows if int(row["count"]) > 0]
    if not rows:
        return
    expert_keys = [key for key in rows[0] if key.startswith("expert_")]
    values = np.asarray([[float(row[key]) for key in expert_keys] for row in rows], dtype=np.float64)
    bucket_labels = [str(row["bucket"]) for row in rows]
    counts = [int(row["count"]) for row in rows]

    fig, ax = plt.subplots(1, 1, figsize=(8.8, 5.6), constrained_layout=True)
    value_max = float(np.max(values)) if values.size else 0.0
    image = ax.imshow(values, aspect="auto", cmap="Blues", vmin=0.0, vmax=max(1.0, value_max))
    ax.set_title("Student MoE Routing Specialization")
    ax.set_xlabel("Expert ID")
    ax.set_ylabel("Delay Bucket")
    ax.set_xticks(np.arange(len(expert_keys)))
    ax.set_xticklabels([key.replace("expert_", "E") for key in expert_keys], fontweight="bold")
    ax.set_yticks(np.arange(len(bucket_labels)))
    ax.set_yticklabels([f"{label} (n={count})" for label, count in zip(bucket_labels, counts)], fontweight="bold")

    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = values[row_idx, col_idx]
            ax.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", color="black", fontsize=10)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}"))
    colorbar.set_label("Mean routing probability", fontweight="bold")
    _save_figure(fig, output_prefix)


def plot_distribution_rows(
    rows: list[dict[str, float | int]],
    delay_series: list[np.ndarray],
    output_prefix: Path,
) -> None:
    expert_labels = [f"E{int(row['expert_id'])}" for row in rows]
    medians = [float(row["delay_median"]) for row in rows]
    positions = np.arange(len(rows))

    fig, ax = plt.subplots(1, 1, figsize=(8.8, 5.4), constrained_layout=True)
    violin_data = [series if series.size else np.array([0.0], dtype=np.float64) for series in delay_series]
    violin = ax.violinplot(violin_data, positions=positions, widths=0.8, showmeans=False, showmedians=False)
    for body, series in zip(violin["bodies"], delay_series):
        body.set_facecolor("#91CAE8")
        body.set_edgecolor("#000000")
        body.set_alpha(0.92 if series.size else 0.18)
    for key in ("cbars", "cmins", "cmaxes"):
        violin[key].set_color("#000000")
        violin[key].set_linewidth(1.1)

    ax.scatter(positions, medians, color="#C77C2E", s=28, zorder=3, label="Median delay")
    for pos, row in zip(positions, rows):
        ax.annotate(
            f"n={int(row['count'])}",
            xy=(pos, float(row["delay_median"])),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )

    ax.set_title("Per-Expert Delay Distribution")
    ax.set_xlabel("Expert ID")
    ax.set_ylabel("Delay")
    ax.set_xticks(positions)
    ax.set_xticklabels(expert_labels, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend(loc="upper right", frameon=False)

    _save_figure(fig, output_prefix)


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _load_checkpoint(checkpoint_path: Path) -> dict[str, object]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint {checkpoint_path} does not contain model_state_dict")
    return checkpoint


def _config_from_checkpoint(
    checkpoint: dict[str, object],
    checkpoint_path: Path,
    dataset_path_override: str | None,
) -> ExtractCutLibStudentConfig:
    config_raw = checkpoint.get("config")
    if not isinstance(config_raw, dict):
        raise ValueError(f"Checkpoint {checkpoint_path} does not contain a valid config payload")
    cfg = ExtractCutLibStudentConfig(**config_raw)
    if dataset_path_override:
        cfg = replace(cfg, dataset_path=dataset_path_override)
    return cfg


def _load_prepared_data(cfg: ExtractCutLibStudentConfig) -> StudentPreparedData:
    seed_all(cfg.random_seed)
    cache_available = bool(cfg.prepared_cache_dir) and Path(cfg.prepared_cache_dir).is_dir()
    if cfg.reuse_prepared_cache and cache_available:
        prepared = load_student_data_cache(cfg.prepared_cache_dir)
        return adapt_prepared_data_for_config(prepared, cfg)
    return load_student_data(cfg)


def _infer_teacher_hidden_dim(
    checkpoint: dict[str, object],
    cfg: ExtractCutLibStudentConfig,
) -> int | None:
    state_dict = checkpoint["model_state_dict"]
    if not isinstance(state_dict, dict):
        raise ValueError("model_state_dict must be a dict")
    adapter_weight = state_dict.get("teacher_align_adapter.weight")
    if adapter_weight is not None:
        return int(adapter_weight.shape[1])
    return cfg.teacher_hidden_dim


def upgrade_legacy_student_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    upgraded_state_dict: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        upgraded_key = key
        for legacy_prefix, modern_prefix in (
            ("group1_encoder.", "group_encoders.0."),
            ("group2_encoder.", "group_encoders.1."),
            ("group3_encoder.", "group_encoders.2."),
        ):
            if upgraded_key.startswith(legacy_prefix):
                upgraded_key = modern_prefix + upgraded_key[len(legacy_prefix) :]
                break
        upgraded_key = re.sub(r"^experts\.(\d+)\.0\.", r"experts.\1.head.0.", upgraded_key)
        upgraded_key = re.sub(r"^experts\.(\d+)\.2\.", r"experts.\1.head.2.", upgraded_key)
        upgraded_state_dict[upgraded_key] = value
    return upgraded_state_dict


def _load_model_from_checkpoint(
    checkpoint: dict[str, object],
    cfg: ExtractCutLibStudentConfig,
    device: torch.device,
) -> StudentModel:
    model = StudentModel(
        input_dim=cfg.input_dim,
        shared_hidden_dim=cfg.shared_hidden_dim,
        with_context8=cfg.with_context8,
        with_moe=cfg.with_moe,
        num_experts=cfg.num_experts,
        gate_temperature=cfg.gate_temperature,
        teacher_hidden_dim=_infer_teacher_hidden_dim(checkpoint, cfg),
    ).to(device)
    model.load_state_dict(upgrade_legacy_student_state_dict(checkpoint["model_state_dict"]), strict=True)
    model.eval()
    return model


def _run_inference(
    prepared: StudentPreparedData,
    indices: np.ndarray,
    model: StudentModel,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dataset = StudentDataset(prepared, indices=indices)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_student_batch,
    )

    y_raw_parts: list[np.ndarray] = []
    expert_prob_parts: list[np.ndarray] = []
    selected_expert_parts: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            output = model(batch["x_scaled"].to(device))
            if output.expert_probs is None or output.selected_expert is None:
                raise ValueError("Student checkpoint is not an MoE model; specialization plots require expert routing.")
            y_raw_parts.append(batch["y_raw"].detach().cpu().numpy())
            expert_prob_parts.append(output.expert_probs.detach().cpu().numpy())
            selected_expert_parts.append(output.selected_expert.detach().cpu().numpy())

    if not y_raw_parts:
        raise ValueError("No samples available for the requested scope")

    return (
        np.concatenate(y_raw_parts, axis=0).reshape(-1),
        np.concatenate(expert_prob_parts, axis=0),
        np.concatenate(selected_expert_parts, axis=0).reshape(-1),
    )


def _write_summary_json(path: Path, summary: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Per-case (--bycase) pipeline
# ---------------------------------------------------------------------------

_ALL_SPLIT_NAMES = (
    "save_json_OpenABC",
    "save_json_ACE",
    "save_json_iwls2022",
    "save_json_iwls2024",
    "save_json_forgeEDA",
)

_BYCASE_SPLIT_NAMES = (
    "save_json_OpenABC",
    "save_json_ACE",
    "save_json_iwls2022",
    "save_json_iwls2024",
)

_BYCASE_MIN_SAMPLES = 5


def save_bucket_specs(path: Path, bucket_specs: list[tuple[str, float, float, bool]]) -> None:
    """Save bucket specs to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_quantiles": len(bucket_specs),
        "bucket_specs": [[label, lo, hi, inc] for label, lo, hi, inc in bucket_specs],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("Saved %d bucket specs to %s", len(bucket_specs), path)


def load_bucket_specs(path: Path) -> list[tuple[str, float, float, bool]]:
    """Load bucket specs from JSON file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [(label, float(lo), float(hi), bool(inc)) for label, lo, hi, inc in payload["bucket_specs"]]


def _load_single_json_case_wrapper(json_path: str) -> tuple[str, object, object, object, int]:
    try:
        return ext_utils._load_single_json_case(json_path, False)
    except Exception as exc:
        raise RuntimeError(f"Failed to load {json_path}") from exc


def _load_cases_with_labels(
    dataset_path: str,
    split_names: tuple[str, ...],
    num_workers: int,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Load raw data per case from JSON files in parallel.

    Returns a list of ``(case_name, x_batch, y_batch)`` where
    ``case_name = "{split_dir}/{case_dir}"``. Cases with no valid samples are skipped.
    """
    from concurrent.futures import ProcessPoolExecutor

    json_paths: list[tuple[str, str]] = []
    for split in split_names:
        split_dir = os.path.join(dataset_path, split)
        if not os.path.isdir(split_dir):
            logger.warning("split directory not found, skipping: %s", split_dir)
            continue
        for case_dir in sorted(os.listdir(split_dir)):
            json_path = os.path.join(split_dir, case_dir, "data_2.json")
            if os.path.isfile(json_path):
                json_paths.append((f"{split}/{case_dir}", json_path))

    if not json_paths:
        return []

    case_names = [cn for cn, _ in json_paths]
    raw_json_paths = [jp for _, jp in json_paths]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        raw_results = list(executor.map(_load_single_json_case_wrapper, raw_json_paths))

    results: list[tuple[str, np.ndarray, np.ndarray]] = []
    for case_name, (_, x_batch, y_batch, _, count) in zip(case_names, raw_results):
        if x_batch is None or count == 0:
            continue
        results.append((case_name, x_batch, y_batch.reshape(-1, 1)))
    return results


def _build_prepared_from_cases(
    cases: list[tuple[str, np.ndarray, np.ndarray]],
    checkpoint: dict[str, object],
    cfg: ExtractCutLibStudentConfig,
) -> tuple[StudentPreparedData, list[str], np.ndarray]:
    """Concatenate per-case data and standardize using checkpoint normalization.

    Returns ``(prepared, case_names, case_labels)`` where ``case_labels`` is an
    int array of shape ``[N_total]`` mapping each sample to its index in
    ``case_names``.
    """
    x_mu = np.asarray(checkpoint["x_mu"], dtype=np.float32)
    x_inv_sigma = np.asarray(checkpoint["x_inv_sigma"], dtype=np.float32)
    y_mu = np.asarray(checkpoint["y_mu"], dtype=np.float32)
    y_inv_sigma = np.asarray(checkpoint["y_inv_sigma"], dtype=np.float32)

    case_names: list[str] = []
    case_labels_parts: list[np.ndarray] = []
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []

    for case_idx, (case_name, x_batch, y_batch) in enumerate(cases):
        case_names.append(case_name)
        x_parts.append(x_batch)
        y_parts.append(y_batch)
        case_labels_parts.append(np.full(x_batch.shape[0], case_idx, dtype=np.int64))

    x_all = np.concatenate(x_parts, axis=0)
    y_all = np.concatenate(y_parts, axis=0).reshape(-1, 1)
    case_labels = np.concatenate(case_labels_parts, axis=0)

    # Extract continuous features matching training pipeline
    x_cont_full = x_all[:, :FEATURE_INPUT_DIM].astype(np.float32)
    x_cont = x_cont_full if cfg.with_context8 else x_cont_full[:, CONTEXT_FEATURE_DIM:]
    x_wl_hash = x_all[:, -1].astype(np.int64)

    # Apply checkpoint normalization (same stats used during training)
    x_scaled = (x_cont - x_mu) * x_inv_sigma

    # y: log1p then checkpoint z-score
    y_raw = y_all.astype(np.float32)
    y_log = np.log1p(y_raw.astype(np.float64)).astype(np.float32)
    y_scaled = (y_log - y_mu) * y_inv_sigma

    prepared = StudentPreparedData(
        x_raw=x_cont,
        x_scaled=x_scaled,
        x_wl_hash=x_wl_hash,
        x_mu=x_mu,
        x_inv_sigma=x_inv_sigma,
        y_scaled=y_scaled,
        y_raw=y_raw,
        y_mu=y_mu,
        y_inv_sigma=y_inv_sigma,
    )
    return prepared, case_names, case_labels


def _generate_case_artifacts(
    case_name: str,
    y_raw: np.ndarray,
    expert_probs: np.ndarray,
    selected_expert: np.ndarray,
    num_experts: int,
    checkpoint_path: str,
    bycase_root: Path,
    bucket_specs: list[tuple[str, float, float, bool]],
) -> bool:
    """Compute and save all artifacts for a single case.

    Returns True if artifacts were written, False if the case was skipped.
    """
    if y_raw.size < _BYCASE_MIN_SAMPLES:
        logger.warning("Skipping %s: only %d samples (minimum %d)", case_name, y_raw.size, _BYCASE_MIN_SAMPLES)
        return False

    apply_publication_style()   # runs in each subprocess

    sanitized = case_name.replace("/", "_")
    case_output_dir = bycase_root / sanitized
    output_paths = build_output_paths(case_output_dir, scope="bycase")

    heatmap_rows = compute_heatmap_rows(y_raw=y_raw, expert_probs=expert_probs, bucket_specs=bucket_specs)
    distribution_rows = compute_distribution_rows(
        y_raw=y_raw, selected_expert=selected_expert, num_experts=num_experts
    )
    distribution_series = collect_distribution_series(
        y_raw=y_raw, selected_expert=selected_expert, num_experts=num_experts
    )
    summary = build_summary(
        checkpoint_path=checkpoint_path,
        scope=f"bycase:{case_name}",
        y_raw=y_raw,
        selected_expert=selected_expert,
        num_experts=num_experts,
        bucket_specs=bucket_specs,
    )

    write_csv_rows(output_paths["heatmap_csv"], heatmap_rows)
    write_csv_rows(output_paths["distribution_csv"], distribution_rows)
    _write_summary_json(output_paths["summary_json"], summary)
    plot_heatmap_rows(heatmap_rows, output_paths["heatmap_plot"])
    plot_distribution_rows(distribution_rows, distribution_series, output_paths["distribution_plot"])
    return True


def _generate_case_artifacts_star(
    args: tuple[str, np.ndarray, np.ndarray, np.ndarray, int, str, Path, list[tuple[str, float, float, bool]]],
) -> bool:
    return _generate_case_artifacts(*args)


def _main_bycase_impl(args: argparse.Namespace) -> int:
    checkpoint_path = Path(args.checkpoint)
    bycase_root = Path(args.bycase_output_dir)
    device = _resolve_device(args.device)

    checkpoint = _load_checkpoint(checkpoint_path)
    cfg = _config_from_checkpoint(checkpoint, checkpoint_path, args.dataset_path)
    model = _load_model_from_checkpoint(checkpoint, cfg, device)

    logger.info("Loading cases from %s (splits: %s)", cfg.dataset_path, _BYCASE_SPLIT_NAMES)
    cases = _load_cases_with_labels(cfg.dataset_path, _BYCASE_SPLIT_NAMES, num_workers=args.num_workers)
    if not cases:
        raise ValueError(f"No valid cases found under {cfg.dataset_path}")
    logger.info("Loaded %d cases, %s total samples", len(cases), f"{sum(x.shape[0] for _, x, _ in cases):,}")

    prepared, case_names, case_labels = _build_prepared_from_cases(cases, checkpoint, cfg)
    all_indices = np.arange(prepared.y_raw.shape[0], dtype=np.int64)

    logger.info("Running inference on %s samples...", f"{all_indices.size:,}")
    y_raw, expert_probs, selected_expert = _run_inference(
        prepared=prepared,
        indices=all_indices,
        model=model,
        device=device,
        batch_size=args.batch_size,
    )

    from concurrent.futures import ProcessPoolExecutor

    apply_publication_style()  # validate scienceplots is available before forking
    num_experts = int(expert_probs.shape[1])
    outlier_mask = _build_label_outlier_mask(y_raw, cfg.label_outlier_filter, cfg.label_outlier_threshold)
    bucket_specs = compute_quantile_bucket_specs(y_raw[outlier_mask])
    logger.info(
        "Quantile bucket edges (from %d / %d non-outlier samples): %s",
        int(outlier_mask.sum()), y_raw.size,
        [(spec[0], f"{spec[1]:.1f}", f"{spec[2]:.1f}") for spec in bucket_specs],
    )
    args_list = []
    for case_idx, case_name in enumerate(case_names):
        mask = case_labels == case_idx
        args_list.append((
            case_name,
            y_raw[mask].copy(),
            expert_probs[mask].copy(),
            selected_expert[mask].copy(),
            num_experts,
            str(checkpoint_path),
            bycase_root,
            bucket_specs,
        ))

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        results = list(executor.map(_generate_case_artifacts_star, args_list))

    written = sum(1 for r in results if r)
    skipped = sum(1 for r in results if not r)
    logger.info("Done: %d cases written, %d skipped. Output root: %s", written, skipped, bycase_root)
    return 0


def _main_single_case_impl(args: argparse.Namespace) -> int:
    if not args.dataset:
        raise ValueError("--dataset is required in --single_case mode")
    if not args.case_name:
        raise ValueError("--case_name is required in --single_case mode")

    apply_publication_style()  # fast-fail before any I/O

    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    device = _resolve_device(args.device)

    checkpoint = _load_checkpoint(checkpoint_path)
    cfg = _config_from_checkpoint(checkpoint, checkpoint_path, args.dataset_path)
    model = _load_model_from_checkpoint(checkpoint, cfg, device)

    json_path = os.path.join(cfg.dataset_path, args.dataset, args.case_name, "data_2.json")
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"data_2.json not found: {json_path}")

    case_name = f"{args.dataset}/{args.case_name}"
    logger.info("Loading single case: %s", case_name)
    _, x_batch, y_batch, _, count = ext_utils._load_single_json_case(json_path, False)
    if x_batch is None or count == 0:
        raise ValueError(f"No valid cuts found in {json_path}")

    prepared, _, _ = _build_prepared_from_cases(
        [(case_name, x_batch, y_batch.reshape(-1, 1))],
        checkpoint,
        cfg,
    )
    all_indices = np.arange(prepared.y_raw.shape[0], dtype=np.int64)
    logger.info("Running inference on %s samples...", f"{all_indices.size:,}")
    y_raw, expert_probs, selected_expert = _run_inference(
        prepared=prepared,
        indices=all_indices,
        model=model,
        device=device,
        batch_size=args.batch_size,
    )

    if args.bucket_file and Path(args.bucket_file).is_file():
        bucket_specs = load_bucket_specs(Path(args.bucket_file))
        logger.info("Loaded global bucket specs from %s", args.bucket_file)
    else:
        outlier_mask = _build_label_outlier_mask(y_raw, cfg.label_outlier_filter, cfg.label_outlier_threshold)
        bucket_specs = compute_quantile_bucket_specs(y_raw[outlier_mask])
    ok = _generate_case_artifacts(
        case_name=case_name,
        y_raw=y_raw,
        expert_probs=expert_probs,
        selected_expert=selected_expert,
        num_experts=int(expert_probs.shape[1]),
        checkpoint_path=str(checkpoint_path),
        bycase_root=output_dir,
        bucket_specs=bucket_specs,
    )
    if not ok:
        logger.warning("Case %s was skipped (too few samples). No artifacts written.", case_name)
        return 1
    logger.info("Done. Output: %s", output_dir / case_name.replace("/", "_"))
    return 0


def _main_export_buckets_impl(args: argparse.Namespace) -> int:
    """Load training set of all 5 splits, filter outliers, compute global quantile buckets, save to file."""
    checkpoint_path = Path(args.checkpoint)
    export_path = Path(args.export_buckets)

    checkpoint = _load_checkpoint(checkpoint_path)
    cfg = _config_from_checkpoint(checkpoint, checkpoint_path, args.dataset_path)

    logger.info("Loading all 5 splits from %s for global bucket computation", cfg.dataset_path)
    cases = _load_cases_with_labels(cfg.dataset_path, _ALL_SPLIT_NAMES, num_workers=args.num_workers)
    if not cases:
        raise ValueError(f"No valid cases found under {cfg.dataset_path}")

    total_samples = sum(x.shape[0] for _, x, _ in cases)
    logger.info("Loaded %d cases, %s total samples", len(cases), f"{total_samples:,}")

    # Concatenate all y_raw
    y_all = np.concatenate([y.reshape(-1) for _, _, y in cases], axis=0).astype(np.float64)

    # Apply train split to match training distribution
    num_samples = y_all.shape[0]
    rng = np.random.RandomState(cfg.random_seed)
    rng.permutation(num_samples)
    perm = rng.permutation(num_samples)
    train_size = max(1, int(num_samples * cfg.train_ratio))
    if num_samples > 1:
        train_size = min(train_size, num_samples - 1)
    train_indices = perm[:train_size]
    y_train = y_all[train_indices]
    logger.info("Train split: %d / %d samples (ratio=%.2f)", train_size, num_samples, cfg.train_ratio)

    # Filter outliers
    outlier_mask = _build_label_outlier_mask(y_train, cfg.label_outlier_filter, cfg.label_outlier_threshold)
    y_filtered = y_train[outlier_mask]
    logger.info("After outlier filter: %d / %d samples", y_filtered.shape[0], y_train.shape[0])

    bucket_specs = compute_quantile_bucket_specs(y_filtered)
    logger.info(
        "Global quantile bucket edges: %s",
        [(spec[0], f"{spec[1]:.1f}", f"{spec[2]:.1f}") for spec in bucket_specs],
    )

    save_bucket_specs(export_path, bucket_specs)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    logger.info("CurPID: %d", os.getpid())
    args = parse_args(argv)
    if args.export_buckets:
        return _main_export_buckets_impl(args)
    if args.single_case and args.bycase:
        logger.warning("Both --single_case and --bycase set; using --single_case")
    if args.single_case:
        return _main_single_case_impl(args)
    if args.bycase:
        return _main_bycase_impl(args)
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    device = _resolve_device(args.device)

    checkpoint = _load_checkpoint(checkpoint_path)
    cfg = _config_from_checkpoint(checkpoint, checkpoint_path, args.dataset_path)
    prepared = _load_prepared_data(cfg)
    indices = resolve_scope_indices(
        num_samples=int(prepared.y_raw.shape[0]),
        train_ratio=cfg.train_ratio,
        random_seed=cfg.random_seed,
        scope=args.scope,
    )
    model = _load_model_from_checkpoint(checkpoint, cfg, device)
    y_raw, expert_probs, selected_expert = _run_inference(
        prepared=prepared,
        indices=indices,
        model=model,
        device=device,
        batch_size=args.batch_size,
    )

    apply_publication_style()
    bucket_specs = compute_quantile_bucket_specs(y_raw)
    logger.info(
        "Quantile bucket edges: %s",
        [(spec[0], f"{spec[1]:.1f}", f"{spec[2]:.1f}") for spec in bucket_specs],
    )
    heatmap_rows = compute_heatmap_rows(y_raw=y_raw, expert_probs=expert_probs, bucket_specs=bucket_specs)
    distribution_rows = compute_distribution_rows(
        y_raw=y_raw,
        selected_expert=selected_expert,
        num_experts=expert_probs.shape[1],
    )
    summary = build_summary(
        checkpoint_path=str(checkpoint_path),
        scope=args.scope,
        y_raw=y_raw,
        selected_expert=selected_expert,
        num_experts=expert_probs.shape[1],
        bucket_specs=bucket_specs,
    )
    distribution_series = collect_distribution_series(
        y_raw=y_raw,
        selected_expert=selected_expert,
        num_experts=expert_probs.shape[1],
    )
    output_paths = build_output_paths(output_dir=output_dir, scope=args.scope)

    write_csv_rows(output_paths["heatmap_csv"], heatmap_rows)
    write_csv_rows(output_paths["distribution_csv"], distribution_rows)
    _write_summary_json(output_paths["summary_json"], summary)
    plot_heatmap_rows(heatmap_rows, output_paths["heatmap_plot"])
    plot_distribution_rows(distribution_rows, distribution_series, output_paths["distribution_plot"])

    logger.info("Saved heatmap CSV to %s", output_paths['heatmap_csv'])
    logger.info("Saved distribution CSV to %s", output_paths['distribution_csv'])
    logger.info("Saved summary JSON to %s", output_paths['summary_json'])
    logger.info("Saved heatmap figures to %s and %s", output_paths['heatmap_plot'].with_suffix('.png'), output_paths['heatmap_plot'].with_suffix('.pdf'))
    logger.info("Saved distribution figures to %s and %s", output_paths['distribution_plot'].with_suffix('.png'), output_paths['distribution_plot'].with_suffix('.pdf'))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
