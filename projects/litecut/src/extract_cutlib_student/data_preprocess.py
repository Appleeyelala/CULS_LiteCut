from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.cross_decomposition import PLSRegression

from subutils import extraction_utils as ext_utils

from extract_cutlib_teacher.feature_branch import FEATURE_INPUT_DIM
from extract_cutlib_teacher.preprocess import fit_pls_projection, standardize_features

from .config import ExtractCutLibStudentConfig

CONTEXT_FEATURE_DIM = 8


@dataclass
class StudentPreparedData:
    x_raw: np.ndarray
    x_scaled: np.ndarray
    x_wl_hash: np.ndarray
    x_mu: np.ndarray
    x_inv_sigma: np.ndarray
    y_scaled: np.ndarray
    y_raw: np.ndarray
    y_mu: np.ndarray
    y_inv_sigma: np.ndarray
    teacher_y_scaled: np.ndarray | None = None
    teacher_hidden: np.ndarray | None = None


def adapt_prepared_data_for_config(
    prepared: StudentPreparedData,
    cfg: ExtractCutLibStudentConfig,
) -> StudentPreparedData:
    if prepared.x_raw.shape[1] not in (FEATURE_INPUT_DIM, cfg.input_dim):
        raise ValueError(
            f"Unsupported cached input dim {prepared.x_raw.shape[1]}; expected full {FEATURE_INPUT_DIM} or target {cfg.input_dim}"
        )

    if prepared.x_raw.shape[1] == FEATURE_INPUT_DIM and not cfg.with_context8:
        x_raw = prepared.x_raw[:, CONTEXT_FEATURE_DIM:]
        x_scaled = prepared.x_scaled[:, CONTEXT_FEATURE_DIM:]
        x_mu = prepared.x_mu[CONTEXT_FEATURE_DIM:]
        x_inv_sigma = prepared.x_inv_sigma[CONTEXT_FEATURE_DIM:]
    else:
        x_raw = prepared.x_raw
        x_scaled = prepared.x_scaled
        x_mu = prepared.x_mu
        x_inv_sigma = prepared.x_inv_sigma

    if x_raw.shape[1] != cfg.input_dim:
        raise ValueError(
            f"Prepared input_dim={x_raw.shape[1]} does not match cfg.input_dim={cfg.input_dim} after runtime adaptation."
        )

    teacher_y_scaled = prepared.teacher_y_scaled if cfg.with_distill else None
    teacher_hidden = prepared.teacher_hidden if cfg.with_distill else None

    return StudentPreparedData(
        x_raw=x_raw.astype(np.float32, copy=False),
        x_scaled=x_scaled.astype(np.float32, copy=False),
        x_wl_hash=prepared.x_wl_hash,
        x_mu=x_mu.astype(np.float32, copy=False),
        x_inv_sigma=x_inv_sigma.astype(np.float32, copy=False),
        y_scaled=prepared.y_scaled,
        y_raw=prepared.y_raw,
        y_mu=prepared.y_mu,
        y_inv_sigma=prepared.y_inv_sigma,
        teacher_y_scaled=teacher_y_scaled,
        teacher_hidden=teacher_hidden,
    )


class StudentDataset(Dataset):
    def __init__(self, prepared: StudentPreparedData, indices: np.ndarray | None = None) -> None:
        self.prepared = prepared
        self.indices = np.arange(prepared.x_scaled.shape[0]) if indices is None else np.asarray(indices)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample_idx = int(self.indices[idx])
        item = {
            "x_raw": torch.from_numpy(self.prepared.x_raw[sample_idx]).float(),
            "x_scaled": torch.from_numpy(self.prepared.x_scaled[sample_idx]).float(),
            "wl_hash": torch.tensor(self.prepared.x_wl_hash[sample_idx], dtype=torch.long),
            "y_scaled": torch.from_numpy(self.prepared.y_scaled[sample_idx]).float(),
            "y_raw": torch.from_numpy(self.prepared.y_raw[sample_idx]).float(),
        }
        if self.prepared.teacher_y_scaled is not None:
            item["teacher_y_scaled"] = torch.from_numpy(self.prepared.teacher_y_scaled[sample_idx]).float()
        else:
            item["teacher_y_scaled"] = None
        if self.prepared.teacher_hidden is not None:
            item["teacher_hidden"] = torch.from_numpy(self.prepared.teacher_hidden[sample_idx]).float()
        else:
            item["teacher_hidden"] = None
        return item


def collate_student_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    teacher_y_items = [item["teacher_y_scaled"] for item in batch]
    teacher_hidden_items = [item["teacher_hidden"] for item in batch]
    return {
        "x_raw": torch.stack([item["x_raw"] for item in batch], dim=0),
        "x_scaled": torch.stack([item["x_scaled"] for item in batch], dim=0),
        "wl_hash": torch.stack([item["wl_hash"] for item in batch], dim=0),
        "y_scaled": torch.stack([item["y_scaled"] for item in batch], dim=0),
        "y_raw": torch.stack([item["y_raw"] for item in batch], dim=0),
        "teacher_y_scaled": None if any(item is None for item in teacher_y_items) else torch.stack(teacher_y_items, dim=0),
        "teacher_hidden": None
        if any(item is None for item in teacher_hidden_items)
        else torch.stack(teacher_hidden_items, dim=0),
    }


def _log1p_transform_y(y: np.ndarray) -> np.ndarray:
    if np.any(y <= -1.0):
        y_min = float(np.min(y))
        raise ValueError(f"y contains values <= -1, cannot apply log1p safely. min(y)={y_min:.6f}")
    return np.log1p(y)


def _build_label_outlier_mask(y: np.ndarray, mode: str, threshold: float) -> np.ndarray:
    y_flat = y.reshape(-1).astype(np.float64)

    if threshold <= 0:
        raise ValueError(f"label_outlier_threshold must be > 0, got {threshold}")

    if mode == "none":
        return np.ones_like(y_flat, dtype=bool)

    if mode == "zscore":
        mu = y_flat.mean()
        sigma = y_flat.std()
        if sigma < 1e-12:
            return np.ones_like(y_flat, dtype=bool)
        score = np.abs((y_flat - mu) / sigma)
    elif mode == "mad":
        median = np.median(y_flat)
        mad = np.median(np.abs(y_flat - median))
        if mad < 1e-12:
            return np.ones_like(y_flat, dtype=bool)
        score = 0.6745 * np.abs(y_flat - median) / mad
    elif mode == "maximum":
        score = np.abs(y_flat)
    else:
        raise ValueError(f"Unsupported label_outlier_filter: {mode}")

    return score <= threshold


def _load_dataset_split(data_root: str, split_name: str, max_workers: int) -> tuple[np.ndarray, np.ndarray]:
    data_path = os.path.join(data_root, split_name)
    x_split, _, y_split, _ = ext_utils.load_json_data(
        data_path,
        logger=None,
        include_graph=False,
        max_workers=max_workers,
    )
    return x_split, y_split


def load_student_data(cfg: ExtractCutLibStudentConfig) -> StudentPreparedData:
    split_names = (
        "save_json_OpenABC",
        "save_json_ACE",
        "save_json_iwls2022",
        "save_json_iwls2024",
        "save_json_forgeEDA",
    )

    x_parts = []
    y_parts = []
    for split_name in split_names:
        x_split, y_split = _load_dataset_split(cfg.dataset_path, split_name, max_workers=cfg.cpu_worker)
        if x_split.size == 0:
            continue
        x_parts.append(x_split)
        y_parts.append(y_split)

    if not x_parts:
        raise ValueError(f"No valid student samples were loaded from dataset_path={cfg.dataset_path}")

    x_all = np.concatenate(x_parts, axis=0)
    y_all = np.concatenate(y_parts, axis=0).reshape(-1, 1)

    keep_mask = _build_label_outlier_mask(y_all, cfg.label_outlier_filter, cfg.label_outlier_threshold)
    x_all = x_all[keep_mask]
    y_all = y_all[keep_mask]

    perm = np.random.permutation(len(y_all))
    x_all = x_all[perm]
    y_all = y_all[perm]

    x_cont_full = x_all[:, :FEATURE_INPUT_DIM]
    if cfg.with_context8:
        x_cont = x_cont_full
    else:
        x_cont = x_cont_full[:, CONTEXT_FEATURE_DIM:]
    x_wl_hash = x_all[:, -1].astype(np.int64)
    if x_cont.shape[1] != cfg.input_dim:
        raise ValueError(
            f"Expected {cfg.input_dim} student continuous features, got {x_cont.shape[1]}. "
            "Please verify exported feature order and Student input_dim."
        )

    x_scaled, x_mu, x_inv_sigma = standardize_features(x_cont)
    y_log = _log1p_transform_y(y_all)
    y_scaled, y_mu, y_inv_sigma = standardize_features(y_log)

    return StudentPreparedData(
        x_raw=x_cont.astype(np.float32),
        x_scaled=x_scaled.astype(np.float32),
        x_wl_hash=x_wl_hash,
        x_mu=x_mu.astype(np.float32),
        x_inv_sigma=x_inv_sigma.astype(np.float32),
        y_scaled=y_scaled.astype(np.float32),
        y_raw=y_all.astype(np.float32),
        y_mu=y_mu.astype(np.float32),
        y_inv_sigma=y_inv_sigma.astype(np.float32),
    )


def save_student_data_cache(prepared: StudentPreparedData, cache_dir: str) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    np.save(os.path.join(cache_dir, "x_raw.npy"), prepared.x_raw, allow_pickle=False)
    np.save(os.path.join(cache_dir, "x_scaled.npy"), prepared.x_scaled, allow_pickle=False)
    np.save(os.path.join(cache_dir, "x_wl_hash.npy"), prepared.x_wl_hash, allow_pickle=False)
    np.save(os.path.join(cache_dir, "x_mu.npy"), prepared.x_mu, allow_pickle=False)
    np.save(os.path.join(cache_dir, "x_inv_sigma.npy"), prepared.x_inv_sigma, allow_pickle=False)
    np.save(os.path.join(cache_dir, "y_scaled.npy"), prepared.y_scaled, allow_pickle=False)
    np.save(os.path.join(cache_dir, "y_raw.npy"), prepared.y_raw, allow_pickle=False)
    np.save(os.path.join(cache_dir, "y_mu.npy"), prepared.y_mu, allow_pickle=False)
    np.save(os.path.join(cache_dir, "y_inv_sigma.npy"), prepared.y_inv_sigma, allow_pickle=False)
    meta = {"input_dim": int(prepared.x_raw.shape[1])}
    with open(os.path.join(cache_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)

    if prepared.teacher_y_scaled is not None:
        np.save(os.path.join(cache_dir, "teacher_y_scaled.npy"), prepared.teacher_y_scaled, allow_pickle=False)
    if prepared.teacher_hidden is not None:
        np.save(os.path.join(cache_dir, "teacher_hidden.npy"), prepared.teacher_hidden, allow_pickle=False)


def load_student_data_cache(cache_dir: str) -> StudentPreparedData:
    teacher_y_scaled_path = os.path.join(cache_dir, "teacher_y_scaled.npy")
    teacher_hidden_path = os.path.join(cache_dir, "teacher_hidden.npy")
    meta_path = os.path.join(cache_dir, "meta.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    x_raw = np.load(os.path.join(cache_dir, "x_raw.npy"))
    expected_input_dim = int(meta.get("input_dim", x_raw.shape[1]))
    if x_raw.shape[1] != expected_input_dim:
        raise ValueError(
            f"Student cache input_dim mismatch: meta={expected_input_dim}, data={x_raw.shape[1]} in {cache_dir}"
        )

    return StudentPreparedData(
        x_raw=x_raw,
        x_scaled=np.load(os.path.join(cache_dir, "x_scaled.npy")),
        x_wl_hash=np.load(os.path.join(cache_dir, "x_wl_hash.npy")),
        x_mu=np.load(os.path.join(cache_dir, "x_mu.npy")),
        x_inv_sigma=np.load(os.path.join(cache_dir, "x_inv_sigma.npy")),
        y_scaled=np.load(os.path.join(cache_dir, "y_scaled.npy")),
        y_raw=np.load(os.path.join(cache_dir, "y_raw.npy")),
        y_mu=np.load(os.path.join(cache_dir, "y_mu.npy")),
        y_inv_sigma=np.load(os.path.join(cache_dir, "y_inv_sigma.npy")),
        teacher_y_scaled=np.load(teacher_y_scaled_path) if os.path.exists(teacher_y_scaled_path) else None,
        teacher_hidden=np.load(teacher_hidden_path) if os.path.exists(teacher_hidden_path) else None,
    )


def collect_teacher_feature_branch_outputs(
    prepared: StudentPreparedData,
    teacher_model: torch.nn.Module,
    batch_size: int = 1024,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    _, features = collect_teacher_all_outputs(prepared, teacher_model, batch_size, device)
    return features


def collect_teacher_predictions(
    prepared: StudentPreparedData,
    teacher_model: torch.nn.Module,
    batch_size: int = 1024,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    predictions, _ = collect_teacher_all_outputs(prepared, teacher_model, batch_size, device)
    return predictions


def collect_teacher_all_outputs(
    prepared: StudentPreparedData,
    teacher_model: torch.nn.Module,
    batch_size: int = 1024,
    device: str | torch.device = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Single-pass collection of both teacher predictions and feature branch outputs.

    Replaces two separate forward passes over the same data with one, roughly
    halving inference time.
    """
    teacher_model = teacher_model.to(device)
    teacher_model.eval()

    if not hasattr(teacher_model, "forward"):
        raise AttributeError("teacher_model must expose `forward(x_scaled, wl_hash, graph_batch)`.")
    if not (hasattr(teacher_model, "forward_feature") or hasattr(teacher_model, "feature_branch")):
        raise AttributeError(
            "teacher_model must expose `forward_feature(x_scaled, wl_hash)` or `feature_branch`."
        )

    predictions, features = [], []
    with torch.inference_mode():
        for start in range(0, prepared.x_scaled.shape[0], batch_size):
            end = min(start + batch_size, prepared.x_scaled.shape[0])
            x_scaled = torch.from_numpy(prepared.x_scaled[start:end]).float().to(device)
            wl_hash = torch.from_numpy(prepared.x_wl_hash[start:end]).long().to(device)

            y_pred, _, _ = teacher_model(x_scaled, wl_hash, None)
            predictions.append(y_pred.detach().cpu().numpy())

            if hasattr(teacher_model, "forward_feature"):
                z_feat = teacher_model.forward_feature(x_scaled, wl_hash)
            else:
                z_feat = teacher_model.feature_branch(x_scaled, wl_hash)
            features.append(z_feat.detach().cpu().numpy())

    return (
        np.concatenate(predictions, axis=0).astype(np.float32),
        np.concatenate(features, axis=0).astype(np.float32),
    )


def fit_teacher_feature_pls(
    teacher_features: np.ndarray,
    labels: np.ndarray,
    latent_dim: int,
) -> tuple[PLSRegression, np.ndarray]:
    max_latent_dim = min(teacher_features.shape[0], teacher_features.shape[1], latent_dim)
    if max_latent_dim <= 0:
        raise ValueError(
            f"Invalid PLS latent dim: features={teacher_features.shape}, requested_latent_dim={latent_dim}"
        )
    pls, teacher_hidden = fit_pls_projection(teacher_features, labels, latent_dim=max_latent_dim)
    teacher_hidden_scaled, _, _ = standardize_features(teacher_hidden)
    return pls, teacher_hidden_scaled.astype(np.float32)


def attach_teacher_pls_targets(
    prepared: StudentPreparedData,
    teacher_model: torch.nn.Module,
    latent_dim: int,
    batch_size: int = 1024,
    device: str | torch.device = "cpu",
    labels_for_pls: np.ndarray | None = None,
    teacher_y_scaled: np.ndarray | None = None,
) -> tuple[StudentPreparedData, PLSRegression]:
    if teacher_y_scaled is None:
        # Single pass: collect predictions and features simultaneously
        teacher_y_scaled, teacher_features = collect_teacher_all_outputs(
            prepared=prepared,
            teacher_model=teacher_model,
            batch_size=batch_size,
            device=device,
        )
    else:
        # predictions already provided; only need features
        teacher_features = collect_teacher_feature_branch_outputs(
            prepared=prepared,
            teacher_model=teacher_model,
            batch_size=batch_size,
            device=device,
        )
    pls_labels = prepared.y_scaled if labels_for_pls is None else labels_for_pls
    pls, teacher_hidden = fit_teacher_feature_pls(
        teacher_features=teacher_features,
        labels=pls_labels,
        latent_dim=latent_dim,
    )
    prepared.teacher_hidden = teacher_hidden
    if teacher_y_scaled is not None:
        prepared.teacher_y_scaled = teacher_y_scaled.astype(np.float32)
    return prepared, pls
