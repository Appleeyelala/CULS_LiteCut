from __future__ import annotations

import gc
import json
import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from subutils import extraction_utils as ext_utils

from .config import ExtractCutLibConfig
from .feature_branch import FEATURE_INPUT_DIM
from .graph_branch import GraphTensorSample, NODE_FEAT_DIM
from .logger import get_logger
from .preprocess import standardize_features

try:
    import ujson as fast_json
except ImportError:
    fast_json = json


@dataclass
class TeacherPreparedData:
    x_raw: np.ndarray | None
    x_scaled: np.ndarray
    x_wl_hash: np.ndarray
    x_graph: tuple[GraphTensorSample, ...] | None
    y_scaled: np.ndarray
    y_raw: np.ndarray
    y_mu: np.ndarray
    y_inv_sigma: np.ndarray


def _pack_graph_samples(x_graph: tuple[GraphTensorSample, ...]) -> dict[str, list]:
    return {
        "node_feats": [sample.node_feats for sample in x_graph],
        "edge_index": [sample.edge_index for sample in x_graph],
        "edge_type": [sample.edge_type for sample in x_graph],
        "pos_vec": [sample.pos_vec for sample in x_graph],
        "root_index": [sample.root_index for sample in x_graph],
        "leaf_mask": [sample.leaf_mask for sample in x_graph],
    }


def _unpack_graph_samples(payload: dict[str, list]) -> tuple[GraphTensorSample, ...]:
    num_graphs = len(payload["node_feats"])
    return tuple(
        GraphTensorSample(
            node_feats=payload["node_feats"][idx],
            edge_index=payload["edge_index"][idx],
            edge_type=payload["edge_type"][idx],
            pos_vec=payload["pos_vec"][idx],
            root_index=int(payload["root_index"][idx]),
            leaf_mask=payload["leaf_mask"][idx],
        )
        for idx in range(num_graphs)
    )


def save_teacher_data_cache(prepared: TeacherPreparedData, cache_dir: str, logger=None) -> None:
    logger = logger or get_logger()
    os.makedirs(cache_dir, exist_ok=True)

    np.save(os.path.join(cache_dir, "x_scaled.npy"), prepared.x_scaled, allow_pickle=False)
    np.save(os.path.join(cache_dir, "x_wl_hash.npy"), prepared.x_wl_hash, allow_pickle=False)
    np.save(os.path.join(cache_dir, "y_scaled.npy"), prepared.y_scaled, allow_pickle=False)
    np.save(os.path.join(cache_dir, "y_raw.npy"), prepared.y_raw, allow_pickle=False)
    np.save(os.path.join(cache_dir, "y_mu.npy"), prepared.y_mu, allow_pickle=False)
    np.save(os.path.join(cache_dir, "y_inv_sigma.npy"), prepared.y_inv_sigma, allow_pickle=False)

    meta = {"has_graph": prepared.x_graph is not None}
    if prepared.x_graph is not None:
        torch.save(_pack_graph_samples(prepared.x_graph), os.path.join(cache_dir, "x_graph.pt"))
    else:
        graph_path = os.path.join(cache_dir, "x_graph.pt")
        if os.path.exists(graph_path):
            os.remove(graph_path)

    with open(os.path.join(cache_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)

    logger.info("Saved prepared teacher data cache to %s", cache_dir)


def load_teacher_data_cache(cache_dir: str, include_graph: bool, logger=None) -> TeacherPreparedData:
    logger = logger or get_logger()

    x_scaled = np.load(os.path.join(cache_dir, "x_scaled.npy"))
    x_wl_hash = np.load(os.path.join(cache_dir, "x_wl_hash.npy"))
    y_scaled = np.load(os.path.join(cache_dir, "y_scaled.npy"))
    y_raw = np.load(os.path.join(cache_dir, "y_raw.npy"))
    y_mu = np.load(os.path.join(cache_dir, "y_mu.npy"))
    y_inv_sigma = np.load(os.path.join(cache_dir, "y_inv_sigma.npy"))

    x_graph = None
    graph_path = os.path.join(cache_dir, "x_graph.pt")
    if include_graph and os.path.exists(graph_path):
        graph_payload = torch.load(graph_path, map_location="cpu")
        x_graph = _unpack_graph_samples(graph_payload)

    logger.info("Loaded prepared teacher data cache from %s", cache_dir)
    return TeacherPreparedData(
        x_raw=None,
        x_scaled=x_scaled,
        x_wl_hash=x_wl_hash,
        x_graph=x_graph,
        y_scaled=y_scaled,
        y_raw=y_raw,
        y_mu=y_mu,
        y_inv_sigma=y_inv_sigma,
    )


def _log1p_transform_y(y: np.ndarray) -> np.ndarray:
    if np.any(y <= -1.0):
        y_min = float(np.min(y))
        raise ValueError(f"y contains values <= -1, cannot apply log1p safely. min(y)={y_min:.6f}")
    return np.log1p(y)


def _build_label_outlier_mask(y: np.ndarray, mode: str, threshold: float, logger) -> np.ndarray:
    y_flat = y.reshape(-1).astype(np.float64)

    if threshold <= 0:
        raise ValueError(f"label_outlier_threshold must be > 0, got {threshold}")

    if mode == "none":
        return np.ones_like(y_flat, dtype=bool)

    if mode == "zscore":
        mu = y_flat.mean()
        sigma = y_flat.std()
        if sigma < 1e-12:
            logger.warning("Label std is nearly zero; skip zscore outlier filtering.")
            return np.ones_like(y_flat, dtype=bool)
        score = np.abs((y_flat - mu) / sigma)
    elif mode == "mad":
        median = np.median(y_flat)
        mad = np.median(np.abs(y_flat - median))
        if mad < 1e-12:
            logger.warning("Label MAD is nearly zero; skip MAD outlier filtering.")
            return np.ones_like(y_flat, dtype=bool)
        score = 0.6745 * np.abs(y_flat - median) / mad
    elif mode == "maximum":
        # 把label绝对值超过threshold的都当作outlier
        score = np.abs(y_flat)
    else:
        raise ValueError(f"Unsupported label_outlier_filter: {mode}")

    keep_mask = score <= threshold
    removed = int((~keep_mask).sum())
    logger.info(
        "Label outlier filter applied: mode=%s, threshold=%.4f, kept=%d, removed=%d",
        mode,
        threshold,
        int(keep_mask.sum()),
        removed,
    )
    return keep_mask


def _plot_label_distribution(y: np.ndarray, output_path: str, logger) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib is not installed; skip plotting label distribution.")
        return

    y_flat = y.reshape(-1).astype(np.float64)
    if y_flat.size == 0:
        logger.warning("No labels available after filtering; skip plotting label distribution.")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    axes[0].hist(y_flat, bins=100, color="#4C72B0", edgecolor="white", alpha=0.9)
    axes[0].set_title("Filtered Label Histogram")
    axes[0].set_xlabel("label")
    axes[0].set_ylabel("count")
    axes[0].grid(alpha=0.2)

    axes[1].boxplot(y_flat, vert=True, patch_artist=True, boxprops={"facecolor": "#55A868", "alpha": 0.7})
    axes[1].set_title("Filtered Label Boxplot")
    axes[1].set_ylabel("label")
    axes[1].grid(alpha=0.2)

    stats_text = (
        f"count={y_flat.size}\n"
        f"mean={y_flat.mean():.4f}\n"
        f"std={y_flat.std():.4f}\n"
        f"min={y_flat.min():.4f}\n"
        f"max={y_flat.max():.4f}"
    )
    axes[0].text(
        0.98,
        0.98,
        stats_text,
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )

    fig.suptitle("Label Distribution After Filtering", fontsize=13)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    logger.info("Saved filtered label distribution plot to %s", output_path)


def _tensorize_graph_payload(payload: str) -> GraphTensorSample:
    data = fast_json.loads(payload)
    node_ids = data["node_ids"]
    node_feats_np = np.asarray(data["node_feats"], dtype=np.float32)

    if node_feats_np.ndim != 2 or node_feats_np.shape[1] != NODE_FEAT_DIM:
        raise ValueError(
            f"Expected node feature shape [N, {NODE_FEAT_DIM}], got {tuple(node_feats_np.shape)}"
        )

    id_map = {int(node_id): idx for idx, node_id in enumerate(node_ids)}
    edge_pairs = []
    edge_types = []
    for src_raw, fanouts in data["subgraph"].items():
        src = id_map[int(src_raw)]
        for dst_raw, is_inv in fanouts:
            edge_pairs.append((src, id_map[int(dst_raw)]))
            edge_types.append(int(is_inv))

    node_feats = torch.from_numpy(node_feats_np)
    if edge_pairs:
        edge_index_np = np.asarray(edge_pairs, dtype=np.int64).T
        edge_type_np = np.asarray(edge_types, dtype=np.int64)
        edge_index = torch.from_numpy(edge_index_np).contiguous()
        edge_type = torch.from_numpy(edge_type_np)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_type = torch.zeros((0,), dtype=torch.long)

    root_candidates = np.flatnonzero(node_feats_np[:, 1] > 0.5)
    root_idx = int(root_candidates[0]) if root_candidates.size > 0 else node_feats_np.shape[0] - 1
    leaf_mask = torch.from_numpy(node_feats_np[:, 0] > 0.5)
    pos_vec = torch.from_numpy(node_feats_np[:, [3, 4, 5, 0, 1]])

    return GraphTensorSample(
        node_feats=node_feats,
        edge_index=edge_index,
        edge_type=edge_type,
        pos_vec=pos_vec,
        root_index=root_idx,
        leaf_mask=leaf_mask,
    )


def _tensorize_graph_payload_chunk(payload_chunk: Sequence[str]) -> list[GraphTensorSample]:
    return [_tensorize_graph_payload(payload) for payload in payload_chunk]


def _permute_and_tensorize_graphs(
    x_graph: Sequence[str],
    perm: np.ndarray,
    logger,
    max_workers: int | None = None,
    chunk_size: int = 128,
) -> tuple[GraphTensorSample, ...]:
    if not x_graph:
        return tuple()

    reordered_payloads = [x_graph[int(i)] for i in perm]

    if max_workers is None:
        max_workers = min(8, len(reordered_payloads), os.cpu_count() or 1)
    max_workers = max(1, max_workers)
    chunk_size = max(1, chunk_size)
    total = len(reordered_payloads)
    payload_chunks = [
        reordered_payloads[start : start + chunk_size]
        for start in range(0, total, chunk_size)
    ]
    total_chunks = len(payload_chunks)

    chunk = 500
    progress_step = max(1, total // chunk)
    start = time.perf_counter()

    def log_progress(done: int) -> None:
        elapsed = max(time.perf_counter() - start, 1e-9)
        rate = done / elapsed
        remaining = max(total - done, 0)
        eta = remaining / rate if rate > 1e-9 else 0.0
        logger.info(
            "Graph tensorize: %d/%d (%.1f%%), elapsed=%.1fs, eta=%.1fs, rate=%.2f it/s",
            done,
            total,
            100.0 * done / total,
            elapsed,
            eta,
            rate,
        )

    logger.info(
        "Starting graph tensorization for %d samples with max_workers=%d chunk_size=%d total_chunks=%d",
        total,
        max_workers,
        chunk_size,
        total_chunks,
    )
    
    if max_workers == 1:
        results = []
        done = 0
        for payload_chunk in payload_chunks:
            tensorized_chunk = _tensorize_graph_payload_chunk(payload_chunk)
            results.extend(tensorized_chunk)
            done += len(tensorized_chunk)
            if done % progress_step == 0 or done == total:
                log_progress(done)
        return tuple(results)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = [None] * total
        done = 0
        max_in_flight = max_workers * 8
        chunk_iter = iter(enumerate(payload_chunks))
        pending = {}

        for _ in range(min(max_in_flight, total_chunks)):
            chunk_idx, payload_chunk = next(chunk_iter)
            pending[executor.submit(_tensorize_graph_payload_chunk, payload_chunk)] = chunk_idx

        while pending:
            finished, _ = wait(pending.keys(), return_when=FIRST_COMPLETED)
            for future in finished:
                chunk_idx = pending.pop(future)
                chunk_results = future.result()
                base_idx = chunk_idx * chunk_size
                results[base_idx : base_idx + len(chunk_results)] = chunk_results
                done += len(chunk_results)
                if done % progress_step == 0 or done == total:
                    log_progress(done)

                try:
                    next_chunk_idx, next_payload_chunk = next(chunk_iter)
                except StopIteration:
                    continue
                pending[executor.submit(_tensorize_graph_payload_chunk, next_payload_chunk)] = next_chunk_idx
        return tuple(results)


def load_teacher_data(cfg: ExtractCutLibConfig, include_graph: bool, inside_logger) -> TeacherPreparedData:
    logger = inside_logger or get_logger()

    data_path = f"{cfg.dataset_path}/save_json_OpenABC"
    x_openabc, x_openabc_graph, y_openabc, elapsed = ext_utils.load_json_data(
        data_path, logger=inside_logger, include_graph=include_graph, max_workers=cfg.cpu_worker
    )
    logger.info("Data loaded from openABC-D: %d samples, Time taken: %.2f seconds", len(y_openabc), elapsed)

    data_path = f"{cfg.dataset_path}/save_json_ACE"
    x_ace, x_ace_graph, y_ace, elapsed = ext_utils.load_json_data(
        data_path, logger=inside_logger, include_graph=include_graph, max_workers=cfg.cpu_worker
    )
    logger.info("Data loaded from ACE: %d samples, Time taken: %.2f seconds", len(y_ace), elapsed)

    data_path = f"{cfg.dataset_path}/save_json_iwls2022"
    x_iwls, x_iwls_graph, y_iwls, elapsed = ext_utils.load_json_data(
        data_path, logger=inside_logger, include_graph=include_graph, max_workers=cfg.cpu_worker
    )
    logger.info("Data loaded from iwls2022: %d samples, Time taken: %.2f seconds", len(y_iwls), elapsed)

    data_path = f"{cfg.dataset_path}/save_json_iwls2024"
    x_iwls2024, x_iwls2024_graph, y_iwls2024, elapsed = ext_utils.load_json_data(
        data_path, logger=inside_logger, include_graph=include_graph, max_workers=cfg.cpu_worker
    )
    logger.info("Data loaded from iwls2024: %d samples, Time taken: %.2f seconds", len(y_iwls2024), elapsed)

    data_path = f"{cfg.dataset_path}/save_json_forgeEDA"
    x_forgeeda, x_forgeeda_graph, y_forgeeda, elapsed = ext_utils.load_json_data(
        data_path, logger=inside_logger, include_graph=include_graph, max_workers=cfg.cpu_worker
    )
    logger.info("Data loaded from forgeEDA: %d samples, Time taken: %.2f seconds", len(y_forgeeda), elapsed)
    # x_forgeeda_graph = tuple()
    # x_forgeeda = np.empty((0, x_openabc.shape[1]), dtype=x_openabc.dtype)
    # y_forgeeda = np.empty((0, y_openabc.shape[1]), dtype=y_openabc.dtype)

    x_graph = None
    if include_graph:
        x_graph = tuple(x_openabc_graph + x_ace_graph + x_iwls_graph + x_iwls2024_graph + x_forgeeda_graph)

    x_raw = ext_utils.np.concatenate([x_openabc, x_ace, x_iwls, x_iwls2024, x_forgeeda], axis=0)
    y = ext_utils.np.concatenate([y_openabc, y_ace, y_iwls, y_iwls2024, y_forgeeda], axis=0).reshape(-1, 1)
    logger.info("Finished concatenating arrays: x_raw=%s, y=%s", tuple(x_raw.shape), tuple(y.shape))

    keep_mask = _build_label_outlier_mask(y, cfg.label_outlier_filter, cfg.label_outlier_threshold, logger)
    if not np.all(keep_mask):
        x_raw = x_raw[keep_mask]
        y = y[keep_mask]
        if x_graph is not None:
            x_graph = tuple(graph for graph, keep in zip(x_graph, keep_mask) if keep)
        logger.info("After label filtering: x_raw=%s, y=%s", tuple(x_raw.shape), tuple(y.shape))

    plot_path = os.path.join(cfg.checkpoint_dir, "label_distribution_filtered.png")
    _plot_label_distribution(y, plot_path, logger)

    del x_openabc, y_openabc, x_ace, y_ace, x_iwls, y_iwls, x_iwls2024, y_iwls2024, x_forgeeda, y_forgeeda
    ext_utils.torch.cuda.empty_cache()
    gc.collect()

    perm = ext_utils.np.random.permutation(len(y))
    x_raw = x_raw[perm]
    y = y[perm]

    if x_graph is not None:
        start = time.perf_counter()
        x_graph = _permute_and_tensorize_graphs(
            x_graph,
            perm,
            logger,
            max_workers=cfg.cpu_graph_worker,
            chunk_size=cfg.graph_tensorize_chunk_size,
        )
        finish = time.perf_counter()
        logger.info(
            "Graph reorder+tensorization completed for %d samples with %d workers, Time taken: %.2f seconds",
            len(y),
            max(1, cfg.cpu_graph_worker),
            finish - start,
        )

    x_scaled, _, _ = standardize_features(x_raw[:, :-6])
    x_wl_hash = x_raw[:, -1].astype(np.int64)
    y_log = _log1p_transform_y(y)
    y_scaled, y_mu, y_inv_sigma = standardize_features(y_log)

    if x_scaled.shape[1] != FEATURE_INPUT_DIM:
        raise ValueError(
            f"Expected {FEATURE_INPUT_DIM} continuous features before WL_Hash, got {x_scaled.shape[1]}. "
            "Please verify feature export order."
        )

    return TeacherPreparedData(
        x_raw = x_raw.astype(np.float32),
        x_scaled=x_scaled.astype(np.float32),
        x_wl_hash=x_wl_hash,
        x_graph=x_graph,
        y_scaled=y_scaled.astype(np.float32),
        y_raw=y.astype(np.float32),
        y_mu=y_mu.astype(np.float32),
        y_inv_sigma=y_inv_sigma.astype(np.float32),
    )
