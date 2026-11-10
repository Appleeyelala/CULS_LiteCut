import json
import os
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

import numpy as np
import torch

try:
    import ujson as fast_json
except ImportError:
    fast_json = json
import time


def _load_single_json_case(json_path, include_graph):
    with open(json_path, "r", encoding="utf-8") as f:
        data = fast_json.load(f)

    case_x = []
    case_y = []
    case_graph = [] if include_graph else None

    cell_struct = data["cell_struct"]
    while cell_struct:
        _, cut_struct = cell_struct.popitem()
        if cut_struct["supergateIndex"] == -1 or "abc_delay" not in cut_struct:
            continue

        case_x.append(cut_struct["feats"]["feats"])
        case_y.append(cut_struct["abc_delay"])
        if include_graph:
            graph_payload = {
                "subgraph": cut_struct["feats"]["subgraph"],
                "node_ids": cut_struct["feats"].get("node_ids", []),
                "node_feats": cut_struct["feats"].get("node_feats", []),
                "node_feat_names": cut_struct["feats"].get("node_feat_names", []),
            }
            case_graph.append(fast_json.dumps(graph_payload))

    if not case_x:
        return json_path, None, None, None, 0

    x_batch = np.asarray(case_x, dtype=np.float32)
    y_batch = np.asarray(case_y, dtype=np.float32).reshape(-1, 1)
    graph_batch = tuple(case_graph) if include_graph else None
    return json_path, x_batch, y_batch, graph_batch, len(case_x)

def load_data_from_infer(data):
    # data: list of [delay, feats]
    delays = []
    feats = []
    indexes = []
    for delay, f, index in data:
        delays.append(delay)
        feats.append(f)
        indexes.append(index)
    X = np.asarray(feats, dtype=np.float64)   # (N, D)
    y = np.asarray(delays, dtype=np.float64)  # (N,)
    y = y.reshape(-1, 1)  # (N, 1)
    return X, y, indexes

def load_data_from_subg(data_path, logger=None):
    x = []
    y = []
    for data_pt in os.listdir(data_path):
        if data_pt.endswith('.pt'):
            data = torch.load(os.path.join(data_path, data_pt))
            if logger:
                logger.info(f"Loading {data_pt} with {len(data)} samples")
            for item in data:
                feats = item['feats']
                label = item['lable'][:,2]
                x += feats.cpu().numpy().tolist()
                y += label.cpu().numpy().tolist()
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    y = y.reshape(-1, 1)
    return x, y

def load_json_data(data_path, logger=None, include_graph=False, max_workers=None):
    start = time.perf_counter()

    x_batches = []
    y_batches = []
    x_graph = [] if include_graph else None

    json_paths = []
    for case_name in sorted(os.listdir(data_path)):
        json_path = os.path.join(data_path, case_name, "data_2.json")
        if os.path.isfile(json_path):
            json_paths.append(json_path)

    if max_workers is None:
        max_workers = min(4, len(json_paths), os.cpu_count() or 1)
    max_workers = max(1, max_workers)

    if max_workers == 1:
        result_iter = (_load_single_json_case(json_path, include_graph) for json_path in json_paths)
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            result_iter = executor.map(_load_single_json_case, json_paths, repeat(include_graph))
            result_iter = list(result_iter)

    for json_path, x_batch, y_batch, graph_batch, loaded_in_case in result_iter:
        if x_batch is not None:
            x_batches.append(x_batch)
            y_batches.append(y_batch)
            if include_graph:
                x_graph.extend(graph_batch)

        if logger:
            logger.info("Loaded %d cuts from %s", loaded_in_case, json_path)

    if not x_batches:
        empty_graph = tuple() if include_graph else None
        return np.empty((0, 0), dtype=np.float32), empty_graph, np.empty((0, 1), dtype=np.float32)

    graph_data = tuple(x_graph) if include_graph else None

    finish = time.perf_counter()
    return np.concatenate(x_batches, axis=0), graph_data, np.concatenate(y_batches, axis=0), finish - start

def mad(x: np.ndarray) -> float:
    x = np.asarray(x)
    m = np.median(x)
    return np.median(np.abs(x - m)) + 1e-12

def filter_small_clusters_and_outliers(
    Z, y, cluster_ids, centers,
    min_abs=300,
    min_rel_to_median=0.2,
    dist_mad_k=7.0,
    y_mad_k=7.0,
    outlier_mode="or",  # "or" or "and"
):
    """
    返回:
      keep_mask: (N,) bool  True 表示保留该样本
      valid_clusters: list[int] 保留的簇 id
    """
    Z = np.asarray(Z)
    y = np.asarray(y).ravel()
    cluster_ids = np.asarray(cluster_ids)

    K = centers.shape[0]
    counts = np.bincount(cluster_ids, minlength=K)
    med_cnt = np.median(counts[counts > 0]) if np.any(counts > 0) else 0
    min_rel = int(np.ceil(min_rel_to_median * med_cnt))
    min_keep = max(min_abs, min_rel)

    valid = np.where(counts >= min_keep)[0].tolist()

    keep_mask = np.zeros(len(y), dtype=bool)

    for k in valid:
        idx = np.where(cluster_ids == k)[0]
        if len(idx) == 0:
            continue

        Zk = Z[idx]
        yk = y[idx]
        ck = centers[k]

        # (1) 距离异常
        dist = np.linalg.norm(Zk - ck, axis=1)
        dist_thr = np.median(dist) + dist_mad_k * mad(dist)
        bad_dist = dist > dist_thr

        # (2) delay 异常
        y_med = np.median(yk)
        y_thr = y_mad_k * mad(yk)
        bad_y = np.abs(yk - y_med) > y_thr

        if outlier_mode == "and":
            bad = bad_dist & bad_y
        else:
            bad = bad_dist | bad_y

        keep_mask[idx[~bad]] = True

    return keep_mask, valid, counts, min_keep
