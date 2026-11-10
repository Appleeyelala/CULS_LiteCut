from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
import time
from typing import Any, Iterator

import numpy as np

from .preprocess import standardize_features

try:
    import ujson as fast_json
except ImportError:
    fast_json = json


DEFAULT_CASE_DIRS = (
    "save_json_OpenABC",
    "save_json_ACE",
    "save_json_iwls2022",
    "save_json_iwls2024",
    "save_json_forgeEDA",
)
FEATURE_PREVIEW_DIM = 16


@dataclass(frozen=True)
class SampleRecord:
    global_index: int
    dataset: str
    case: str
    case_id: str
    cell_struct_key: str
    delay: float
    x_dim_before_trim: int
    x_dim_after_trim: int
    gate_ids: tuple[str, ...]
    cell_struct_signature: str
    cell_struct_summary: dict[str, Any]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load cutlib baseline x/y samples, remove the first 8 feature dims, "
            "then find nearest samples and summarize their case/cell_struct/delay."
        )
    )
    parser.add_argument("--dataset_path", type=str, required=True, help="Root directory that contains save_json_*")
    parser.add_argument(
        "--case_dirs",
        nargs="+",
        default=list(DEFAULT_CASE_DIRS),
        help="Dataset subdirectories to scan",
    )
    parser.add_argument(
        "--query_indices",
        nargs="+",
        type=int,
        default=None,
        help="Global sample indices to analyze; defaults to the first query_count samples",
    )
    parser.add_argument(
        "--query_count",
        type=int,
        default=3,
        help="How many samples to analyze when query_indices is not provided",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed used when sampling query indices",
    )
    parser.add_argument("--neighbor_k", type=int, default=10, help="Number of nearest neighbors to report")
    parser.add_argument(
        "--metric",
        type=str,
        choices=["euclidean", "cosine"],
        default="euclidean",
        help="Distance metric used for nearest-neighbor search",
    )
    parser.add_argument(
        "--standardize",
        action="store_true",
        help="Standardize trimmed features before nearest-neighbor search",
    )
    parser.add_argument(
        "--include_self",
        action="store_true",
        help="Whether the query sample itself can appear in the neighbor list",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional cap for loaded samples, useful for quick debugging",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of worker processes for JSON loading; defaults to up to 8",
    )
    parser.add_argument(
        "--delay_bins",
        type=int,
        default=8,
        help="Number of bins for neighbor delay histogram",
    )
    parser.add_argument(
        "--top_count",
        type=int,
        default=10,
        help="Maximum number of top items kept for case/cell_struct counters",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./log/cutlib_baseline_neighbor_analysis.json",
        help="Where to save the JSON report",
    )
    return parser.parse_args()


def _dataset_name(case_dir: str) -> str:
    prefix = "save_json_"
    return case_dir[len(prefix) :] if case_dir.startswith(prefix) else case_dir


def _iter_json_files(dataset_path: str, case_dirs: list[str]) -> Iterator[tuple[str, str, str]]:
    for case_dir in case_dirs:
        case_root = os.path.join(dataset_path, case_dir)
        if not os.path.isdir(case_root):
            raise FileNotFoundError(f"Case directory does not exist: {case_root}")
        for entry in sorted(os.scandir(case_root), key=lambda item: item.name):
            if not entry.is_dir():
                continue
            json_path = os.path.join(entry.path, "data_2.json")
            if os.path.isfile(json_path):
                yield case_dir, entry.name, json_path


def _build_cell_struct_signature(cell_struct: dict[str, Any]) -> str:
    gate_ids = cell_struct.get("gate_ids") or []
    if gate_ids:
        return "|".join(str(gate_id) for gate_id in gate_ids)

    feats = cell_struct.get("feats") or {}
    parts = [
        f"supergate={cell_struct.get('supergateIndex', 'NA')}",
        f"cut_id={feats.get('cut_id', 'NA')}",
        f"cuts={len(cell_struct.get('cuts') or [])}",
        f"nodes={len(cell_struct.get('nodes') or [])}",
    ]
    return ";".join(parts)


def _build_cell_struct_summary(cell_key: str, cell_struct: dict[str, Any]) -> dict[str, Any]:
    feats = cell_struct.get("feats") or {}
    return {
        "cell_struct_key": cell_key,
        "supergateIndex": cell_struct.get("supergateIndex"),
        "gate_ids": cell_struct.get("gate_ids") or [],
        "cut_count": len(cell_struct.get("cuts") or []),
        "node_count": len(cell_struct.get("nodes") or []),
        "area": cell_struct.get("Area"),
        "rt_id": feats.get("rt_id"),
        "cut_id": feats.get("cut_id"),
    }


def _load_single_case_samples(job: tuple[str, str, str]) -> tuple[np.ndarray | None, np.ndarray | None, list[dict[str, Any]]]:
    case_dir, case_name, json_path = job
    dataset_name = _dataset_name(case_dir)

    with open(json_path, "r", encoding="utf-8") as f:
        payload = fast_json.load(f)

    trimmed_features: list[list[float]] = []
    delays: list[float] = []
    record_payloads: list[dict[str, Any]] = []

    cell_struct_map = payload.get("cell_struct") or {}
    for cell_key, cell_struct in cell_struct_map.items():
        if cell_struct.get("supergateIndex") == -1 or "abc_delay" not in cell_struct:
            continue

        feats = ((cell_struct.get("feats") or {}).get("feats")) or []
        if len(feats) <= 8:
            continue

        trimmed = feats[8:]
        delay = float(cell_struct["abc_delay"])
        trimmed_features.append(trimmed)
        delays.append(delay)
        record_payloads.append(
            {
                "dataset": dataset_name,
                "case": case_name,
                "case_id": f"{dataset_name}/{case_name}",
                "cell_struct_key": str(cell_key),
                "delay": delay,
                "x_dim_before_trim": len(feats),
                "x_dim_after_trim": len(trimmed),
                "gate_ids": tuple(str(gate_id) for gate_id in (cell_struct.get("gate_ids") or [])),
                "cell_struct_signature": _build_cell_struct_signature(cell_struct),
                "cell_struct_summary": _build_cell_struct_summary(str(cell_key), cell_struct),
            }
        )

    if not trimmed_features:
        return None, None, []

    return (
        np.asarray(trimmed_features, dtype=np.float32),
        np.asarray(delays, dtype=np.float32).reshape(-1, 1),
        record_payloads,
    )


def _load_samples(
    dataset_path: str,
    case_dirs: list[str],
    max_samples: int | None,
    num_workers: int | None,
) -> tuple[np.ndarray, np.ndarray, list[SampleRecord]]:
    jobs = list(_iter_json_files(dataset_path, case_dirs))
    if not jobs:
        return np.empty((0, 0), dtype=np.float32), np.empty((0, 1), dtype=np.float32), []

    if num_workers is None:
        num_workers = min(8, len(jobs), os.cpu_count() or 1)
    num_workers = max(1, num_workers)

    # Keep small debug loads eager-stop friendly.
    if max_samples is not None:
        num_workers = 1

    total_jobs = len(jobs)
    progress_interval = max(1, total_jobs // 20)
    start_time = time.perf_counter()
    print(
        f"[load] start: {total_jobs} cases, num_workers={num_workers}, max_samples={max_samples}",
        flush=True,
    )

    if num_workers == 1:
        result_iter = (_load_single_case_samples(job) for job in jobs)
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            result_iter = executor.map(_load_single_case_samples, jobs)

            x_batches: list[np.ndarray] = []
            y_batches: list[np.ndarray] = []
            record_payloads: list[dict[str, Any]] = []
            loaded = 0
            processed_jobs = 0

            for x_batch, y_batch, payload_batch in result_iter:
                processed_jobs += 1

                if x_batch is not None and y_batch is not None and payload_batch:
                    if max_samples is not None:
                        remaining = max_samples - loaded
                        if remaining <= 0:
                            break
                        if len(payload_batch) > remaining:
                            x_batch = x_batch[:remaining]
                            y_batch = y_batch[:remaining]
                            payload_batch = payload_batch[:remaining]

                    x_batches.append(x_batch)
                    y_batches.append(y_batch)
                    record_payloads.extend(payload_batch)
                    loaded += len(payload_batch)

                if (
                    processed_jobs == 1
                    or processed_jobs % progress_interval == 0
                    or processed_jobs == total_jobs
                    or (max_samples is not None and loaded >= max_samples)
                ):
                    elapsed = time.perf_counter() - start_time
                    print(
                        f"[load] progress: {processed_jobs}/{total_jobs} cases, "
                        f"{loaded} samples, elapsed={elapsed:.1f}s",
                        flush=True,
                    )

                if max_samples is not None and loaded >= max_samples:
                    break
    if num_workers == 1:
        x_batches = []
        y_batches = []
        record_payloads = []
        loaded = 0
        processed_jobs = 0

        for x_batch, y_batch, payload_batch in result_iter:
            processed_jobs += 1

            if x_batch is not None and y_batch is not None and payload_batch:
                if max_samples is not None:
                    remaining = max_samples - loaded
                    if remaining <= 0:
                        break
                    if len(payload_batch) > remaining:
                        x_batch = x_batch[:remaining]
                        y_batch = y_batch[:remaining]
                        payload_batch = payload_batch[:remaining]

                x_batches.append(x_batch)
                y_batches.append(y_batch)
                record_payloads.extend(payload_batch)
                loaded += len(payload_batch)

            if (
                processed_jobs == 1
                or processed_jobs % progress_interval == 0
                or processed_jobs == total_jobs
                or (max_samples is not None and loaded >= max_samples)
            ):
                elapsed = time.perf_counter() - start_time
                print(
                    f"[load] progress: {processed_jobs}/{total_jobs} cases, "
                    f"{loaded} samples, elapsed={elapsed:.1f}s",
                    flush=True,
                )

            if max_samples is not None and loaded >= max_samples:
                break

    if not x_batches:
        return np.empty((0, 0), dtype=np.float32), np.empty((0, 1), dtype=np.float32), []

    x = np.concatenate(x_batches, axis=0)
    y = np.concatenate(y_batches, axis=0)
    records = [
        SampleRecord(
            global_index=idx,
            dataset=payload["dataset"],
            case=payload["case"],
            case_id=payload["case_id"],
            cell_struct_key=payload["cell_struct_key"],
            delay=payload["delay"],
            x_dim_before_trim=payload["x_dim_before_trim"],
            x_dim_after_trim=payload["x_dim_after_trim"],
            gate_ids=payload["gate_ids"],
            cell_struct_signature=payload["cell_struct_signature"],
            cell_struct_summary=payload["cell_struct_summary"],
        )
        for idx, payload in enumerate(record_payloads)
    ]
    elapsed = time.perf_counter() - start_time
    print(f"[load] done: {len(records)} samples loaded in {elapsed:.1f}s", flush=True)
    return x, y, records


def _resolve_query_indices(
    num_samples: int,
    query_indices: list[int] | None,
    query_count: int,
    random_seed: int,
) -> list[int]:
    if num_samples == 0:
        return []

    if query_indices is None:
        rng = np.random.default_rng(random_seed)
        picked = rng.choice(num_samples, size=min(query_count, num_samples), replace=False)
        return picked.tolist()

    normalized: list[int] = []
    for idx in query_indices:
        if idx < 0 or idx >= num_samples:
            raise IndexError(f"Query index {idx} is out of range for {num_samples} samples")
        normalized.append(idx)
    return normalized


def _compute_distances(x: np.ndarray, query_indices: list[int], metric: str) -> np.ndarray:
    queries = x[query_indices]
    if metric == "cosine":
        x_norm = x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)
        q_norm = queries / np.clip(np.linalg.norm(queries, axis=1, keepdims=True), 1e-12, None)
        return 1.0 - q_norm @ x_norm.T

    diff = queries[:, None, :] - x[None, :, :]
    return np.linalg.norm(diff, axis=2)


def _summarize_counter(counter: Counter[str], top_count: int) -> list[dict[str, Any]]:
    return [{"name": name, "count": count} for name, count in counter.most_common(top_count)]


def _build_delay_summary(delay_values: list[float], delay_bins: int) -> dict[str, Any]:
    if not delay_values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "histogram": [],
        }

    delay_array = np.asarray(delay_values, dtype=np.float64)
    hist, edges = np.histogram(delay_array, bins=min(delay_bins, len(delay_array)))
    histogram = []
    for idx, count in enumerate(hist.tolist()):
        histogram.append(
            {
                "start": float(edges[idx]),
                "end": float(edges[idx + 1]),
                "count": int(count),
            }
        )
    return {
        "count": int(delay_array.size),
        "min": float(delay_array.min()),
        "max": float(delay_array.max()),
        "mean": float(delay_array.mean()),
        "median": float(np.median(delay_array)),
        "histogram": histogram,
    }


def _analyze_neighbors(
    x: np.ndarray,
    y: np.ndarray,
    records: list[SampleRecord],
    query_indices: list[int],
    neighbor_k: int,
    metric: str,
    include_self: bool,
    delay_bins: int,
    top_count: int,
) -> list[dict[str, Any]]:
    distances = _compute_distances(x, query_indices, metric)
    results: list[dict[str, Any]] = []

    for row_idx, query_index in enumerate(query_indices):
        order = np.argsort(distances[row_idx], kind="stable").tolist()
        if not include_self:
            order = [idx for idx in order if idx != query_index]
        order = order[:neighbor_k]

        case_counter: Counter[str] = Counter()
        struct_counter: Counter[str] = Counter()
        neighbor_details: list[dict[str, Any]] = []
        neighbor_delays: list[float] = []

        for neighbor_index in order:
            neighbor_record = records[neighbor_index]
            case_counter[neighbor_record.case_id] += 1
            struct_counter[neighbor_record.cell_struct_signature] += 1
            neighbor_delays.append(float(y[neighbor_index, 0]))
            neighbor_details.append(
                {
                    "rank": len(neighbor_details) + 1,
                    "distance": float(distances[row_idx, neighbor_index]),
                    "global_index": neighbor_record.global_index,
                    "dataset": neighbor_record.dataset,
                    "case": neighbor_record.case,
                    "case_id": neighbor_record.case_id,
                    "cell_struct_key": neighbor_record.cell_struct_key,
                    "delay": neighbor_record.delay,
                    "gate_ids": list(neighbor_record.gate_ids),
                    "x_feature_preview": x[neighbor_index, :FEATURE_PREVIEW_DIM].tolist(),
                    "cell_struct_signature": neighbor_record.cell_struct_signature,
                    "cell_struct_summary": neighbor_record.cell_struct_summary,
                }
            )

        query_record = records[query_index]
        results.append(
            {
                "query": {
                    "global_index": query_record.global_index,
                    "dataset": query_record.dataset,
                    "case": query_record.case,
                    "case_id": query_record.case_id,
                    "cell_struct_key": query_record.cell_struct_key,
                    "delay": query_record.delay,
                    "gate_ids": list(query_record.gate_ids),
                    "x_feature": x[query_index].tolist(),
                    "cell_struct_signature": query_record.cell_struct_signature,
                    "cell_struct_summary": query_record.cell_struct_summary,
                },
                "neighbors": neighbor_details,
                "stats": {
                    "case_counts": _summarize_counter(case_counter, top_count),
                    "cell_struct_counts": _summarize_counter(struct_counter, top_count),
                    "delay_summary": _build_delay_summary(neighbor_delays, delay_bins),
                },
            }
        )

    return results


def _print_summary(report: dict[str, Any]) -> None:
    print(
        f"Loaded {report['num_samples']} samples, feature dim {report['feature_dim_after_trim']} "
        f"after trimming the first 8 dims."
    )
    print(
        f"Analyzed {len(report['query_results'])} query samples with metric={report['metric']} "
        f"and neighbor_k={report['neighbor_k']}."
    )

    for item in report["query_results"]:
        query = item["query"]
        delay_summary = item["stats"]["delay_summary"]
        print(
            f"[Query {query['global_index']}] case={query['case_id']} "
            f"cell_struct={query['cell_struct_key']} delay={query['delay']:.6f}"
        )
        print(f"  top cases: {item['stats']['case_counts'][:3]}")
        print(f"  top cell_struct: {item['stats']['cell_struct_counts'][:3]}")
        print(
            "  neighbor delay summary: "
            f"min={delay_summary['min']} max={delay_summary['max']} "
            f"mean={delay_summary['mean']} median={delay_summary['median']}"
        )


def main() -> None:
    args = _parse_args()
    x, y, records = _load_samples(
        args.dataset_path,
        args.case_dirs,
        args.max_samples,
        args.num_workers,
    )
    if x.size == 0:
        raise RuntimeError("No valid samples were loaded from the provided dataset_path")

    x_for_search = x
    if args.standardize:
        x_for_search, _, _ = standardize_features(x_for_search)

    query_indices = _resolve_query_indices(
        len(records),
        args.query_indices,
        args.query_count,
        args.random_seed,
    )
    query_results = _analyze_neighbors(
        x=x_for_search,
        y=y,
        records=records,
        query_indices=query_indices,
        neighbor_k=args.neighbor_k,
        metric=args.metric,
        include_self=args.include_self,
        delay_bins=args.delay_bins,
        top_count=args.top_count,
    )

    report = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "dataset_path": args.dataset_path,
        "case_dirs": list(args.case_dirs),
        "num_samples": len(records),
        "feature_dim_before_trim": records[0].x_dim_before_trim,
        "feature_dim_after_trim": int(x_for_search.shape[1]),
        "metric": args.metric,
        "neighbor_k": args.neighbor_k,
        "standardize": args.standardize,
        "include_self": args.include_self,
        "query_results": query_results,
        "first_sample_preview": asdict(records[0]) if records else None,
    }

    if args.output_path:
        output_dir = os.path.dirname(args.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    _print_summary(report)
    if args.output_path:
        print(f"Saved report to {args.output_path}")


if __name__ == "__main__":
    main()
