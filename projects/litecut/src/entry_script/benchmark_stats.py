"""Compute benchmark statistics table for cutLib dataset.

Columns: Source | #Designs | #Cuts Train (×10³) | #Cuts Test (×10³) | Delay Mean (ps) | Delay Std (ps)
Rows: OpenABC | ACE | IWLS (iwls2022+iwls2024 merged) | forgeEDA | Total

ForgeEDA uses stratified subsampling by per-design median delay to keep
its delay distribution shape comparable with other sources.

Usage:
    conda run -n dg2 python src/entry_script/benchmark_stats.py \
        --dataset_path /data/zhoulingfeng/data_cutLib/cutLibData2 \
        --train_ratio 0.9
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np


logging.basicConfig(
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger("benchmark_stats")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DesignRecord:
    source: str
    design: str
    json_path: str
    n_cuts: int
    delays: np.ndarray
    median_delay: float
    mean_delay: float
    std_delay: float
    min_delay: float
    max_delay: float


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute benchmark statistics table")
    parser.add_argument("--dataset_path", type=str,
                        default="/data/zhoulingfeng/data_cutLib/cutLibData2")
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Save summary table CSV")
    parser.add_argument("--output_detail_csv", type=str, default=None,
                        help="Save per-design detail CSV")
    parser.add_argument("--n_workers", type=int, default=16)

    # forgeEDA stratified sampling
    parser.add_argument("--forgeeda_sample_size", type=int, default=50,
                        help="Target number of forgeEDA designs after stratified sampling")
    parser.add_argument("--forgeeda_n_bins", type=int, default=10,
                        help="Number of quantile bins for stratified sampling")
    parser.add_argument("--random_seed", type=int, default=42)

    # outlier filter (match training pipeline)
    parser.add_argument("--label_outlier_filter", type=str, default="mad",
                        choices=["none", "zscore", "mad"])
    parser.add_argument("--label_outlier_threshold", type=float, default=10.0)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# jq helpers
# ---------------------------------------------------------------------------

_JQ_BIN = "jq"

_JQ_DELAY_FILTER = (
    ".cell_struct | to_entries[] "
    "| select(.value.supergateIndex != -1 and .value.abc_delay != null) "
    "| .value.abc_delay"
)


def _extract_delays_one(json_path: str) -> list[float]:
    """Extract all valid abc_delay values from one data_2.json via jq."""
    result = subprocess.run(
        [_JQ_BIN, "-r", _JQ_DELAY_FILTER, json_path],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    return [float(v) for v in result.stdout.splitlines() if v.strip()]


# ---------------------------------------------------------------------------
# Per-design collection
# ---------------------------------------------------------------------------

def _collect_designs(
    source_dirs: list[str],
    source_name: str,
    n_workers: int,
) -> list[DesignRecord]:
    """Extract delays for every design under source_dirs, returning per-design records."""
    records: list[DesignRecord] = []

    for source_dir in source_dirs:
        if not os.path.isdir(source_dir):
            logger.warning("Directory not found, skipping: %s", source_dir)
            continue

        json_paths: list[tuple[str, str]] = []  # (design_name, json_path)
        for design in sorted(os.listdir(source_dir)):
            p = os.path.join(source_dir, design, "data_2.json")
            if os.path.isfile(p):
                json_paths.append((design, p))

        total = len(json_paths)
        done = 0

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(_extract_delays_one, p): (name, p)
                for name, p in json_paths
            }
            for fut in as_completed(futures):
                design_name, path = futures[fut]
                done += 1
                try:
                    delays_list = fut.result()
                except Exception as e:
                    logger.warning("Failed %s: %s", path, e)
                    continue

                if not delays_list:
                    continue
                arr = np.array(delays_list, dtype=np.float64)
                records.append(DesignRecord(
                    source=source_name,
                    design=design_name,
                    json_path=path,
                    n_cuts=len(arr),
                    delays=arr,
                    median_delay=float(np.median(arr)),
                    mean_delay=float(np.mean(arr)),
                    std_delay=float(np.std(arr)),
                    min_delay=float(np.min(arr)),
                    max_delay=float(np.max(arr)),
                ))

                if done % 20 == 0 or done == total:
                    logger.info(
                        "[%s] %s progress: %d/%d designs done, records=%d",
                        source_name, os.path.basename(source_dir),
                        done, total, len(records),
                    )

        logger.info(
            "[%s] Finished %s — %d/%d designs with valid cuts",
            source_name, os.path.basename(source_dir),
            len([r for r in records
                 if os.path.dirname(r.json_path) .startswith(source_dir)]),
            total,
        )

    return records


# ---------------------------------------------------------------------------
# Stratified subsampling
# ---------------------------------------------------------------------------

def _stratified_sample(
    records: list[DesignRecord],
    n_bins: int,
    sample_size: int,
    seed: int,
) -> list[DesignRecord]:
    """Stratified sample by per-design median delay.

    1. Compute quantile bin edges from all designs' median delays.
    2. Assign each design to a bin.
    3. Sample ceil(sample_size / n_bins) from each bin (or all if bin is smaller).
    4. Trim to sample_size total.
    """
    rng = np.random.RandomState(seed)
    medians = np.array([r.median_delay for r in records])
    bin_edges = np.quantile(medians, np.linspace(0, 1, n_bins + 1))
    # np.digitize returns 1-based; clamp to [1, n_bins]
    assignments = np.clip(np.digitize(medians, bin_edges[1:-1], right=True), 0, n_bins - 1)

    per_bin = max(1, int(np.ceil(sample_size / n_bins)))
    selected: list[DesignRecord] = []

    for b in range(n_bins):
        idxs = np.where(assignments == b)[0]
        if len(idxs) == 0:
            continue
        chosen = rng.choice(idxs, size=min(per_bin, len(idxs)), replace=False)
        for i in chosen:
            selected.append(records[i])
        logger.info(
            "[stratified] bin %d/%d: range=[%.1f, %.1f) ps, available=%d, sampled=%d",
            b + 1, n_bins,
            bin_edges[b], bin_edges[b + 1],
            len(idxs), len(chosen),
        )

    # trim to exact sample_size if over-sampled
    if len(selected) > sample_size:
        rng.shuffle(selected)
        selected = selected[:sample_size]

    logger.info(
        "[stratified] Total selected: %d designs from %d (target=%d)",
        len(selected), len(records), sample_size,
    )
    return selected


# ---------------------------------------------------------------------------
# Outlier filter (replicate training pipeline)
# ---------------------------------------------------------------------------

def _build_label_outlier_mask(y: np.ndarray, mode: str, threshold: float) -> np.ndarray:
    y_flat = y.reshape(-1).astype(np.float64)
    if mode == "none":
        return np.ones(len(y_flat), dtype=bool)
    if mode == "zscore":
        mu, sigma = y_flat.mean(), y_flat.std()
        if sigma < 1e-12:
            return np.ones(len(y_flat), dtype=bool)
        score = np.abs((y_flat - mu) / sigma)
    elif mode == "mad":
        median = np.median(y_flat)
        mad = np.median(np.abs(y_flat - median))
        if mad < 1e-12:
            return np.ones(len(y_flat), dtype=bool)
        score = 0.6745 * np.abs(y_flat - median) / mad
    else:
        raise ValueError(f"Unsupported label_outlier_filter: {mode}")
    return score <= threshold


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _format_table(rows: list[dict], train_ratio: float) -> str:
    header = (
        f"{'Source':<12} {'#Designs':>10} {'#Cuts Train (k)':>16} "
        f"{'#Cuts Test (k)':>15} {'Delay Min (ps)':>15} {'Delay Max (ps)':>15} "
        f"{'Delay Mean (ps)':>16} {'Delay Std (ps)':>15}"
    )
    sep = "-" * len(header)
    lines = [header, sep]
    for r in rows:
        n_train = int(r["n_cuts_total"] * train_ratio)
        n_test = r["n_cuts_total"] - n_train
        lines.append(
            f"{r['name']:<12} {r['n_designs']:>10} {n_train/1e3:>16.1f} "
            f"{n_test/1e3:>15.1f} {r['delay_min']:>15.2f} {r['delay_max']:>15.2f} "
            f"{r['delay_mean']:>16.2f} {r['delay_std']:>15.2f}"
        )
    return "\n".join(lines)


def _save_summary_csv(rows: list[dict], train_ratio: float, path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Source", "#Designs", "#Cuts Train (k)", "#Cuts Test (k)",
            "Delay Min (ps)", "Delay Max (ps)",
            "Delay Mean (ps)", "Delay Std (ps)",
        ])
        for r in rows:
            n_train = int(r["n_cuts_total"] * train_ratio)
            n_test = r["n_cuts_total"] - n_train
            writer.writerow([
                r["name"], r["n_designs"],
                f"{n_train/1e3:.1f}", f"{n_test/1e3:.1f}",
                f"{r['delay_min']:.2f}", f"{r['delay_max']:.2f}",
                f"{r['delay_mean']:.2f}", f"{r['delay_std']:.2f}",
            ])


def _save_detail_csv(all_records: dict[str, list[DesignRecord]], path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Source", "Design", "#Cuts",
            "Delay Min (ps)", "Delay Max (ps)",
            "Delay Median (ps)", "Delay Mean (ps)", "Delay Std (ps)",
        ])
        for source_name in all_records:
            for r in sorted(all_records[source_name], key=lambda x: x.design):
                writer.writerow([
                    source_name, r.design, r.n_cuts,
                    f"{r.min_delay:.2f}", f"{r.max_delay:.2f}",
                    f"{r.median_delay:.2f}", f"{r.mean_delay:.2f}", f"{r.std_delay:.2f}",
                ])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"CurPID: {os.getpid()}")
    args = parse_args()
    dataset_path = args.dataset_path
    train_ratio = args.train_ratio
    logger.info("dataset_path=%s  train_ratio=%.2f", dataset_path, train_ratio)

    source_configs = [
        ("OpenABC", [os.path.join(dataset_path, "save_json_OpenABC")]),
        ("ACE",     [os.path.join(dataset_path, "save_json_ACE")]),
        ("IWLS",    [os.path.join(dataset_path, "save_json_iwls2022"),
                     os.path.join(dataset_path, "save_json_iwls2024")]),
        ("forgeEDA",[os.path.join(dataset_path, "save_json_forgeEDA")]),
    ]

    # --- Phase 1: collect per-design data ---
    all_records: dict[str, list[DesignRecord]] = {}
    for name, dirs in source_configs:
        logger.info("=== Scanning source: %s ===", name)
        records = _collect_designs(dirs, name, n_workers=args.n_workers)
        logger.info(
            "Source %s (raw): %d designs, %d total cuts",
            name, len(records), sum(r.n_cuts for r in records),
        )

        # --- forgeEDA: stratified subsampling ---
        if name == "forgeEDA" and args.forgeeda_sample_size > 0:
            records = _stratified_sample(
                records,
                n_bins=args.forgeeda_n_bins,
                sample_size=args.forgeeda_sample_size,
                seed=args.random_seed,
            )
            logger.info(
                "Source %s (after stratified): %d designs, %d total cuts",
                name, len(records), sum(r.n_cuts for r in records),
            )

        all_records[name] = records

    # --- Phase 2: global outlier filter ---
    all_delays_raw = np.concatenate(
        [r.delays for recs in all_records.values() for r in recs]
    ) if any(all_records.values()) else np.array([], dtype=np.float64)

    global_mask = _build_label_outlier_mask(
        all_delays_raw, args.label_outlier_filter, args.label_outlier_threshold,
    )
    n_removed = int((~global_mask).sum())
    logger.info(
        "Global outlier filter: mode=%s, threshold=%.1f, raw=%d, removed=%d, kept=%d",
        args.label_outlier_filter, args.label_outlier_threshold,
        len(all_delays_raw), n_removed, int(global_mask.sum()),
    )

    # split mask back per source
    offset = 0
    rows: list[dict] = []
    for name in [c[0] for c in source_configs]:
        records = all_records[name]
        n_src = sum(r.n_cuts for r in records)
        src_mask = global_mask[offset:offset + n_src]
        offset += n_src
        filtered = all_delays_raw[offset - n_src:offset][src_mask]
        rows.append({
            "name": name,
            "n_designs": len(records),
            "n_cuts_total": int(src_mask.sum()),
            "delay_min": float(np.min(filtered)) if len(filtered) > 0 else float("nan"),
            "delay_max": float(np.max(filtered)) if len(filtered) > 0 else float("nan"),
            "delay_mean": float(np.mean(filtered)) if len(filtered) > 0 else float("nan"),
            "delay_std": float(np.std(filtered)) if len(filtered) > 0 else float("nan"),
        })
        logger.info(
            "Source %s (filtered): designs=%d, cuts=%d, min=%.2f, max=%.2f, mean=%.2f, std=%.2f",
            name, len(records), len(filtered),
            rows[-1]["delay_min"], rows[-1]["delay_max"],
            rows[-1]["delay_mean"], rows[-1]["delay_std"],
        )

    # Total
    all_filtered = all_delays_raw[global_mask]
    rows.append({
        "name": "Total",
        "n_designs": sum(r["n_designs"] for r in rows),
        "n_cuts_total": int(global_mask.sum()),
        "delay_min": float(np.min(all_filtered)) if len(all_filtered) > 0 else float("nan"),
        "delay_max": float(np.max(all_filtered)) if len(all_filtered) > 0 else float("nan"),
        "delay_mean": float(np.mean(all_filtered)) if len(all_filtered) > 0 else float("nan"),
        "delay_std": float(np.std(all_filtered)) if len(all_filtered) > 0 else float("nan"),
    })

    # --- Output ---
    table_str = _format_table(rows, train_ratio)
    print("\n" + table_str + "\n")

    if args.output_csv:
        _save_summary_csv(rows, train_ratio, args.output_csv)
        logger.info("Saved summary CSV to %s", args.output_csv)

    if args.output_detail_csv:
        _save_detail_csv(all_records, args.output_detail_csv)
        logger.info("Saved detail CSV to %s", args.output_detail_csv)


if __name__ == "__main__":
    main()
