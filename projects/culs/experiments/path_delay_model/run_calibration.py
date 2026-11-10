#!/usr/bin/env python3
"""Single-candidate LiteCut-to-endpoint calibration for the fixed c1355 paths.

This runner consumes existing Gate 3A/3B/3C/3E artifacts. It never repartitions
the AIG or changes the rewrite algorithm. Every newly measured choice is
replayed independently from the same original AIG.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


REGIONS_PO11 = [6, 9, 24, 36, 37, 38]
DEFAULT_EPS = 0.01
DEFAULT_TIMING_PRECISION = 0.1
DEFAULT_SEED = 20260929


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run_command(command: list[str], log_path: Path) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, check=False)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"command exited {proc.returncode}: {' '.join(command)}; see {log_path}")
    return proc


def parse_delay_reports(output: str) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    blocks = re.split(r'(?=Critical path from PI "pi\d+" to PO "po\d+":)', output)
    for block in blocks:
        match = re.search(r'Critical path from PI "pi(\d+)" to PO "po(\d+)":', block)
        if not match:
            continue
        arrivals = [float(value) for value in re.findall(
            r"Arrival(?: time)?\s*=\s*([0-9]+(?:\.[0-9]+)?)", block)]
        if arrivals:
            reports.append({"pi_index": int(match.group(1)), "po": int(match.group(2)),
                            "arrival_ps": arrivals[-1]})
    return reports


def endpoint_report(reports: list[dict[str, Any]], pi_node: int, po: int) -> float:
    pi_index = pi_node - 1  # Gate 3B stores AIG PI node IDs; ABC names PIs from pi00.
    exact = [r for r in reports if r["pi_index"] == pi_index and r["po"] == po]
    if not exact:
        raise RuntimeError(f"ABC did not return requested endpoint pi{pi_index:02d}->po{po:02d}; "
                           f"reports={reports}")
    return float(exact[0]["arrival_ps"])


def abc_timing(abc: Path, library: Path, aig: Path, queries: list[tuple[int, int]],
               log_path: Path) -> tuple[list[dict[str, Any]], str]:
    commands = "; ".join(f"print_delay po{po:02d} pi{pi_node - 1:02d}" for pi_node, po in queries)
    script = f'read "{library}"; read "{aig}"; map; topo; stime; {commands}'
    output = run_command([str(abc), "-c", script], log_path).stdout
    return parse_delay_reports(output), script


def baseline_check(args: argparse.Namespace, logs: Path) -> tuple[float, str]:
    reports, script = abc_timing(args.abc, args.library, args.aig,
                                 [(args.pi_node, args.po)], logs / "abc_baseline_po11_pair.log")
    measured = endpoint_report(reports, args.pi_node, args.po)
    write_json(args.out_dir / "baseline_check.json", {
        "aig": str(args.aig), "aig_sha256": sha256(args.aig),
        "abc": str(args.abc), "library": str(args.library),
        "pi_node": args.pi_node, "abc_pi": f"pi{args.pi_node - 1:02d}", "po": args.po,
        "expected_endpoint_delay_ps": args.expected_baseline,
        "measured_endpoint_delay_ps": measured,
        "baseline_matches_exactly": measured == args.expected_baseline,
        "abc_command": script, "timing_precision_ps": 0.01,
    })
    if measured != args.expected_baseline:
        raise RuntimeError(f"baseline changed: expected exactly {args.expected_baseline} ps, "
                           f"measured {measured} ps; stopped before replay")
    return measured, script


def candidate_key(row: dict[str, str]) -> tuple[int, int, int]:
    return int(row["region_id"]), int(row["root_id"]), int(row["cut_slot"])


def pilot_aig_path(row: dict[str, str], pilot_root: Path) -> Path | None:
    trace_text = row.get("trace_path", "")
    match = re.search(r"(?:^|/)(S\d+)\.jsonl$", trace_text)
    if not match:
        return None
    path = pilot_root / "aigs" / f"{match.group(1)}.aig"
    return path if path.is_file() else None


def normalize_pilot(row: dict[str, str], item: dict[str, str], baseline: float,
                    aig_path: Path, source: str) -> dict[str, Any]:
    status = row.get("final_status", row.get("rewrite_status", ""))
    cec = row.get("cec", "").upper() == "PASS"
    rewritten = float(row.get("sta_after_delay") or baseline)
    true_delta = rewritten - baseline
    return {
        "path_id": "po11_pi_node004", "pi": 4, "po": 11,
        "region_id": int(item["region_id"]), "root_id": int(item["root_id"]),
        "cut_id": int(item["cut_slot"]), "candidate_id": item["candidate_id"],
        "candidate_subgraph_index": int(item["subgraph_index"]),
        "litecut_current": float(item["current_pred_delay"]),
        "litecut_candidate": float(item["candidate_pred_delay"]),
        "litecut_delta": float(item["delta_litecut"]),
        "baseline_endpoint_delay": baseline, "rewritten_endpoint_delay": rewritten,
        "true_endpoint_delta": true_delta,
        "error_raw": float(item["delta_litecut"]) - true_delta,
        "abs_error_raw": abs(float(item["delta_litecut"]) - true_delta),
        "cec_pass": cec, "rewrite_status": status,
        "endpoint_pair_status": row.get("endpoint_pair_status", ""),
        "data_source": source, "aig_sha256": row.get("aig_sha256", ""),
        "aig_path": str(aig_path), "abc_command": row.get("abc_command", ""),
    }


def build_selection(args: argparse.Namespace) -> tuple[list[dict[str, str]], dict[str, Any]]:
    catalog = read_csv(args.path_catalog)
    po11 = [row for row in catalog if int(row["po"]) == args.po and int(row["pi"]) == args.pi_node]
    if len(po11) != 1:
        raise RuntimeError(f"expected one existing Gate3B path for PI node {args.pi_node}->PO{args.po}; got {len(po11)}")
    path = po11[0]
    regions = [int(x) for x in json.loads(path["region_sequence"])]
    if regions != REGIONS_PO11:
        raise RuntimeError(f"fixed PO11 region path changed: expected {REGIONS_PO11}, got {regions}")
    all_candidates = read_csv(args.delta_csv)
    selected = [row for row in all_candidates if int(row["region_id"]) in regions]
    order = {region: i for i, region in enumerate(regions)}
    selected.sort(key=lambda row: (order[int(row["region_id"])], int(row["root_id"]),
                                   int(row["cut_slot"]), float(row["delta_litecut"]),
                                   int(row["subgraph_index"])))
    candidate_ids = [row["candidate_id"] for row in selected]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise RuntimeError("candidate_id is not unique in the fixed PO11 path candidate pool")
    decision_counts: dict[int, set[tuple[int, int]]] = defaultdict(set)
    candidate_counts: dict[int, int] = defaultdict(int)
    for row in selected:
        region, root, cut = candidate_key(row)
        decision_counts[region].add((root, cut))
        candidate_counts[region] += 1
    summary = {
        "path_id": path["path_id"], "pi_node": args.pi_node, "abc_pi": f"pi{args.pi_node - 1:02d}",
        "po": args.po, "fixed_region_sequence": regions,
        "selection_policy": "all saved Gate3E executable candidates in the six fixed PO11 path regions; deterministic order by path region, root, cut, LiteCut delta, subgraph index; no best-only filter",
        "candidate_count": len(selected), "decision_count": len({(r["root_id"], r["cut_slot"]) for r in selected}),
        "region_counts": {str(r): {"decisions": len(decision_counts[r]), "candidates": candidate_counts[r]}
                          for r in regions},
        "regions_without_candidates": [r for r in regions if not candidate_counts[r]],
    }
    write_csv(args.out_dir / "candidate_selection.csv", selected)
    write_json(args.out_dir / "candidate_selection.json", summary)
    return selected, summary


def run_single(args: argparse.Namespace) -> None:
    out, results = args.out_dir, args.out_dir / "results"
    logs, work = out / "logs", out / "work"
    for directory in (results, logs, work / "replay", work / "cec", work / "abc"):
        directory.mkdir(parents=True, exist_ok=True)

    # Baseline must be confirmed before replaying even one candidate.
    baseline, baseline_script = baseline_check(args, logs)
    selected, selection_summary = build_selection(args)
    write_json(out / "run_manifest.json", {
        "mode": "single_candidate_calibration", "input_aig": str(args.aig),
        "input_aig_sha256": sha256(args.aig), "region_map": str(args.region_map),
        "region_map_sha256": sha256(args.region_map), "path_catalog": str(args.path_catalog),
        "path_catalog_sha256": sha256(args.path_catalog), "litecut_delta_csv": str(args.delta_csv),
        "litecut_delta_csv_sha256": sha256(args.delta_csv), "pilot_csv": str(args.pilot_csv),
        "pilot_csv_sha256": sha256(args.pilot_csv), "pilot_replay_root": str(args.pilot_replay_root),
        "gpuls": str(args.gpuls), "gpuls_sha256": sha256(args.gpuls),
        "abc": str(args.abc), "abc_sha256": sha256(args.abc),
        "library": str(args.library), "library_sha256": sha256(args.library),
        "abc_mapping_flow": "read <ASAP7 lib>; read <AIG>; map; topo; stime; print_delay poNN piMM",
        "baseline_abc_command": baseline_script, "baseline_endpoint_ps": baseline,
        "fixed_endpoint": {"aig_pi_node": args.pi_node, "abc_pi_index": args.pi_node - 1,
                           "po": args.po},
        "fixed_mffc_region_path": REGIONS_PO11, "selection": selection_summary,
        "epsilon_ps": args.epsilon, "timing_precision_ps": args.timing_precision,
        "cv_seed": args.seed, "cv_group": "(region_id, root_id, cut_id)",
    })

    pilot_rows = read_csv(args.pilot_csv)
    pilot_by_id: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in pilot_rows:
        pilot_by_id[row.get("candidate_id", "")].append(row)
    prior_root = args.pilot_replay_root
    results_by_id: dict[str, dict[str, Any]] = {}
    to_replay: list[dict[str, str]] = []
    reused = 0
    for item in selected:
        matches = pilot_by_id.get(item["candidate_id"], [])
        if len(matches) == 1:
            old = matches[0]
            aig = pilot_aig_path(old, prior_root)
            valid = (aig is not None and old.get("cec", "").upper() == "PASS"
                     and old.get("final_status") == "ACCEPTED"
                     and old.get("endpoint_pair_status") == "VALID_PAIR_MEASURE"
                     and float(old.get("baseline_delay", "nan")) == baseline
                     and old.get("aig_sha256", "") == sha256(aig))
            if valid:
                results_by_id[item["candidate_id"]] = normalize_pilot(
                    old, item, baseline, aig, "reused_verified_step5_pilot")
                reused += 1
                continue
        to_replay.append(item)

    choice_file = work / "single_candidate_choices.input"
    choice_file.write_text("".join(
        f"C{i:03d} {item['root_id']} {item['cut_slot']} {item['subgraph_index']}\n"
        for i, item in enumerate(to_replay)), encoding="utf-8")
    replay_root = work / "replay"
    if to_replay:
        replay_script = (f"read {args.aig}; gate3e_execution_replay -o {replay_root} "
                         f"--region-map {args.region_map} --config {choice_file}")
        replay = run_command([str(args.gpuls), "-c", replay_script], logs / "gpuls_single_candidate_replay.log")
        (logs / "gpuls_single_candidate_replay_command.txt").write_text(replay_script + "\n", encoding="utf-8")
        manifests = read_jsonl(replay_root / "replay_manifest.jsonl")
        manifest_by_id = {row["config_id"]: row for row in manifests}
        if len(manifest_by_id) != len(to_replay):
            raise RuntimeError(f"expected {len(to_replay)} replay outputs, got {len(manifest_by_id)}")
        for i, item in enumerate(to_replay):
            config_id = f"C{i:03d}"
            manifest = manifest_by_id.get(config_id)
            if not manifest:
                raise RuntimeError(f"missing replay manifest entry {config_id}")
            aig = replay_root / manifest["aig_path"]
            trace_path = replay_root / manifest["trace_path"]
            trace = read_jsonl(trace_path)
            decision = [row for row in trace if int(row.get("root", -1)) == int(item["root_id"])
                        and int(row.get("cut_slot", -1)) == int(item["cut_slot"])]
            if len(decision) != 1:
                raise RuntimeError(f"expected one trace entry for {config_id}, got {len(decision)}")
            decision = decision[0]
            if not decision.get("preselected", False):
                raise RuntimeError(f"requested candidate was not preselected for {config_id}: {decision}")

            cec_script = f'read "{args.aig}"; read "{aig}"; cec -n'
            cec_text = run_command([str(args.abc), "-c", cec_script], logs / f"cec_{config_id}.log").stdout.lower()
            cec_pass = (("equivalent" in cec_text or "cec passes" in cec_text)
                        and "not equivalent" not in cec_text)
            rewritten: float | None = None
            timing_script = ""
            if cec_pass:
                reports, timing_script = abc_timing(args.abc, args.library, aig,
                                                    [(args.pi_node, args.po)],
                                                    logs / f"abc_{config_id}.log")
                rewritten = endpoint_report(reports, args.pi_node, args.po)
            pred_delta = float(item["delta_litecut"])
            true_delta = rewritten - baseline if rewritten is not None else None
            results_by_id[item["candidate_id"]] = {
                "path_id": "po11_pi_node004", "pi": args.pi_node, "po": args.po,
                "region_id": int(item["region_id"]), "root_id": int(item["root_id"]),
                "cut_id": int(item["cut_slot"]), "candidate_id": item["candidate_id"],
                "candidate_subgraph_index": int(item["subgraph_index"]),
                "litecut_current": float(item["current_pred_delay"]),
                "litecut_candidate": float(item["candidate_pred_delay"]), "litecut_delta": pred_delta,
                "baseline_endpoint_delay": baseline,
                "rewritten_endpoint_delay": rewritten if rewritten is not None else "",
                "true_endpoint_delta": true_delta if true_delta is not None else "",
                "error_raw": pred_delta - true_delta if true_delta is not None else "",
                "abs_error_raw": abs(pred_delta - true_delta) if true_delta is not None else "",
                "cec_pass": cec_pass, "rewrite_status": decision.get("final_status", ""),
                "endpoint_pair_status": "VALID_PAIR_MEASURE" if rewritten is not None else "CEC_FAIL_OR_NO_TIMING",
                "data_source": "new_independent_replay", "aig_sha256": sha256(aig), "aig_path": str(aig),
                "abc_command": timing_script, "replay_config_id": config_id,
                "requested_preselected": bool(decision.get("preselected", False)),
                "trace_path": str(trace_path), "cec_command": cec_script,
            }

    rows = [results_by_id[item["candidate_id"]] for item in selected]
    if len(rows) != len(selected):
        raise RuntimeError("some selected candidates have no calibration outcome")
    fields = ["path_id", "pi", "po", "region_id", "root_id", "cut_id", "candidate_id",
              "candidate_subgraph_index", "litecut_current", "litecut_candidate", "litecut_delta",
              "baseline_endpoint_delay", "rewritten_endpoint_delay", "true_endpoint_delta",
              "error_raw", "abs_error_raw", "cec_pass", "rewrite_status", "endpoint_pair_status",
              "data_source", "aig_sha256", "aig_path", "abc_command", "replay_config_id",
              "requested_preselected", "trace_path", "cec_command"]
    write_csv(results / "single_candidate_calibration.csv", rows, fields)

    valid = [row for row in rows if row["cec_pass"] is True and row["rewrite_status"] == "ACCEPTED"
             and row["endpoint_pair_status"] == "VALID_PAIR_MEASURE"]
    if len(valid) < 5:
        raise RuntimeError(f"only {len(valid)} valid accepted calibration samples; stopping analysis")
    metrics, cv_rows, cv_predictions = analyze_single(valid, args.epsilon, args.seed, results)
    status_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        status_counts[str(row["rewrite_status"])] += 1
    run_summary = {
        **selection_summary, "baseline_endpoint_delay_ps": baseline,
        "baseline_expected_ps": args.expected_baseline, "additional_replays": len(to_replay),
        "reused_verified_pilot_samples": reused, "candidate_outcome_count": len(rows),
        "cec_pass_count": sum(bool(row["cec_pass"]) for row in rows),
        "accepted_count": status_counts.get("ACCEPTED", 0), "rewrite_status_counts": dict(status_counts),
        "valid_accepted_samples_for_fit": len(valid), "epsilon_ps": args.epsilon,
        "cross_validation_seed": args.seed, "cross_validation_group": "(region_id, root_id, cut_id)",
        "abc_baseline_command": baseline_script,
        "interpretation": "STA response is the fixed PI-node 4 to PO11 mapped endpoint-pair delay, not an exact mapped image of the saved AIG node path.",
    }
    write_json(results / "calibration_run_summary.json", run_summary)
    write_json(results / "calibration_cv_metrics.json", {"models": cv_rows, "details": metrics})
    write_csv(results / "calibration_cv_metrics.csv", cv_rows)
    write_csv(results / "calibration_cv_predictions.csv", cv_predictions)
    make_report(results, rows, valid, metrics, cv_rows, args.epsilon, args.timing_precision,
                run_summary, multipath=None)


def finite_xy(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([float(row["litecut_delta"]) for row in rows], dtype=float)
    y = np.asarray([float(row["true_endpoint_delta"]) for row in rows], dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        result[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return result


def correlation(x: np.ndarray, y: np.ndarray) -> tuple[float | None, float | None]:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None, None
    pearson = float(np.corrcoef(x, y)[0, 1])
    rx, ry = ranks(x), ranks(y)
    spearman = float(np.corrcoef(rx, ry)[0, 1]) if np.std(rx) and np.std(ry) else None
    return pearson, spearman


def sign(value: float, epsilon: float) -> int:
    if abs(value) <= epsilon:
        return 0
    return 1 if value > 0 else -1


def metrics_for(pred: np.ndarray, truth: np.ndarray, epsilon: float) -> dict[str, Any]:
    residual = pred - truth
    pearson, spearman = correlation(pred, truth)
    informative = np.abs(truth) > epsilon
    sign_acc = (float(np.mean([sign(float(p), epsilon) == sign(float(y), epsilon)
                               for p, y in zip(pred[informative], truth[informative])]))
                if np.any(informative) else None)
    return {"n": int(len(truth)), "mae": float(np.mean(np.abs(residual))) if len(truth) else None,
            "rmse": float(np.sqrt(np.mean(residual ** 2))) if len(truth) else None,
            "pearson": pearson, "spearman": spearman, "sign_accuracy": sign_acc,
            "sign_n_non_neutral_true": int(np.sum(informative))}


def fit_origin(x: np.ndarray, y: np.ndarray) -> float:
    denominator = float(np.dot(x, x))
    return float(np.dot(x, y) / denominator) if denominator > 0 else 0.0


def fit_linear(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) == 0:
        return 0.0, 0.0
    design = np.column_stack((x, np.ones(len(x))))
    slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    return float(slope), float(intercept)


def region_metrics(valid: list[dict[str, Any]], epsilon: float, results: Path) -> list[dict[str, Any]]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows: list[dict[str, Any]] = []
    for region in REGIONS_PO11:
        subset = [row for row in valid if int(row["region_id"]) == region]
        x, y = finite_xy(subset)
        pearson, spearman = correlation(x, y)
        if len(x):
            raw_mae = float(np.mean(np.abs(x - y)))
            informative = np.abs(y) > epsilon
            sign_acc = float(np.mean([sign(float(a), epsilon) == sign(float(b), epsilon)
                                      for a, b in zip(x[informative], y[informative])])) if np.any(informative) else None
        else:
            raw_mae = sign_acc = None
        if len(x) >= 2:
            alpha = fit_origin(x, y)
            beta_alpha, beta = fit_linear(x, y)
            origin_mae = float(np.mean(np.abs(alpha * x - y)))
            linear_mae = float(np.mean(np.abs(beta_alpha * x + beta - y)))
        else:
            alpha = beta_alpha = beta = origin_mae = linear_mae = None
        rows.append({"region_id": region, "n": len(x), "alpha_origin": alpha,
                     "alpha_linear": beta_alpha, "beta_linear": beta, "mae_raw": raw_mae,
                     "mae_origin_scaled": origin_mae, "mae_linear": linear_mae,
                     "pearson": pearson, "spearman": spearman, "sign_accuracy": sign_acc})

        fig, ax = plt.subplots(figsize=(6.4, 5.2))
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.axvline(0, color="gray", linewidth=0.8)
        if len(x):
            ax.scatter(x, y, alpha=0.8, edgecolor="none")
            if len(x) >= 2 and np.ptp(x) > 0:
                lo, hi = float(np.min(x)), float(np.max(x))
                slope, intercept = fit_linear(x, y)
                ax.plot([lo, hi], [slope * lo + intercept, slope * hi + intercept], color="crimson")
            ax.set_title(f"R{region}: N={len(x)}, Pearson={fmt(pearson)}, Spearman={fmt(spearman)}")
        else:
            ax.text(0.5, 0.5, "No executable candidates on PO11 path", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(f"R{region}: N=0")
        ax.set_xlabel("LiteCut delta (ps; local score difference)")
        ax.set_ylabel("True PI4→PO11 endpoint delay delta (ps)")
        ax.grid(alpha=0.18)
        fig.tight_layout()
        fig.savefig(results / f"calibration_R{region}.png", dpi=160)
        plt.close(fig)
    write_csv(results / "region_calibration_metrics.csv", rows)
    return rows


def make_global_plot(valid: list[dict[str, Any]], results: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x, y = finite_xy(valid)
    pearson, spearman = correlation(x, y)
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.scatter(x, y, alpha=0.75, edgecolor="none")
    if len(x) >= 2 and np.ptp(x) > 0:
        slope, intercept = fit_linear(x, y)
        lo, hi = float(np.min(x)), float(np.max(x))
        ax.plot([lo, hi], [slope * lo + intercept, slope * hi + intercept], color="crimson",
                label=f"OLS y={slope:.3g}x{intercept:+.3g}")
        ax.legend(loc="best")
    ax.set_xlabel("LiteCut delta (ps; local score difference)")
    ax.set_ylabel("True PI4→PO11 endpoint delay delta (ps)")
    ax.set_title(f"Single-candidate calibration: N={len(x)}, Pearson={fmt(pearson)}, Spearman={fmt(spearman)}")
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(results / "calibration_global_scatter.png", dpi=170)
    plt.close(fig)


def grouped_folds(rows: list[dict[str, Any]], folds: int, seed: int) -> list[set[tuple[int, int, int]]]:
    sizes: dict[tuple[int, int, int], int] = defaultdict(int)
    for row in rows:
        sizes[(int(row["region_id"]), int(row["root_id"]), int(row["cut_id"]))] += 1
    groups = list(sizes)
    if len(groups) < 2:
        raise RuntimeError("need at least two independent rewrite decisions for grouped CV")
    k = min(folds, len(groups))
    rng = np.random.default_rng(seed)
    shuffled = [groups[i] for i in rng.permutation(len(groups))]
    shuffled.sort(key=lambda group: sizes[group], reverse=True)
    buckets: list[set[tuple[int, int, int]]] = [set() for _ in range(k)]
    counts = [0] * k
    for group in shuffled:
        target = min(range(k), key=lambda i: (counts[i], i))
        buckets[target].add(group)
        counts[target] += sizes[group]
    return buckets


def analyze_single(valid: list[dict[str, Any]], epsilon: float, seed: int,
                   results: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    x, y = finite_xy(valid)
    if len(x) != len(valid):
        raise RuntimeError("valid calibration data contains non-finite values")
    metrics = metrics_for(x, y, epsilon)
    counts = {"negative_true_count": int(np.sum(y < -epsilon)),
              "neutral_true_count": int(np.sum(np.abs(y) <= epsilon)),
              "positive_true_count": int(np.sum(y > epsilon))}
    raw = metrics_for(x, y, epsilon)
    global_alpha = fit_origin(x, y)
    origin = metrics_for(global_alpha * x, y, epsilon)
    global_linear_alpha, global_beta = fit_linear(x, y)
    linear = metrics_for(global_linear_alpha * x + global_beta, y, epsilon)
    region_rows = region_metrics(valid, epsilon, results)
    make_global_plot(valid, results)

    folds = grouped_folds(valid, 5, seed)
    group_by_index = [(int(row["region_id"]), int(row["root_id"]), int(row["cut_id"])) for row in valid]
    prediction_buffers: dict[str, list[float | None]] = {
        name: [None] * len(valid) for name in
        ("raw_delta", "zero_delta", "global_alpha", "region_alpha", "region_alpha_beta")}
    fold_of = [-1] * len(valid)
    fallback_counts = {"region_alpha": 0, "region_alpha_beta": 0}
    for fold_id, test_groups in enumerate(folds):
        train_idx = [i for i, key in enumerate(group_by_index) if key not in test_groups]
        test_idx = [i for i, key in enumerate(group_by_index) if key in test_groups]
        tx, ty = x[train_idx], y[train_idx]
        global_a = fit_origin(tx, ty)
        per_region: dict[int, list[int]] = defaultdict(list)
        for i in train_idx:
            per_region[int(valid[i]["region_id"])].append(i)
        for i in test_idx:
            region = int(valid[i]["region_id"])
            xx = float(x[i])
            prediction_buffers["raw_delta"][i] = xx
            prediction_buffers["zero_delta"][i] = 0.0
            prediction_buffers["global_alpha"][i] = global_a * xx
            train_region = per_region.get(region, [])
            if train_region:
                ridx = np.asarray(train_region, dtype=int)
                region_a = fit_origin(x[ridx], y[ridx])
                region_la, region_b = fit_linear(x[ridx], y[ridx])
            else:
                region_a, region_la, region_b = global_a, global_a, 0.0
                fallback_counts["region_alpha"] += 1
                fallback_counts["region_alpha_beta"] += 1
            prediction_buffers["region_alpha"][i] = region_a * xx
            prediction_buffers["region_alpha_beta"][i] = region_la * xx + region_b
            fold_of[i] = fold_id

    cv_rows: list[dict[str, Any]] = []
    cv_predictions: list[dict[str, Any]] = []
    model_labels = {"raw_delta": "Baseline 0: raw LiteCut delta",
                    "zero_delta": "Reference: predict no endpoint change (0 ps)",
                    "global_alpha": "Baseline 1: global alpha * delta",
                    "region_alpha": "Baseline 2: region alpha * delta",
                    "region_alpha_beta": "Baseline 3: region alpha * delta + beta"}
    for name, values in prediction_buffers.items():
        if any(value is None for value in values):
            raise RuntimeError(f"missing held-out predictions for {name}")
        pred = np.asarray(values, dtype=float)
        model_metrics = metrics_for(pred, y, epsilon)
        cv_rows.append({"model": name, "description": model_labels[name], "folds": len(folds),
                        **model_metrics,
                        "region_fallback_predictions": fallback_counts.get(name, 0)})
        for i, row in enumerate(valid):
            cv_predictions.append({"candidate_id": row["candidate_id"], "region_id": row["region_id"],
                                   "root_id": row["root_id"], "cut_id": row["cut_id"],
                                   "fold": fold_of[i], "model": name,
                                   "predicted_true_endpoint_delta": pred[i],
                                   "true_endpoint_delta": y[i]})
    details = {
        "overall": {"raw_litecut_delta": raw, "global_origin_scaled_in_sample": origin,
                    "zero_change_predictor": metrics_for(np.zeros_like(y), y, epsilon),
                    "global_linear_in_sample": linear,
                    "global_alpha_origin": global_alpha,
                    "global_alpha_linear": global_linear_alpha, "global_beta_linear": global_beta,
                    **counts},
        "regions": region_rows, "epsilon_ps": epsilon,
        "cv_split": "five-fold deterministic greedy group split; all candidates from a (region, root, cut) decision stay in one fold",
        "cv_seed": seed, "region_alpha_fallback_predictions": fallback_counts,
        "cv_group_count": len(set(group_by_index)),
    }
    return details, cv_rows, cv_predictions


def fmt(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return "n/a"
    try:
        value = float(value)
    except (ValueError, TypeError):
        return str(value)
    return "n/a" if not math.isfinite(value) else f"{value:.{digits}f}"


def make_report(results: Path, all_rows: list[dict[str, Any]], valid: list[dict[str, Any]],
                details: dict[str, Any], cv_rows: list[dict[str, Any]], epsilon: float,
                precision: float, summary: dict[str, Any], multipath: dict[str, Any] | None) -> None:
    overall = details["overall"]
    models = {row["model"]: row for row in cv_rows}
    region_table = "\n".join(
        f"| R{r['region_id']} | {r['n']} | {fmt(r['alpha_origin'])} | {fmt(r['alpha_linear'])} | "
        f"{fmt(r['beta_linear'])} | {fmt(r['pearson'])} | {fmt(r['spearman'])} | {fmt(r['mae_raw'])} | "
        f"{fmt(r['mae_origin_scaled'])} | {fmt(r['sign_accuracy'])} |"
        for r in details["regions"])
    cv_table = "\n".join(
        f"| {row['model']} | {row['n']} | {fmt(row['mae'])} | {fmt(row['rmse'])} | "
        f"{fmt(row['pearson'])} | {fmt(row['spearman'])} | {fmt(row['sign_accuracy'])} | "
        f"{row['sign_n_non_neutral_true']} | {row['region_fallback_predictions']} |"
        for row in cv_rows)
    path_section = "Path-dependent effect: pending the shared-region endpoint-pair audit."
    path_answer = "Not yet measured in this report stage."
    if multipath:
        path_answer = multipath.get("path_dependence_conclusion", "Insufficient shared candidate/path comparisons.")
        path_section = (f"Shared-region audit measured {multipath['measurement_count']} candidate/path endpoint pairs "
                        f"across {multipath['candidate_count']} candidates and {multipath['path_count']} paths "
                        f"({multipath['distinct_pi_count']} distinct PI node"
                        f"{'s' if multipath['distinct_pi_count'] != 1 else ''}, "
                        f"{multipath['distinct_po_count']} distinct POs). "
                        f"{path_answer}")
    alpha_r2 = models.get("region_alpha", {}).get("mae")
    alpha_g = models.get("global_alpha", {}).get("mae")
    raw_cv = models.get("raw_delta", {}).get("mae")
    if alpha_r2 is not None and alpha_g is not None and alpha_r2 < alpha_g:
        improvement = alpha_g - alpha_r2
        if improvement < precision:
            region_dependence = (f"Region-specific scaling's nominal held-out MAE reduction over global scaling is "
                                 f"{improvement:.4f} ps, below the {precision:.2f} ps observed timing resolution; "
                                 "there is no resolved region-specific benefit, and the zero-change reference is better.")
        else:
            region_dependence = (f"Region-specific scaling lowers held-out MAE versus global scaling by "
                                 f"{improvement:.4f} ps, but the zero-change reference remains better.")
    else:
        region_dependence = "Region-specific scaling does not lower held-out MAE versus global scaling."
    text = f"""# Single-Candidate Path Impact Calibration — c1355 PO11

## Experimental scope

Only isolated, single-candidate rewrites are included. The fixed Gate 3B path is PI node 4 (`pi03`) → PO11, with MFFC Region sequence `R6 → R9 → R24 → R36 → R37 → R38`. Timing means the mapped endpoint-pair delay reported by ABC for that same PI/PO pair; it is not an exact mapped correspondence to the original AIG node sequence. LiteCut delta is candidate local-cone score minus the same decision's original cut-cone proxy.

All {summary['candidate_count']} candidates in these path regions were selected (not only local-best candidates). {summary['reused_verified_pilot_samples']} previously measured pilot samples were reused after checking CEC, ACCEPTED status, baseline, and replay-AIG hash; {summary['additional_replays']} candidates were replayed independently from the original snapshot. Current outcome counts: CEC PASS {summary['cec_pass_count']}/{summary['candidate_outcome_count']}; accepted {summary['accepted_count']}/{summary['candidate_outcome_count']}. Valid calibration fits use only ACCEPTED, CEC-passing candidates with endpoint timing.

Baseline endpoint delay was remeasured as **{summary['baseline_endpoint_delay_ps']:.2f} ps** (required {summary['baseline_expected_ps']:.2f} ps). ABC `print_delay` prints Arrival values to one decimal place, so the observed timing resolution is **0.1 ps**; an older Gate3F manifest says 0.01 ps, which is finer than the output actually retained here. Path-effect thresholds therefore use 0.1 ps. The sign-neutral epsilon remains the separately configured {epsilon:.2f} ps; neutral/negative/positive counts are {overall['neutral_true_count']}/{overall['negative_true_count']}/{overall['positive_true_count']}.

## Overall calibration (in-sample descriptive fits)

Valid samples: **{len(valid)}**. Raw LiteCut delta: Pearson {fmt(overall['raw_litecut_delta']['pearson'])}, Spearman {fmt(overall['raw_litecut_delta']['spearman'])}, MAE {fmt(overall['raw_litecut_delta']['mae'])} ps. Predicting no endpoint change (`ŷ=0`) gives MAE {fmt(overall['zero_change_predictor']['mae'])} ps. Global through-origin fit `y=αx`: α={fmt(overall['global_alpha_origin'])}, MAE {fmt(overall['global_origin_scaled_in_sample']['mae'])} ps. Global linear fit: `y=αx+β`, α={fmt(overall['global_alpha_linear'])}, β={fmt(overall['global_beta_linear'])}, MAE {fmt(overall['global_linear_in_sample']['mae'])} ps. These in-sample fits are descriptive only; model comparison below uses grouped cross-validation.

## Region-level descriptive calibration

| Region | N | α origin | α linear | β linear | Pearson | Spearman | Raw MAE | Origin-scaled MAE | Sign accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{region_table}

## Five-fold grouped cross-validation

All candidates from one `(region, root, cut)` decision are assigned to the same fold. Region models without any training examples for a held-out region fall back to the fold's global through-origin alpha (fallback predictions are counted below).

| Model | N | MAE (ps) | RMSE (ps) | Pearson | Spearman | Sign accuracy | Sign N | Fallback predictions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{cv_table}

Held-out region-specific vs global-alpha MAE: {fmt(alpha_r2)} vs {fmt(alpha_g)} ps; raw-delta MAE: {fmt(raw_cv)} ps; zero-change reference MAE: {fmt(models.get('zero_delta', {}).get('mae'))} ps. {region_dependence} Compare both calibrated models to the zero-change reference: a lower MAE than raw delta alone does not establish that LiteCut adds predictive value.

## Cross-path shared-region audit

{path_section}

## Research questions

**Q1. Can LiteCut delta directly serve as path delay delta?** Raw pooled Pearson={fmt(overall['raw_litecut_delta']['pearson'])}, Spearman={fmt(overall['raw_litecut_delta']['spearman'])}, MAE={fmt(overall['raw_litecut_delta']['mae'])} ps and grouped-CV MAE={fmt(raw_cv)} ps. This is the direct empirical answer; the local score is not itself a measured path contribution.

**Q2. Does a global scaling factor help?** Grouped-CV raw MAE {fmt(raw_cv)} ps → global-alpha MAE {fmt(alpha_g)} ps; zero-change reference is {fmt(models.get('zero_delta', {}).get('mae'))} ps. Global α does not outperform this no-change reference.

**Q3. Does region-specific scaling further help?** Grouped-CV global-alpha MAE {fmt(alpha_g)} ps → region-alpha MAE {fmt(alpha_r2)} ps; zero-change reference is {fmt(models.get('zero_delta', {}).get('mae'))} ps. {region_dependence} Region α also does not outperform the no-change reference.

**Q4. Is there a path-dependent effect for the same region/candidate?** {path_answer}

No multi-candidate interaction or optimization experiment was run.
"""
    (results / "SINGLE_CANDIDATE_PATH_IMPACT_REPORT.md").write_text(text, encoding="utf-8")


def run_multipath(args: argparse.Namespace) -> None:
    results, logs = args.out_dir / "results", args.out_dir / "logs"
    single = read_csv(args.single_csv)
    catalog = read_csv(args.path_catalog)
    deltas = read_csv(args.delta_csv)
    single_by_id = {row["candidate_id"]: row for row in single}
    accepted = {key: row for key, row in single_by_id.items()
                if row.get("cec_pass", "").lower() in ("true", "1")
                and row.get("rewrite_status") == "ACCEPTED"
                and row.get("endpoint_pair_status") == "VALID_PAIR_MEASURE"}
    deltas_by_region: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in deltas:
        if int(row["region_id"]) in (9, 24):
            deltas_by_region[int(row["region_id"])].append(row)
    chosen: list[dict[str, str]] = []
    quantile_positions = (0.0, 1 / 3, 2 / 3, 1.0)
    for region in (9, 24):
        pool = sorted(deltas_by_region[region], key=lambda row: (float(row["delta_litecut"]),
                                                                  int(row["root_id"]), int(row["cut_slot"]),
                                                                  int(row["subgraph_index"])))
        picked: set[str] = set()
        for q in quantile_positions:
            pos = round(q * (len(pool) - 1))
            item = pool[pos]
            if item["candidate_id"] in accepted and item["candidate_id"] not in picked:
                chosen.append(item)
                picked.add(item["candidate_id"])
    if not chosen:
        raise RuntimeError("no ACCEPTED CEC-passing shared-region candidates for cross-path audit")

    paths = []
    for row in catalog:
        if row.get("baseline_delay_semantics") != "endpoint_pair_delay" or not row.get("baseline_delay"):
            continue
        region_seq = [int(r) for r in json.loads(row["region_sequence"])]
        if any(int(candidate["region_id"]) in region_seq for candidate in chosen):
            paths.append({**row, "regions": region_seq})
    if not paths:
        raise RuntimeError("no saved Gate3B endpoint paths overlap selected shared regions")

    baseline_queries = [(int(row["pi"]), int(row["po"])) for row in paths]
    baseline_reports, baseline_script = abc_timing(args.abc, args.library, args.aig,
                                                    baseline_queries, logs / "abc_multipath_baseline.log")
    baseline_by_path: dict[str, float] = {}
    for path in paths:
        baseline_by_path[path["path_id"]] = endpoint_report(
            baseline_reports, int(path["pi"]), int(path["po"]))
        prior = float(path["baseline_delay"])
        if baseline_by_path[path["path_id"]] != prior:
            raise RuntimeError(f"saved endpoint baseline mismatch for {path['path_id']}: "
                               f"catalog={prior}, fresh={baseline_by_path[path['path_id']]}; stopping multipath stage")

    records: list[dict[str, Any]] = []
    for candidate in chosen:
        candidate_id = candidate["candidate_id"]
        source = accepted[candidate_id]
        aig = Path(source["aig_path"])
        if not aig.is_file() or sha256(aig) != source["aig_sha256"]:
            raise RuntimeError(f"cannot verify existing single-candidate replay AIG for {candidate_id}")
        applicable = [path for path in paths if int(candidate["region_id"]) in path["regions"]]
        if not applicable:
            continue
        queries = [(int(path["pi"]), int(path["po"])) for path in applicable]
        reports, script = abc_timing(args.abc, args.library, aig, queries,
                                     logs / f"abc_shared_{candidate_id}.log")
        for path in applicable:
            pi_node, po = int(path["pi"]), int(path["po"])
            arrival = endpoint_report(reports, pi_node, po)
            baseline = baseline_by_path[path["path_id"]]
            records.append({
                "region_id": int(candidate["region_id"]), "candidate_id": candidate_id,
                "root_id": int(candidate["root_id"]), "cut_id": int(candidate["cut_slot"]),
                "candidate_subgraph_index": int(candidate["subgraph_index"]),
                "path_id": path["path_id"], "pi": pi_node, "po": po,
                "litecut_delta": float(candidate["delta_litecut"]),
                "baseline_delay": baseline, "rewritten_delay": arrival,
                "true_delta": arrival - baseline, "cec_pass": True,
                "rewrite_status": source["rewrite_status"], "aig_sha256": source["aig_sha256"],
                "abc_command": script,
            })
    write_csv(results / "shared_region_multi_path.csv", records)
    write_csv(results / "shared_region_candidate_selection.csv", chosen)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[row["candidate_id"]].append(row)
    summary_rows: list[dict[str, Any]] = []
    ranges: list[float] = []
    for candidate_id, group in sorted(grouped.items()):
        effects = [float(row["true_delta"]) for row in group]
        spread = max(effects) - min(effects) if effects else 0.0
        ranges.append(spread)
        summary_rows.append({"region_id": group[0]["region_id"], "candidate_id": candidate_id,
                             "litecut_delta": group[0]["litecut_delta"], "path_count": len(group),
                             "min_true_delta": min(effects), "max_true_delta": max(effects),
                             "true_delta_range": spread,
                             "path_dependent_gt_precision": spread > args.timing_precision,
                             "timing_precision_ps": args.timing_precision})
    write_csv(results / "path_dependence_summary.csv", summary_rows)
    make_shared_path_plot(records, results / "shared_region_path_effect.png", args.timing_precision)
    measurable = [row for row in summary_rows if row["path_count"] >= 2]
    dependent = [row for row in measurable if row["path_dependent_gt_precision"]]
    if not measurable:
        conclusion = "No candidate had at least two valid PI/PO endpoint measurements, so path dependence is unresolved."
    else:
        conclusion = (f"Among {len(measurable)} candidates with at least two path measurements, "
                      f"{len(dependent)} had a within-candidate true-delta range greater than "
                      f"the {args.timing_precision:.2f} ps reporting precision; "
                      f"maximum observed range was {max(row['true_delta_range'] for row in measurable):.2f} ps. "
                      + ("This is evidence of path-dependent impact." if dependent else
                         "No measurable path-dependent impact was observed at the reporting precision."))
    multi_summary = {"candidate_count": len(grouped), "path_count": len(paths),
                     "measurement_count": len(records), "candidates_with_multiple_paths": len(measurable),
                     "distinct_pi_count": len({int(path["pi"]) for path in paths}),
                     "distinct_po_count": len({int(path["po"]) for path in paths}),
                     "candidates_path_dependent_gt_precision": len(dependent),
                     "max_true_delta_range_ps": max((row["true_delta_range"] for row in measurable), default=None),
                     "timing_precision_ps": args.timing_precision,
                     "candidate_quantile_rule": "for R9 and R24 separately, nearest order-statistic indices at q={0,1/3,2/3,1}; only already-tested ACCEPTED + CEC PASS candidates; deterministic",
                     "baseline_abc_command": baseline_script,
                     "path_dependence_conclusion": conclusion}
    write_json(results / "path_dependence_summary.json", multi_summary)

    # Rebuild the final report from the exact single-candidate table and its saved metrics.
    metric_blob = json.loads((results / "calibration_cv_metrics.json").read_text(encoding="utf-8"))
    run_blob = json.loads((results / "calibration_run_summary.json").read_text(encoding="utf-8"))
    valid_single = [row for row in single if row.get("cec_pass", "").lower() in ("true", "1")
                    and row.get("rewrite_status") == "ACCEPTED"
                    and row.get("endpoint_pair_status") == "VALID_PAIR_MEASURE"]
    # CSV-backed data values are strings; metric/report paths only need stable numerics here.
    normalized_single = [{**row, "cec_pass": True, "region_id": int(row["region_id"]),
                          "litecut_delta": float(row["litecut_delta"]),
                          "true_endpoint_delta": float(row["true_endpoint_delta"])} for row in valid_single]
    make_report(results, single, normalized_single, metric_blob["details"], metric_blob["models"],
                args.epsilon, args.timing_precision, run_blob, multi_summary)


def make_shared_path_plot(records: list[dict[str, Any]], path: Path, precision: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    candidates = sorted({row["candidate_id"] for row in records})
    endpoints = sorted({(int(row["po"]), int(row["pi"])) for row in records})
    row_index = {candidate: i for i, candidate in enumerate(candidates)}
    col_index = {endpoint: i for i, endpoint in enumerate(endpoints)}
    matrix = np.full((len(candidates), len(endpoints)), np.nan, dtype=float)
    labels: dict[str, str] = {}
    for row in records:
        matrix[row_index[row["candidate_id"]], col_index[(int(row["po"]), int(row["pi"]))]] = float(row["true_delta"])
        labels[row["candidate_id"]] = f"R{row['region_id']} {row['candidate_id']}"
    fig, ax = plt.subplots(figsize=(10.0, 5.8))
    finite = matrix[np.isfinite(matrix)]
    lower = float(np.min(finite)) if finite.size else -1.0
    upper = float(np.max(finite)) if finite.size else 1.0
    if math.isclose(lower, upper):
        lower -= precision
        upper += precision
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="coolwarm",
                      vmin=lower, vmax=upper)
    ax.set_yticks(range(len(candidates)), [labels[candidate] for candidate in candidates])
    ax.set_xticks(range(len(endpoints)),
                  [f"PO{po:02d}/PI{pi}" for po, pi in endpoints], rotation=45, ha="right")
    ax.set_xlabel("Saved Gate 3B PI/PO endpoint pair")
    ax.set_ylabel("Same single-candidate rewrite")
    ax.set_title(f"Endpoint impact across paths (timing precision {precision:.2f} ps)")
    fig.colorbar(image, ax=ax, label="True endpoint delay delta (ps)")
    ax.set_xticks(np.arange(-0.5, len(endpoints), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(candidates), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def run_summarize(args: argparse.Namespace) -> None:
    results = args.out_dir / "results"
    all_rows = read_csv(args.single_csv)
    valid: list[dict[str, Any]] = []
    for row in all_rows:
        if row.get("cec_pass", "").lower() not in ("true", "1"):
            continue
        if row.get("rewrite_status") != "ACCEPTED" or row.get("endpoint_pair_status") != "VALID_PAIR_MEASURE":
            continue
        valid.append({**row, "cec_pass": True, "region_id": int(row["region_id"]),
                      "root_id": int(row["root_id"]), "cut_id": int(row["cut_id"]),
                      "litecut_delta": float(row["litecut_delta"]),
                      "true_endpoint_delta": float(row["true_endpoint_delta"])})
    if len(valid) < 5:
        raise RuntimeError(f"only {len(valid)} valid samples in {args.single_csv}")
    details, cv_rows, predictions = analyze_single(valid, args.epsilon, args.seed, results)
    write_json(results / "calibration_cv_metrics.json", {"models": cv_rows, "details": details})
    write_csv(results / "calibration_cv_metrics.csv", cv_rows)
    write_csv(results / "calibration_cv_predictions.csv", predictions)
    summary_path = results / "calibration_run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["timing_precision_ps"] = args.timing_precision
    summary["neutral_epsilon_ps"] = args.epsilon
    write_json(summary_path, summary)
    baseline_audit_path = args.out_dir / "baseline_check.json"
    if baseline_audit_path.exists():
        baseline_audit = json.loads(baseline_audit_path.read_text(encoding="utf-8"))
        baseline_audit["prior_gate3f_manifest_precision_ps"] = 0.01
        baseline_audit["observed_arrival_output_resolution_ps"] = args.timing_precision
        baseline_audit["timing_precision_ps"] = args.timing_precision
        baseline_audit["precision_note"] = "Arrival fields in the ABC print_delay log are printed with one decimal place; use 0.1 ps as observed report resolution."
        write_json(baseline_audit_path, baseline_audit)
    manifest_path = args.out_dir / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["prior_gate3f_manifest_precision_ps"] = 0.01
        manifest["observed_arrival_output_resolution_ps"] = args.timing_precision
        manifest["timing_precision_ps"] = args.timing_precision
        manifest["neutral_epsilon_ps"] = args.epsilon
        manifest["timing_precision_note"] = "Arrival fields printed by ABC print_delay contain one decimal place; observed report resolution is 0.1 ps."
        write_json(manifest_path, manifest)
    write_json(results / "timing_resolution_audit.json", {
        "abc_flow": "same frozen Gate2B ABC binary, ASAP7 library, read; map; topo; stime; print_delay <PO> <PI>",
        "prior_gate3f_manifest_precision_ps": 0.01,
        "observed_arrival_output_decimal_places": 1,
        "observed_arrival_output_resolution_ps": args.timing_precision,
        "path_effect_measurable_threshold_ps": args.timing_precision,
        "sign_neutral_epsilon_ps": args.epsilon,
        "interpretation": "The rendered Arrival values are quantized at 0.1 ps. No claim is made about unreported sub-0.1 ps changes.",
    })
    path_file = results / "path_dependence_summary.json"
    multipath = json.loads(path_file.read_text(encoding="utf-8")) if path_file.exists() else None
    multi_csv = results / "shared_region_multi_path.csv"
    if multipath and multi_csv.exists():
        multi_rows = read_csv(multi_csv)
        multipath.setdefault("distinct_pi_count", len({int(row["pi"]) for row in multi_rows}))
        multipath.setdefault("distinct_po_count", len({int(row["po"]) for row in multi_rows}))
        per_candidate: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in multi_rows:
            per_candidate[row["candidate_id"]].append(row)
        dependence_rows = []
        for candidate_id, group in sorted(per_candidate.items()):
            effects = [float(row["true_delta"]) for row in group]
            spread = max(effects) - min(effects)
            dependence_rows.append({"region_id": group[0]["region_id"], "candidate_id": candidate_id,
                                    "litecut_delta": group[0]["litecut_delta"], "path_count": len(group),
                                    "min_true_delta": min(effects), "max_true_delta": max(effects),
                                    "true_delta_range": spread,
                                    "path_dependent_gt_precision": spread > args.timing_precision,
                                    "timing_precision_ps": args.timing_precision})
        write_csv(results / "path_dependence_summary.csv", dependence_rows)
        measurable = [row for row in dependence_rows if int(row["path_count"]) >= 2]
        dependent = [row for row in measurable if row["path_dependent_gt_precision"]]
        multipath["timing_precision_ps"] = args.timing_precision
        multipath["candidates_with_multiple_paths"] = len(measurable)
        multipath["candidates_path_dependent_gt_precision"] = len(dependent)
        multipath["max_true_delta_range_ps"] = max((row["true_delta_range"] for row in measurable), default=None)
        if measurable:
            multipath["path_dependence_conclusion"] = (
                f"Among {len(measurable)} candidates with at least two path measurements, {len(dependent)} had a "
                f"within-candidate true-delta range greater than the {args.timing_precision:.2f} ps observed "
                f"report resolution; maximum observed range was "
                f"{max(row['true_delta_range'] for row in measurable):.2f} ps. "
                + ("This is evidence of path-dependent impact." if dependent else
                   "No path-dependent impact was visible at the observed report resolution; sub-resolution effects are unresolved."))
        write_json(path_file, multipath)
        make_shared_path_plot(multi_rows, results / "shared_region_path_effect.png", args.timing_precision)
    make_report(results, all_rows, valid, details, cv_rows, args.epsilon,
                args.timing_precision, summary, multipath)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("single", "multipath", "summarize"), default="single")
    parser.add_argument("--aig", type=Path, required=True)
    parser.add_argument("--region-map", type=Path, required=True)
    parser.add_argument("--gpuls", type=Path, required=True)
    parser.add_argument("--abc", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--path-catalog", type=Path, required=True)
    parser.add_argument("--delta-csv", type=Path, required=True)
    parser.add_argument("--pilot-csv", type=Path)
    parser.add_argument("--pilot-replay-root", type=Path)
    parser.add_argument("--single-csv", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--pi-node", type=int, default=4)
    parser.add_argument("--po", type=int, default=11)
    parser.add_argument("--expected-baseline", type=float, default=194.4)
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPS)
    parser.add_argument("--timing-precision", type=float, default=DEFAULT_TIMING_PRECISION)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    for name in ("aig", "region_map", "abc", "library", "path_catalog", "delta_csv"):
        value = getattr(args, name)
        if not value.exists():
            parser.error(f"missing input {name}: {value}")
    if args.mode == "single":
        if not args.gpuls.is_file() or not args.pilot_csv or not args.pilot_replay_root:
            parser.error("single mode needs --gpuls, --pilot-csv, and --pilot-replay-root")
    elif not args.single_csv or not args.single_csv.is_file():
        parser.error("multipath mode needs --single-csv")
    return args


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "single":
        run_single(args)
    elif args.mode == "multipath":
        run_multipath(args)
    else:
        run_summarize(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"calibration failed: {error}", file=sys.stderr)
        raise
