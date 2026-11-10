#!/usr/bin/env python3
"""Build existing-data catalogs and run a bounded isolated single-choice pilot."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def predictor_class(cutlib: Path, tools_dir: Path):
    import sys
    sys.path.insert(0, str(tools_dir))
    sys.path.insert(0, str(cutlib))
    from gate3e_run import LiteCutPredictor
    return LiteCutPredictor


def infer_raw19(predictor, raw19: list[float]) -> tuple[float, str]:
    import torch
    arr = np.asarray(raw19, dtype=np.float32).reshape(1, 19)
    scaled = (arr - predictor.x_mu[None, :]) * predictor.x_inv[None, :]
    with torch.inference_mode():
        output = float(predictor.model(torch.from_numpy(scaled)).prediction.detach().cpu().numpy().reshape(-1)[0])
    log_delay = max(-10.0, min(10.0, output / predictor.y_inv + predictor.y_mu))
    feature_hash = hashlib.sha256(np.ascontiguousarray(arr, dtype=np.float64).tobytes()).hexdigest()
    return float(math.expm1(log_delay)), feature_hash


def original_cut_cone(graph: dict[int, list[tuple[int, int]]], root: int,
                      cut_leaves: list[int]) -> dict[int, list[tuple[int, int]]]:
    """Original-AIG fanin cone stopped exactly at this saved decision's leaves."""
    preds: dict[int, list[tuple[int, int]]] = {}
    for src, edges in graph.items():
        for dst, inv in edges:
            preds.setdefault(dst, []).append((src, inv))
    boundary = set(cut_leaves)
    included: set[int] = set()
    stack = [root]
    while stack:
        node = stack.pop()
        if node in included:
            continue
        included.add(node)
        if node == 0 or node in boundary:
            continue
        for pred, _ in preds.get(node, []):
            stack.append(pred)
    subg: dict[int, list[tuple[int, int]]] = {}
    for src in sorted(included):
        subg[src] = [(int(dst), int(inv)) for dst, inv in graph.get(src, []) if dst in included]
    if root not in included:
        raise ValueError(f"root {root} absent from current cut cone")
    return subg


def build_tables(args: argparse.Namespace) -> dict[str, Any]:
    out = args.out_dir
    results = out / "results"
    results.mkdir(parents=True, exist_ok=True)
    gate3a = args.gate3a
    gate3b = args.gate3b
    gate3c = args.gate3c
    gate3e = args.gate3e

    region_payload = json.loads((gate3a / "maximal_standard_mffcs.json").read_text())
    regions = {int(r["region_id"]): r for r in region_payload["regions"]}
    paths = read_jsonl(gate3b / "example_pi_po_paths.jsonl")
    decisions = read_jsonl(gate3c / "executable_rewrite_decisions.jsonl")
    candidate_scores = read_jsonl(gate3e / "candidate_litecut_scores.jsonl")
    if len(paths) != 32 or len(decisions) != 91 or len(candidate_scores) != 326:
        raise ValueError(f"saved artifact count mismatch: paths={len(paths)}, decisions={len(decisions)}, candidates={len(candidate_scores)}")

    score_key = {(int(x["root"]), int(x["cut_slot"]), int(x["subgraph_index"])): x for x in candidate_scores}
    if len(score_key) != len(candidate_scores):
        raise ValueError("duplicate (root,cut_slot,subgraph_index) in Gate3E candidate scores")
    LiteCutPredictor = predictor_class(args.cutlib, args.tools_dir)
    predictor = LiteCutPredictor(args.cutlib, args.aig, args.checkpoint, args.rewrite_library)

    current_rows: list[dict[str, Any]] = []
    current_by_decision: dict[tuple[int, int], dict[str, Any]] = {}
    decision_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    for d in decisions:
        root, cut = int(d["root"]), int(d["cut_slot"])
        key = (root, cut)
        if key in decision_by_key:
            raise ValueError(f"duplicate executable decision {key}")
        decision_by_key[key] = d
        graph = original_cut_cone(predictor.feat_graph, root, [int(x) for x in d["cut_leaves"]])
        raw, names = predictor.extractor.extract_all(graph)
        if len(raw) != 25 or len(raw[:19]) != 19 or not np.isfinite(np.asarray(raw, dtype=np.float64)).all():
            raise ValueError(f"invalid current-cone features for root={root}, cut={cut}")
        prediction, feature_hash = infer_raw19(predictor, raw[:19])
        row = {
            "decision_id": d.get("decision_id", ""), "region_id": int(d.get("decision_owner_region", d["root_region"])),
            "root_id": root, "cut_slot": cut, "cut_leaves": json.dumps(d["cut_leaves"], separators=(",", ":")),
            "current_predicted_local_abc_delay": prediction,
            "current_feature_hash": feature_hash,
            "current_feature_source": "original_AIG_cut_bounded_cone_for_same_root_cut",
            "feature_names_match": bool(names[:19] == predictor.feature_names[:19]),
        }
        current_rows.append(row)
        current_by_decision[key] = row
    write_csv(results / "current_cut_cone_scores.csv", current_rows, list(current_rows[0]))
    with (results / "current_cut_cone_scores.jsonl").open("w", encoding="utf-8") as stream:
        for row in current_rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")

    delta_rows: list[dict[str, Any]] = []
    for d in decisions:
        root, cut = int(d["root"]), int(d["cut_slot"])
        current = current_by_decision[(root, cut)]
        expected_subgraphs = [int(x) for x in d["candidate_subgraph_indices"]]
        for subgraph in expected_subgraphs:
            scored = score_key.get((root, cut, subgraph))
            if scored is None:
                raise ValueError(f"missing Gate3E score root={root} cut={cut} subgraph={subgraph}")
            candidate_delay = float(scored["predicted_local_abc_delay"])
            delta = candidate_delay - float(current["current_predicted_local_abc_delay"])
            delta_rows.append({
                "region_id": int(d.get("decision_owner_region", d["root_region"])),
                "root_id": root, "cut_slot": cut,
                "candidate_id": f"root{root}_cut{cut}_subgraph{subgraph}",
                "subgraph_index": subgraph, "candidate_type": scored["candidate_type"],
                "candidate_count_for_decision": len(expected_subgraphs),
                "current_pred_delay": current["current_predicted_local_abc_delay"],
                "current_feature_hash": current["current_feature_hash"],
                "candidate_pred_delay": candidate_delay,
                "candidate_feature_hash": scored["feature_hash"],
                "delta_litecut": delta,
                "delta_semantics": "candidate_predicted_local_abc_delay_minus_current_AIG_cut_cone_proxy",
            })
    if len(delta_rows) != 326:
        raise ValueError(f"expected 326 delta records, got {len(delta_rows)}")
    write_csv(results / "litecut_candidate_delta.csv", delta_rows, list(delta_rows[0]))

    path_rows: list[dict[str, Any]] = []
    for p in paths:
        region_seq = [int(x) for x in p["compressed_region_sequence"]]
        root_seq = [int(regions[r]["root"]) for r in region_seq]
        path_id = f"po{int(p['po_id']):02d}_pi_node{int(p['pi']):03d}"
        path_rows.append({
            "path_id": path_id, "pi": int(p["pi"]), "po": int(p["po_id"]),
            "node_count": len(p["node_path"]), "region_count": len(region_seq),
            "node_sequence": json.dumps(p["node_path"], separators=(",", ":")),
            "region_sequence": json.dumps(region_seq, separators=(",", ":")),
            "root_sequence": json.dumps(root_seq, separators=(",", ":")),
            "baseline_delay": "", "baseline_delay_semantics": "pending_ABC_baseline_PI_match_audit",
            "baseline_po_endpoint_max_arrival_ps": "", "baseline_critical_pi": "",
            "saved_pi_matches_baseline_critical_pi": "",
            "exact_original_node_path_delay_available": "false",
        })
    write_csv(results / "path_catalog.csv", path_rows, list(path_rows[0]))

    return {
        "path_count": len(paths), "decision_count": len(decisions), "candidate_count": len(delta_rows),
        "current_proxy_count": len(current_rows), "input_sha256": sha256(args.aig),
        "checkpoint_sha256": sha256(args.checkpoint), "saved_path_policy": "use saved Gate3B paths only; no path enumeration",
    }


def run_process(command: list[str], log_path: Path) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(proc.stdout, encoding="utf-8")
    return proc


def abc_command(abc: Path, script: str, log_path: Path) -> str:
    proc = run_process([str(abc), "-c", script], log_path)
    if proc.returncode != 0:
        raise RuntimeError(f"ABC exited {proc.returncode}; see {log_path}")
    return proc.stdout


def parse_delay_reports(text: str) -> dict[int, dict[str, Any]]:
    reports: dict[int, dict[str, Any]] = {}
    blocks = re.split(r"(?=Critical path from PI \"pi\d+\" to PO \"po\d+\":)", text)
    for block in blocks:
        m = re.search(r'Critical path from PI "pi(\d+)" to PO "po(\d+)":', block)
        if not m:
            continue
        start_pi, po = int(m.group(1)), int(m.group(2))
        arrivals = [float(x) for x in re.findall(r"Arrival(?: time)?\s*=\s*([0-9]+(?:\.[0-9]+)?)", block)]
        if not arrivals:
            continue
        reports[po] = {"critical_pi_index": start_pi, "po_endpoint_max_arrival_ps": arrivals[-1]}
    return reports


def parse_delay_report_list(text: str) -> list[dict[str, Any]]:
    """Parse every `print_delay` report, including separate pair and PO queries."""
    reports: list[dict[str, Any]] = []
    blocks = re.split(r"(?=Critical path from PI \"pi\d+\" to PO \"po\d+\":)", text)
    for block in blocks:
        m = re.search(r'Critical path from PI "pi(\d+)" to PO "po(\d+)":', block)
        if not m:
            continue
        arrivals = [float(x) for x in re.findall(r"Arrival(?: time)?\s*=\s*([0-9]+(?:\.[0-9]+)?)", block)]
        if arrivals:
            reports.append({
                "critical_pi_index": int(m.group(1)), "po_id": int(m.group(2)),
                "arrival_ps": arrivals[-1],
            })
    return reports


def parse_global_delay(text: str) -> float | None:
    matches = re.findall(r"Delay\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*ps", text, re.I)
    return float(matches[-1]) if matches else None


def run_experiment(args: argparse.Namespace, static_summary: dict[str, Any]) -> dict[str, Any]:
    out = args.out_dir
    results = out / "results"
    logs = out / "logs"
    data = out / "work"
    for directory in (logs, data / "replay", data / "cec", data / "abc"):
        directory.mkdir(parents=True, exist_ok=True)

    paths = read_jsonl(args.gate3b / "example_pi_po_paths.jsonl")
    decisions = read_jsonl(args.gate3c / "executable_rewrite_decisions.jsonl")
    delta_rows = list(csv.DictReader((results / "litecut_candidate_delta.csv").open(encoding="utf-8")))
    decisions_by_root: dict[int, list[dict[str, Any]]] = {}
    for d in decisions:
        decisions_by_root.setdefault(int(d["root"]), []).append(d)
    candidates_by_decision: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in delta_rows:
        candidates_by_decision.setdefault((int(row["root_id"]), int(row["cut_slot"])), []).append(row)

    # Record the ABC command/help capabilities and a frozen baseline timing sweep.
    help_script = "help print_delay; help stime; help set_input_delay; help set_input_arrival; help set_delay"
    help_out = abc_command(args.abc, help_script, logs / "abc_timing_command_help.log")
    pair_terms = "; ".join(
        f"print_delay po{int(p['po_id']):02d} pi{int(p['pi']) - 1:02d}" for p in paths
    )
    baseline_script = f'read "{args.library}"; read "{args.aig}"; map; topo; stime; {pair_terms}; print_io'
    baseline_text = abc_command(args.abc, baseline_script, logs / "abc_baseline_endpoint_pairs.log")
    baseline_pair_list = parse_delay_report_list(baseline_text)
    baseline_pair_reports = {int(r["po_id"]): r for r in baseline_pair_list}
    missing_pair_reports = sorted({int(p["po_id"]) for p in paths} - set(baseline_pair_reports))
    po_terms = "; ".join(f"print_delay po{int(p['po_id']):02d}" for p in paths)
    baseline_po_script = f'read "{args.library}"; read "{args.aig}"; map; topo; stime; {po_terms}; print_io'
    baseline_po_text = abc_command(args.abc, baseline_po_script, logs / "abc_baseline_po_maxima.log")
    baseline_po_list = parse_delay_report_list(baseline_po_text)
    baseline_reports = {int(r["po_id"]): r for r in baseline_po_list}
    if len(baseline_reports) != 32:
        raise RuntimeError(f"expected PO-max reports for 32 POs, got {len(baseline_reports)}")
    global_baseline = parse_global_delay(baseline_text)

    path_by_po: dict[int, dict[str, Any]] = {int(p["po_id"]): p for p in paths}
    catalog_rows = list(csv.DictReader((results / "path_catalog.csv").open(encoding="utf-8")))
    for row in catalog_rows:
        po = int(row["po"])
        pair = baseline_pair_reports.get(po)
        po_max = baseline_reports[po]
        expected_pi = int(row["pi"]) - 1
        pair_ok = pair is not None and int(pair["critical_pi_index"]) == expected_pi
        row["baseline_po_endpoint_max_arrival_ps"] = po_max["arrival_ps"]
        row["baseline_critical_pi"] = po_max["critical_pi_index"]
        row["saved_pi_matches_baseline_critical_pi"] = str(expected_pi == int(po_max["critical_pi_index"])).lower()
        row["baseline_delay"] = pair["arrival_ps"] if pair_ok else ""
        row["baseline_delay_semantics"] = (
            "endpoint_pair_delay" if pair_ok else
            "pair_query_source_mismatch" if pair is not None else
            "no_mapped_combinational_path_for_saved_PI_PO_pair"
        )
    fields = list(catalog_rows[0])
    write_csv(results / "path_catalog.csv", catalog_rows, fields)

    # `print_delay <CO> <CI>` asks ABC for the maximum mapped path for this
    # exact PI/PO pair; it does not require that pair to be globally critical.
    path_decisions: list[tuple[tuple[int, int, int], dict[str, Any], list[dict[str, Any]]]] = []
    for p in paths:
        po = int(p["po_id"])
        pair_report = baseline_pair_reports.get(po)
        if pair_report is None:
            continue
        if int(p["pi"]) - 1 != int(pair_report["critical_pi_index"]):
            continue
        rows: list[dict[str, Any]] = []
        seen: set[tuple[int, int]] = set()
        for node in p["node_path"]:
            for d in decisions_by_root.get(int(node), []):
                key = (int(d["root"]), int(d["cut_slot"]))
                if key not in seen and candidates_by_decision.get(key):
                    seen.add(key)
                    rows.append(d)
        rank = (-len(rows), -len(p["compressed_region_sequence"]), po)
        path_decisions.append((rank, p, rows))
    path_decisions.sort(key=lambda x: x[0])

    cap = int(args.max_candidates)
    selected: list[dict[str, Any]] = []
    selected_keys: set[tuple[int, int]] = set()
    path_selection: dict[tuple[int, int], dict[str, Any]] = {}
    for _, p, rows in path_decisions:
        for d in rows:
            key = (int(d["root"]), int(d["cut_slot"]))
            if key in selected_keys:
                continue
            choices = candidates_by_decision[key]
            chosen = min(choices, key=lambda x: (float(x["delta_litecut"]), int(x["subgraph_index"])))
            selected.append({
                "path_id": f"po{int(p['po_id']):02d}_pi_node{int(p['pi']):03d}",
                "pi_node": int(p["pi"]), "po_id": int(p["po_id"]),
                "region_id": int(chosen["region_id"]), "root": key[0], "cut_slot": key[1],
                "subgraph_index": int(chosen["subgraph_index"]),
                "candidate_id": chosen["candidate_id"], "current_pred_delay": float(chosen["current_pred_delay"]),
                "candidate_pred_delay": float(chosen["candidate_pred_delay"]), "predicted_delta": float(chosen["delta_litecut"]),
            })
            selected_keys.add(key)
            path_selection[key] = p
            if len(selected) >= cap:
                break
        if len(selected) >= cap:
            break

    timing_caps = {
        "path_count": len(paths), "paths_with_valid_endpoint_pair_report": len(path_decisions),
        "saved_pairs_without_STA_path": missing_pair_reports,
        "selected_candidate_count": len(selected), "configured_cap": cap,
        "min_target": int(args.min_candidates),
        "selection_rule": "one minimum-delta candidate per executable root/cut, over saved Gate3B paths with an explicit ABC print_delay PO,PI endpoint-pair report, ranked by decision count desc, region count desc, PO asc",
        "no_matched_path_reason": "ABC did not return a fixed PI/PO path report for any saved Gate3B path." if not path_decisions else None,
    }
    (results / "baseline_timing_summary.json").write_text(json.dumps({
        "abc_binary": str(args.abc), "library": str(args.library), "endpoint_pair_script": baseline_script,
        "po_max_script": baseline_po_script,
        "global_critical_delay_ps": global_baseline,
        "endpoint_pair_by_po": baseline_pair_reports, "po_endpoint_max_by_po": baseline_reports,
        "timing_precision_ps": args.timing_precision, "timing_command_help": help_out,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (results / "pilot_selection.json").write_text(json.dumps({"policy": timing_caps, "selected": selected}, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Isolated replay configurations; each line names exactly one choice.
    config_path = data / "single_choices.input"
    config_path.write_text("".join(f"S{i:03d} {x['root']} {x['cut_slot']} {x['subgraph_index']}\n" for i, x in enumerate(selected)), encoding="utf-8")
    replay_root = data / "replay"
    # CULS' internal command tokenizer treats double quotes as part of the
    # filename (unlike ABC); staged paths contain no spaces, so use bare paths.
    replay_script = f'read {args.aig}; gate3e_execution_replay -o {replay_root} --region-map {args.region_map} --config {config_path}'
    if selected:
        replay_text = run_process([str(args.gpuls), "-c", replay_script], logs / "gpuls_single_replay.log")
        if replay_text.returncode != 0:
            raise RuntimeError(f"GPULS replay exited {replay_text.returncode}; see {logs / 'gpuls_single_replay.log'}")
    manifest_path = replay_root / "replay_manifest.jsonl"
    manifests = read_jsonl(manifest_path) if manifest_path.exists() else []
    manifest_by_id = {m["config_id"]: m for m in manifests}
    pilot_rows: list[dict[str, Any]] = []
    for i, choice in enumerate(selected):
        config_id = f"S{i:03d}"
        if config_id not in manifest_by_id:
            raise RuntimeError(f"missing replay manifest for {config_id}")
        manifest = manifest_by_id[config_id]
        aig = replay_root / manifest["aig_path"]
        trace_path = replay_root / manifest["trace_path"]
        trace = read_jsonl(trace_path)
        matching = [t for t in trace if int(t.get("root", -1)) == choice["root"] and int(t.get("cut_slot", -1)) == choice["cut_slot"]]
        if len(matching) != 1:
            raise RuntimeError(f"expected one requested trace entry for {config_id}, got {len(matching)}")
        decision_trace = matching[0]
        if not decision_trace.get("preselected"):
            raise RuntimeError(f"requested choice was not preselected in {config_id}: {decision_trace}")
        cec_script = f'read "{args.aig}"; read "{aig}"; cec -n'
        cec_text = abc_command(args.abc, cec_script, logs / f"cec_{config_id}.log")
        cec_lower = cec_text.lower()
        cec_pass = ("equivalent" in cec_lower or "cec passes" in cec_lower) and "not equivalent" not in cec_lower
        after_delay: float | None = None
        after_pi: int | None = None
        global_after: float | None = None
        timing_script = (f'read "{args.library}"; read "{aig}"; map; topo; stime; '
                         f'print_delay po{choice["po_id"]:02d} pi{choice["pi_node"] - 1:02d}; '
                         f'print_delay po{choice["po_id"]:02d}; print_io')
        if cec_pass:
            timing_text = abc_command(args.abc, timing_script, logs / f"abc_{config_id}.log")
            report_list = parse_delay_report_list(timing_text)
            pair_reports = [r for r in report_list if int(r["po_id"]) == choice["po_id"] and int(r["critical_pi_index"]) == choice["pi_node"] - 1]
            po_reports = [r for r in report_list if int(r["po_id"]) == choice["po_id"]]
            po_max_reports = [r for r in po_reports if int(r["critical_pi_index"]) != choice["pi_node"] - 1]
            if not pair_reports:
                raise RuntimeError(f"missing fixed PI/PO timing block in {config_id}")
            after_delay = float(pair_reports[0]["arrival_ps"])
            after_pi = int(pair_reports[0]["critical_pi_index"])
            after_po_max = float((po_max_reports or po_reports)[-1]["arrival_ps"])
            after_po_max_pi = int((po_max_reports or po_reports)[-1]["critical_pi_index"])
            global_after = parse_global_delay(timing_text)
        else:
            after_po_max = None
            after_po_max_pi = None
        base_report = baseline_pair_reports[choice["po_id"]]
        po_base_report = baseline_reports[choice["po_id"]]
        pair_match_base = int(base_report["critical_pi_index"]) == int(choice["pi_node"]) - 1
        baseline_delay = float(base_report["arrival_ps"])
        true_delta_pair = (after_delay - baseline_delay) if after_delay is not None and pair_match_base else None
        true_delta_endpoint = (after_po_max - float(po_base_report["arrival_ps"])) if after_po_max is not None else None
        pilot_rows.append({
            **choice,
            "baseline_delay": baseline_delay if pair_match_base else "",
            "baseline_delay_semantics": "endpoint_pair_delay",
            "baseline_critical_pi_index": po_base_report["critical_pi_index"],
            "sta_after_delay": after_delay if after_delay is not None else "",
            "sta_after_critical_pi_index": after_po_max_pi if after_po_max_pi is not None else "",
            "sta_after_po_endpoint_max_arrival_ps": after_po_max if after_po_max is not None else "",
            "true_delta_endpoint_po": true_delta_endpoint if true_delta_endpoint is not None else "",
            "true_delta_endpoint_pair": true_delta_pair if true_delta_pair is not None else "",
            "predicted_delta": choice["predicted_delta"],
            "error_endpoint_pair": choice["predicted_delta"] - true_delta_pair if true_delta_pair is not None else "",
            "abs_error_endpoint_pair": abs(choice["predicted_delta"] - true_delta_pair) if true_delta_pair is not None else "",
            "endpoint_pair_status": "VALID_PAIR_MEASURE" if pair_match_base and after_delay is not None else "PAIR_QUERY_NO_PATH",
            "final_status": decision_trace.get("final_status", ""),
            "preselected": decision_trace.get("preselected", False),
            "cec": "PASS" if cec_pass else "FAIL",
            "aig_sha256": sha256(aig), "trace_path": str(trace_path),
            "cec_command": cec_script, "abc_command": timing_script if cec_pass else "",
            "global_delay_after_ps": global_after if global_after is not None else "",
            "replay_semantics": "single requested decision from original baseline; all other roots unrequested",
        })

    if pilot_rows:
        write_csv(results / "single_candidate_validation.csv", pilot_rows, list(pilot_rows[0]))
    else:
        write_csv(results / "single_candidate_validation.csv", [], ["candidate_id", "region_id", "root", "cut_slot", "subgraph_index", "endpoint_pair_status", "final_status", "cec"])

    valid = [r for r in pilot_rows if r["endpoint_pair_status"] == "VALID_PAIR_MEASURE" and r["cec"] == "PASS"]
    summary = {
        **static_summary, "baseline_global_critical_delay_ps": global_baseline,
        "paths_with_valid_endpoint_pair_report": len(path_decisions), "pilot_count": len(pilot_rows),
        "pilot_minimum_target_met": len(pilot_rows) >= int(args.min_candidates),
        "cec_pass_count": sum(r["cec"] == "PASS" for r in pilot_rows),
        "valid_endpoint_pair_count": len(valid), "po_endpoint_only_or_unresolved_count": len(pilot_rows) - len(valid),
        "candidate_final_status_counts": {status: sum(r["final_status"] == status for r in pilot_rows) for status in sorted({r["final_status"] for r in pilot_rows})},
        "timing_precision_ps": args.timing_precision,
        "path_identity_note": "Exact mapped correspondence to the original AIG node sequence is unavailable. ABC's explicit print_delay <PO> <PI> reports the maximum mapped timing path between the same PI/PO endpoints; the PO-wide maximum is also recorded separately.",
        "pilot_selection": timing_caps,
    }
    (results / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Step 5 single-candidate pilot",
        "",
        f"- Saved paths reused: {summary['path_count']} (no path enumeration).",
        f"- Saved paths with an explicit ABC PI/PO pair report: {summary['paths_with_valid_endpoint_pair_report']}",
        f"- Isolated candidate replays: {summary['pilot_count']} (cap {cap}; minimum target {args.min_candidates}).",
        f"- CEC PASS: {summary['cec_pass_count']}/{summary['pilot_count']}; valid same PI/PO endpoint-pair measures: {summary['valid_endpoint_pair_count']}/{summary['pilot_count']}.",
        f"- Baseline global critical delay: {global_baseline} ps.",
        "- LiteCut delta is candidate local-score minus the same decision's original-AIG cut-cone proxy; it is not treated as an MFFC timing contribution.",
        "- Endpoint-pair true delta comes from ABC's explicit `print_delay <PO> <PI>` query. PO-wide maximum deltas are a distinct field and may have a different critical PI.",
        "",
        "| Path | Region | Decision | Candidate | LiteCut delta | PO | pair status | realized status | CEC | PI/PO true delta (ps) |",
        "|---|---:|---|---|---:|---:|---|---|---|---:|",
    ]
    for r in pilot_rows:
        val = r["true_delta_endpoint_pair"]
        lines.append(f"| {r['path_id']} | R{r['region_id']} | ({r['root']},{r['cut_slot']}) | {r['subgraph_index']} | {float(r['predicted_delta']):.6g} | {r['po_id']} | {r['endpoint_pair_status']} | {r['final_status']} | {r['cec']} | {val} |")
    if not pilot_rows:
        lines.append("| — | — | — | — | — | — | no baseline-matched path | not run | — | — |")
    (results / "step5_pilot_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest = {
        "design": "c1355_orig", "input_aig": str(args.aig), "input_sha256": sha256(args.aig),
        "gpuls": str(args.gpuls), "abc": str(args.abc), "library": str(args.library),
        "library_sha256": sha256(args.library), "gate3e_replay_command": replay_script,
        "baseline_abc_command": baseline_script, "timing_precision_ps": args.timing_precision,
        "selected_candidate_count": len(selected), "worktree_core_modified": False,
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--aig", type=Path, required=True)
    p.add_argument("--gate3a", type=Path, required=True)
    p.add_argument("--gate3b", type=Path, required=True)
    p.add_argument("--gate3c", type=Path, required=True)
    p.add_argument("--gate3e", type=Path, required=True)
    p.add_argument("--cutlib", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--rewrite-library", type=Path, required=True)
    p.add_argument("--region-map", type=Path, required=True)
    p.add_argument("--gpuls", type=Path, required=True)
    p.add_argument("--abc", type=Path, required=True)
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--tools-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--max-candidates", type=int, default=15)
    p.add_argument("--min-candidates", type=int, default=10)
    p.add_argument("--timing-precision", type=float, default=0.01)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if (args.out_dir / "run_manifest.json").exists():
        raise SystemExit(f"Refusing to overwrite existing experiment output: {args.out_dir}")
    static = build_tables(args)
    summary = run_experiment(args, static)
    print(json.dumps(summary, indent=2, sort_keys=True))
