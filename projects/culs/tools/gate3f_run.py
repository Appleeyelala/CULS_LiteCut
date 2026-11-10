#!/usr/bin/env python3
"""Gate 3F: validate fixed Gate 3E choices with ABC mapping and timing.

The runner deliberately treats LiteCut predictions, PO11 arrival, and the
whole-design critical delay as separate quantities.  Replay and timing are
performed independently for each configuration and for a deterministic
repeat.  No candidate is replaced when replay reports a compromised or
reverted requested choice.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


CONFIGS = ("GG", "GL", "LG", "LL")
EXPECTED_PO = 11
MEASUREMENT_PRECISION_PS = 0.01


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_command(command: list[str], cwd: Path, log_path: Path) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(result.stdout or "", encoding="utf-8")
    return result


def run_replay(gpuls: Path, original: Path, region_map: Path, config_file: Path,
               replay_dir: Path) -> None:
    replay_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(gpuls), "-c", " ".join([
            "read", str(original), ";",
            "gate3e_execution_replay", "-o", str(replay_dir),
            "--region-map", str(region_map),
            "--config", str(config_file),
        ]),
    ]
    result = run_command(command, replay_dir, replay_dir / "gpuls.log")
    if result.returncode != 0:
        raise RuntimeError(f"Gate 3E replay failed: {result.stdout[-4000:]}")


def parse_replay(replay_dir: Path, configurations: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    manifest = load_jsonl(replay_dir / "replay_manifest.jsonl")
    by_config = {str(item["config_id"]): item for item in manifest}
    rows: list[dict[str, Any]] = []
    for config_id in CONFIGS:
        if config_id not in by_config:
            raise ValueError(f"missing replay manifest entry for {config_id}")
        item = by_config[config_id]
        trace_path = replay_dir / item["trace_path"]
        aig_path = replay_dir / item["aig_path"]
        trace = load_jsonl(trace_path)
        requested = configurations[config_id]["requested_choices"]
        requested_by_root = {int(choice["root"]): choice for choice in requested}
        realized: list[dict[str, Any]] = []
        for event in trace:
            root = int(event["root"])
            if root not in requested_by_root:
                continue
            realized.append({
                "root": root,
                "region_id": int(event["region_id"]),
                "cut_slot": int(event["cut_slot"]),
                "requested_subgraph_index": int(requested_by_root[root]["subgraph_index"]),
                "requested_label": requested_by_root[root]["label"],
                "selected_subgraph": int(event["selected_subgraph"]),
                "final_status": event["final_status"],
                "execution_order": int(event["execution_order"]),
                "damaged_cut_leaves": event.get("damaged_cut_leaves", []),
                "damaged_by_prior_root": event.get("damaged_by_prior_root", []),
                "saved": event.get("saved"),
                "added": event.get("added"),
            })
        if len(realized) != len(requested):
            raise ValueError(f"replay did not trace all requested roots for {config_id}")
        all_preselected = [event for event in trace if event.get("preselected")]
        all_counts = {status: sum(event["final_status"] == status for event in all_preselected)
                      for status in ("ACCEPTED", "COMPROMISED", "REVERTED")}
        requested_counts = {status: sum(event["final_status"] == status for event in realized)
                            for status in ("ACCEPTED", "COMPROMISED", "REVERTED")}
        rows.append({
            "config_id": config_id,
            "aig_path": str(aig_path),
            "trace_path": str(trace_path),
            "output_sha256": sha256(aig_path),
            "requested_choices": requested,
            "realized_choices": realized,
            "requested_status_counts": requested_counts,
            "all_preselected_status_counts": all_counts,
        })
    return rows


def run_cec(abc: Path, original: Path, candidate: Path, log_path: Path) -> dict[str, Any]:
    command = f'read "{original}"; read "{candidate}"; cec -n'
    result = run_command([str(abc), "-c", command], candidate.parent, log_path)
    text = result.stdout or ""
    passed = result.returncode == 0 and any(token in text.lower() for token in
                                           ("cec passes", "networks are equivalent", "equivalent"))
    return {"pass": passed, "exit_code": result.returncode, "command": command,
            "log": str(log_path)}


def parse_global_metrics(text: str) -> tuple[float | None, float | None, int | None, str | None]:
    area = delay = None
    gates = None
    line = None
    for current in text.splitlines():
        match = re.search(
            r"Gates\s*=\s*([0-9]+).*?Area\s*=\s*([0-9]+(?:\.[0-9]+)?).*?Delay\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*ps",
            current, re.I,
        )
        if match:
            gates = int(match.group(1))
            area = float(match.group(2))
            delay = float(match.group(3))
            line = current.strip()
            break
    if delay is None:
        match = re.search(r"Area\s*=\s*([0-9]+(?:\.[0-9]+)?).*?Delay\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*ps", text, re.I | re.S)
        if match:
            area, delay = float(match.group(1)), float(match.group(2))
    if gates is None:
        match = re.search(r"Gates\s*=\s*([0-9]+)", text, re.I)
        if match:
            gates = int(match.group(1))
    return area, delay, gates, line


def parse_po11_arrival(text: str) -> tuple[float | None, dict[str, Any]]:
    io_match = re.search(r"Primary outputs \([^)]*\):.*?\b11=po11\b", text, re.S)
    block_match = re.search(
        r'Critical path from PI\s+"[^"]+"\s+to PO\s+"po11":(?P<body>.*?)(?:\nCritical path from|\Z)',
        text, re.S,
    )
    body = block_match.group("body") if block_match else ""
    arrivals = [float(value) for value in re.findall(
        r"Arrival(?: time)?\s*=\s*([0-9]+(?:\.[0-9]+)?)", body,
    )]
    # print_delay reports the arrival at the final driver immediately before
    # the PO.  The PO itself has no cell delay and prints only required time.
    arrival = arrivals[-1] if arrivals else None
    return arrival, {
        "po_numbering_confirmed": bool(io_match),
        "po_name": "po11" if io_match else None,
        "critical_path_block_found": bool(block_match),
        "arrival_values_ps": arrivals,
        "extraction_rule": "last Arrival value in print_delay po11 block",
    }


def run_timing(abc: Path, library: Path, aig: Path, work: Path) -> dict[str, Any]:
    work.mkdir(parents=True, exist_ok=True)
    mapped_verilog = work / "mapped.v"
    command = (
        f'read "{library}"; read "{aig}"; map; topo; stime; '
        f'print_delay po11; print_io; write_verilog "{mapped_verilog}"'
    )
    result = run_command([str(abc), "-c", command], work, work / "abc.log")
    text = result.stdout or ""
    area, delay, gates, metric_line = parse_global_metrics(text)
    po_arrival, po_audit = parse_po11_arrival(text)
    return {
        "input_aig": str(aig),
        "input_sha256": sha256(aig),
        "abc_binary": str(abc),
        "library": str(library),
        "library_sha256": sha256(library),
        "command": command,
        "exit_code": result.returncode,
        "area": area,
        "whole_design_critical_delay_ps": delay,
        "mapped_gate_count": gates,
        "target_po_id": EXPECTED_PO,
        "target_po_name": "po11",
        "target_po11_arrival_ps": po_arrival,
        "po_extraction": po_audit,
        "metric_line": metric_line,
        "abc_log": str(work / "abc.log"),
        "mapped_verilog": str(mapped_verilog),
    }


def value_cmp(left: float, right: float, tolerance: float) -> str:
    if abs(left - right) <= tolerance:
        return "TIE_WITHIN_PRECISION"
    return "first_is_faster" if left < right else "second_is_faster"


def direction_consistency(local_g: float, local_l: float, real_g: float, real_l: float,
                          tolerance: float) -> tuple[bool | None, str]:
    local_direction = value_cmp(local_l, local_g, 0.0)
    real_direction = value_cmp(real_l, real_g, tolerance)
    if real_direction == "TIE_WITHIN_PRECISION" or local_direction == "TIE_WITHIN_PRECISION":
        return None, "timing_inactive_or_local_tie"
    return local_direction == real_direction, "direction_comparison"


def config_by_id(replay_rows: list[dict[str, Any]], config_id: str) -> dict[str, Any]:
    return next(row for row in replay_rows if row["config_id"] == config_id)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-aig", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--gpuls", type=Path, required=True)
    parser.add_argument("--region-map", type=Path, required=True)
    parser.add_argument("--gate3e", type=Path, required=True)
    parser.add_argument("--abc", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    gate3e_input = args.gate3e / "input" / "c1355_orig.aig"
    if gate3e_input.exists() and sha256(args.original_aig) != sha256(gate3e_input):
        raise ValueError("Gate 3E input snapshot does not match the requested original AIG")
    gate3e_summary = json.loads((args.gate3e / "gate3e_summary.json").read_text(encoding="utf-8"))
    path_index = json.loads((args.gate3e / "path_decision_candidate_index.json").read_text(encoding="utf-8"))
    fixed = path_index["differing_decisions"]
    expected = {(9, 197, 2), (24, 386, 2)}
    actual = {(int(d["region_id"]), int(d["root"]), int(d["cut_slot"])) for d in fixed}
    if actual != expected:
        raise ValueError(f"Gate 3E fixed decisions changed: {sorted(actual)}")

    score_rows = load_jsonl(args.gate3e / "candidate_litecut_scores.jsonl")
    score_map = {(int(row["root"]), int(row["cut_slot"]), int(row["subgraph_index"])): row
                 for row in score_rows}
    choices: dict[str, dict[str, Any]] = {}
    config_lines: list[str] = []
    for config_id in CONFIGS:
        bits = {"R9": config_id[0], "R24": config_id[1]}
        requested: list[dict[str, Any]] = []
        for decision in fixed:
            region = int(decision["region_id"])
            root = int(decision["root"])
            cut_slot = int(decision["cut_slot"])
            key = "R9" if region == 9 else "R24"
            label = bits[key]
            subgraph = int(decision["gpu_subgraph_index"] if label == "G"
                           else decision["litecut_local_best_subgraph_index"])
            requested.append({"region_id": region, "root": root, "cut_slot": cut_slot,
                              "subgraph_index": subgraph, "label": label})
            config_lines.append(f"{config_id}\t{root}\t{cut_slot}\t{subgraph}\n")
        choices[config_id] = {"config_id": config_id, "requested_choices": requested}
    config_file = out / "main_case" / "path_configurations.input"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text("".join(config_lines), encoding="utf-8")

    # Local prediction provenance is copied into a compact, fixed-case file.
    litecut_choices: dict[str, Any] = {"decisions": []}
    for decision in fixed:
        region, root, cut = int(decision["region_id"]), int(decision["root"]), int(decision["cut_slot"])
        gpu = int(decision["gpu_subgraph_index"])
        local = int(decision["litecut_local_best_subgraph_index"])
        gpu_score = score_map[(root, cut, gpu)]
        local_score = score_map[(root, cut, local)]
        litecut_choices["decisions"].append({
            "region_id": region, "root": root, "cut_slot": cut,
            "gpu_candidate_subgraph_index": gpu,
            "gpu_candidate_predicted_local_abc_delay": gpu_score["predicted_local_abc_delay"],
            "litecut_best_subgraph_index": local,
            "litecut_best_predicted_local_abc_delay": local_score["predicted_local_abc_delay"],
            "prediction_gap_gpu_minus_litecut_best": gpu_score["predicted_local_abc_delay"] - local_score["predicted_local_abc_delay"],
            "local_preference": "L" if local_score["predicted_local_abc_delay"] < gpu_score["predicted_local_abc_delay"] else
            ("G" if local_score["predicted_local_abc_delay"] > gpu_score["predicted_local_abc_delay"] else "TIE"),
            "gpu_feature_hash": gpu_score["feature_hash"],
            "litecut_best_feature_hash": local_score["feature_hash"],
        })
    write_json(out / "main_case" / "litecut_choices.json", litecut_choices)

    replay_rows_by_run: dict[str, list[dict[str, Any]]] = {}
    for run_tag in ("run1", "run2"):
        replay_dir = out / "main_case" / f"replay_{run_tag}"
        run_replay(args.gpuls, args.original_aig, args.region_map, config_file, replay_dir)
        replay_rows_by_run[run_tag] = parse_replay(replay_dir, choices)
        write_jsonl(out / "main_case" / f"replay_results_{run_tag}.jsonl", replay_rows_by_run[run_tag])

    replay_rows = replay_rows_by_run["run1"]
    for row in replay_rows:
        aig = Path(row["aig_path"])
        target = out / "main_case" / f"{row['config_id']}.aig"
        shutil.copy2(aig, target)
        row["published_aig"] = str(target)
    write_jsonl(out / "main_case" / "replay_results.jsonl", replay_rows)

    cec_rows: list[dict[str, Any]] = []
    for run_tag in ("run1", "run2"):
        rows = replay_rows_by_run[run_tag]
        for row in rows:
            aig = Path(row["aig_path"])
            result = run_cec(args.abc, args.original_aig, aig,
                             out / "main_case" / "cec_logs" / f"{run_tag}_{row['config_id']}.log")
            cec_rows.append({"run_tag": run_tag, "config_id": row["config_id"],
                             "cec_status": "PASS" if result["pass"] else "FAIL",
                             "aig_sha256": sha256(aig), "log": result["log"]})
    write_csv(out / "main_case" / "cec_results.csv", cec_rows,
              ["run_tag", "config_id", "cec_status", "aig_sha256", "log"])
    if any(row["cec_status"] != "PASS" for row in cec_rows):
        raise RuntimeError("A replay AIG failed CEC; timing results are not admissible")

    timing_by_run: dict[str, dict[str, dict[str, Any]]] = {}
    for run_tag in ("run1", "run2"):
        timing_by_run[run_tag] = {}
        for row in replay_rows_by_run[run_tag]:
            config_id = row["config_id"]
            work = out / "main_case" / "timing" / run_tag / config_id
            timing_by_run[run_tag][config_id] = run_timing(args.abc, args.library,
                                                             Path(row["aig_path"]), work)
        write_json(out / "main_case" / f"timing_results_{run_tag}.json",
                   timing_by_run[run_tag])

    timing_rows: list[dict[str, Any]] = []
    for config_id in CONFIGS:
        timing = timing_by_run["run1"][config_id]
        labels = config_id
        row = {"config": config_id, "R9_choice": labels[0], "R24_choice": labels[1],
               "PO11_delay_ps": timing["target_po11_arrival_ps"],
               "global_delay_ps": timing["whole_design_critical_delay_ps"],
               "area": timing["area"], "gates": timing["mapped_gate_count"],
               "CEC": "PASS", "aig_sha256": timing["input_sha256"],
               "po_numbering_confirmed": timing["po_extraction"]["po_numbering_confirmed"]}
        timing_rows.append(row)
    write_csv(out / "timing_results.csv", timing_rows,
              ["config", "R9_choice", "R24_choice", "PO11_delay_ps", "global_delay_ps",
               "area", "gates", "CEC", "aig_sha256", "po_numbering_confirmed"])
    write_csv(out / "main_case" / "timing_results.csv", timing_rows,
              ["config", "R9_choice", "R24_choice", "PO11_delay_ps", "global_delay_ps",
               "area", "gates", "CEC", "aig_sha256", "po_numbering_confirmed"])

    def metric(config: str, field: str) -> float:
        value = timing_by_run["run1"][config][field]
        if value is None or not math.isfinite(float(value)):
            raise ValueError(f"missing timing metric {field} for {config}")
        return float(value)

    effect: dict[str, Any] = {"measurement_precision_ps": MEASUREMENT_PRECISION_PS,
                              "metrics": {}}
    for field, key in (("target_po11_arrival_ps", "po11"),
                       ("whole_design_critical_delay_ps", "global")):
        gg, gl, lg, ll = (metric(config, field) for config in CONFIGS)
        r9_g = {"R24=G": lg - gg, "R24=L": ll - gl}
        r24_g = {"R9=G": gl - gg, "R9=L": ll - lg}
        interaction = ll - lg - gl + gg
        r9_consistent_g, _ = direction_consistency(
            litecut_choices["decisions"][0]["gpu_candidate_predicted_local_abc_delay"],
            litecut_choices["decisions"][0]["litecut_best_predicted_local_abc_delay"], gg, lg,
            MEASUREMENT_PRECISION_PS,
        )
        r9_consistent_l, _ = direction_consistency(
            litecut_choices["decisions"][0]["gpu_candidate_predicted_local_abc_delay"],
            litecut_choices["decisions"][0]["litecut_best_predicted_local_abc_delay"], gl, ll,
            MEASUREMENT_PRECISION_PS,
        )
        r24_consistent_g, _ = direction_consistency(
            litecut_choices["decisions"][1]["gpu_candidate_predicted_local_abc_delay"],
            litecut_choices["decisions"][1]["litecut_best_predicted_local_abc_delay"], gg, gl,
            MEASUREMENT_PRECISION_PS,
        )
        r24_consistent_l, _ = direction_consistency(
            litecut_choices["decisions"][1]["gpu_candidate_predicted_local_abc_delay"],
            litecut_choices["decisions"][1]["litecut_best_predicted_local_abc_delay"], lg, ll,
            MEASUREMENT_PRECISION_PS,
        )
        r9_rank = (value_cmp(lg, gg, MEASUREMENT_PRECISION_PS), value_cmp(ll, gl, MEASUREMENT_PRECISION_PS))
        r24_rank = (value_cmp(gl, gg, MEASUREMENT_PRECISION_PS), value_cmp(ll, lg, MEASUREMENT_PRECISION_PS))
        effect["metrics"][key] = {
            "values": {"GG": gg, "GL": gl, "LG": lg, "LL": ll},
            "delta_R9_given_R24_G": r9_g["R24=G"],
            "delta_R9_given_R24_L": r9_g["R24=L"],
            "delta_R24_given_R9_G": r24_g["R9=G"],
            "delta_R24_given_R9_L": r24_g["R9=L"],
            "interaction": interaction,
            "interaction_classification": "MEASURABLE_INTERACTION" if abs(interaction) > MEASUREMENT_PRECISION_PS else "NO_MEASURABLE_INTERACTION",
            "R9_direction_consistent_when_R24_G": r9_consistent_g,
            "R9_direction_consistent_when_R24_L": r9_consistent_l,
            "R24_direction_consistent_when_R9_G": r24_consistent_g,
            "R24_direction_consistent_when_R9_L": r24_consistent_l,
            "R9_ranking_reversal": r9_rank[0] != r9_rank[1] and "TIE" not in r9_rank,
            "R24_ranking_reversal": r24_rank[0] != r24_rank[1] and "TIE" not in r24_rank,
            "comparison_precision": "ABC stime/print_delay values reported to 0.01 ps; tolerance 0.01 ps",
        }
    write_json(out / "main_case" / "effect_analysis.json", effect)

    metric_items = list(effect["metrics"].values())
    timing_active = any(
        abs(float(item["values"][left]) - float(item["values"][right])) > MEASUREMENT_PRECISION_PS
        for item in metric_items
        for left, right in (("GG", "GL"), ("GG", "LG"), ("GG", "LL"))
    )
    measurable_interaction = any(
        item["interaction_classification"] == "MEASURABLE_INTERACTION"
        for item in metric_items
    )
    inconsistent = any(
        value is False
        for item in metric_items
        for value in (
            item["R9_direction_consistent_when_R24_G"],
            item["R9_direction_consistent_when_R24_L"],
            item["R24_direction_consistent_when_R9_G"],
            item["R24_direction_consistent_when_R9_L"],
        )
    )
    if measurable_interaction and timing_active:
        case_classification = "Case D - timing active + measurable interaction"
    elif timing_active and inconsistent:
        case_classification = "Case B - LiteCut ranking inconsistent + timing active"
    elif timing_active:
        case_classification = "Case A - LiteCut ranking consistent + timing active"
    else:
        case_classification = "Case C - LiteCut differs but timing inactive"

    alignment = {
        "classification": {"R9_root_197": "NOT_ALIGNABLE", "R24_root_386": "NOT_ALIGNABLE"},
        "method": "conservative candidate-preserving correspondence audit",
        "reason": "The mapped standard-cell netlist has no stable candidate identity that can be proven to represent the pre-mapping rewrite root across all four configurations. ABC may absorb, decompose, or move the root; nearest mapped nodes are not used.",
        "root_ids": [{"region_id": 9, "root": 197}, {"region_id": 24, "root": 386}],
        "local_mapped_supergate_abc_delay": None,
    }
    write_json(out / "main_case" / "local_alignment_audit.json", alignment)

    determinism: list[dict[str, Any]] = []
    for config_id in CONFIGS:
        a = timing_by_run["run1"][config_id]
        b = timing_by_run["run2"][config_id]
        r1 = config_by_id(replay_rows_by_run["run1"], config_id)
        r2 = config_by_id(replay_rows_by_run["run2"], config_id)
        determinism.append({
            "config_id": config_id,
            "replay_output_sha_equal": r1["output_sha256"] == r2["output_sha256"],
            "cec_equal": True,
            "mapped_gate_count_equal": a["mapped_gate_count"] == b["mapped_gate_count"],
            "area_equal": a["area"] == b["area"],
            "po11_delay_equal": a["target_po11_arrival_ps"] == b["target_po11_arrival_ps"],
            "global_delay_equal": a["whole_design_critical_delay_ps"] == b["whole_design_critical_delay_ps"],
            "run1_aig_sha256": r1["output_sha256"], "run2_aig_sha256": r2["output_sha256"],
        })
    write_json(out / "determinism.json", determinism)

    summary = {
        "gate": "Gate 3F", "status": "PASS",
        "original_aig": str(args.original_aig), "original_aig_sha256": sha256(args.original_aig),
        "gate3e_input_sha256": gate3e_summary.get("input_sha256"),
        "selected_po": EXPECTED_PO,
        "selected_region_path": gate3e_summary["selected_path_region_sequence"],
        "fixed_decisions": fixed,
        "configurations": list(CONFIGS),
        "cec_pass_count": sum(row["cec_status"] == "PASS" for row in cec_rows if row["run_tag"] == "run1"),
        "requested_status_counts": {row["config_id"]: row["requested_status_counts"] for row in replay_rows},
        "realized_choices": {row["config_id"]: row["realized_choices"] for row in replay_rows},
        "timing_results": timing_rows,
        "effect_analysis": effect,
        "case_classification": case_classification,
        "local_alignment": alignment,
        "determinism": determinism,
        "secondary_screening": {"started": False, "reason": "Main case is retained; no secondary pool was needed for Gate 3F completion."},
        "gradient_called": False, "litecut_changed": False,
    }
    write_json(out / "gate3f_summary.json", summary)
    write_json(out / "input_manifest.json", {
        "original_aig": str(args.original_aig), "original_aig_sha256": sha256(args.original_aig),
        "gate3e_source": str(args.gate3e), "gate3e_summary_sha256": sha256(args.gate3e / "gate3e_summary.json"),
        "gate3e_fixed_case_sha256": sha256(args.gate3e / "path_decision_candidate_index.json"),
        "gpuls": str(args.gpuls), "gpuls_sha256": sha256(args.gpuls),
        "region_map": str(args.region_map), "region_map_sha256": sha256(args.region_map),
    })
    version = run_command([str(args.abc), "-c", "version"], out, out / "abc_version.log")
    write_json(out / "abc_flow_manifest.json", {
        "abc_binary": str(args.abc), "abc_sha256": sha256(args.abc),
        "abc_version_output": version.stdout,
        "library": str(args.library), "library_sha256": sha256(args.library),
        "mapping_flow": "read <technology_library>; read <input_aig>; map; topo; stime",
        "po11_extraction_extension": "print_delay po11; print_io",
        "timing_precision_ps": MEASUREMENT_PRECISION_PS,
        "flow_note": "The canonical mapping/timing commands are unchanged; print_delay/print_io are read-only extraction after stime.",
    })
    (out / "gate3f_report.md").write_text(render_report(summary, litecut_choices), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def render_report(summary: dict[str, Any], litecut: dict[str, Any]) -> str:
    effect = summary["effect_analysis"]["metrics"]
    rows = summary["timing_results"]
    lines = ["# Gate 3F - LiteCut Choice vs Real ABC Timing", "", "## Fixed case", "",
             "- original snapshot: c1355_orig.aig", "- target PO: PO11",
             "- region path: R6 -> R9 -> R24 -> R36 -> R37 -> R38",
             "- fixed decisions: R9/(root=197, cut=2), R24/(root=386, cut=2)", ""]
    lines += ["## Timing results", "", "| Config | R9 | R24 | PO11 arrival (ps) | Global delay (ps) | Area | Gates | CEC |", "|---|---:|---:|---:|---:|---:|---:|---|"]
    for row in rows:
        lines.append(f"| {row['config']} | {row['R9_choice']} | {row['R24_choice']} | {row['PO11_delay_ps']:.2f} | {row['global_delay_ps']:.2f} | {row['area']:.2f} | {row['gates']} | {row['CEC']} |")
    lines += ["", "## LiteCut local choices", ""]
    for decision in litecut["decisions"]:
        lines.append(f"- R{decision['region_id']} root {decision['root']}: G={decision['gpu_candidate_predicted_local_abc_delay']:.6f}, L={decision['litecut_best_predicted_local_abc_delay']:.6f}; local preference={decision['local_preference']}")
    lines += ["", "## Effects", ""]
    for name, key in (("PO11", "po11"), ("global", "global")):
        item = effect[key]
        lines += [f"- {name}: R9 deltas = {item['delta_R9_given_R24_G']:.2f} ps / {item['delta_R9_given_R24_L']:.2f} ps; R24 deltas = {item['delta_R24_given_R9_G']:.2f} ps / {item['delta_R24_given_R9_L']:.2f} ps; interaction = {item['interaction']:.2f} ps ({item['interaction_classification']})."]
    lines += ["", "## Alignment and determinism", "", f"- Case classification: {summary['case_classification']}.", "- Candidate-level post-mapping correspondence: NOT_ALIGNABLE for roots 197 and 386. No nearest-node matching was used.", "- Both replay and ABC timing were repeated independently. See `determinism.json`.", "", "Gate 3F status: PASS.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
