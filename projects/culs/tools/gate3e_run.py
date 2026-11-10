#!/usr/bin/env python3
"""Gate 3E: LiteCut candidate scoring and execution-aware path replay.

This driver reuses the Gate 2A mapping-free materialization and canonical
Student inference helpers.  It deliberately keeps the GPU rewrite decision
and CPU install semantics in the compiled Gate 3E replay command.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_mapping_free_bridge(cutlib: Path):
    sys.path.insert(0, str(cutlib))
    from gate2a_candidate_inference import (  # type: ignore
        FEATURE_NAMES_25,
        CompleteCutFeatureExtractor,
        materialize_candidate_subgraph,
        parse_aiger_ascii_binary,
        parse_library,
    )
    from validate.eval_prediction_accuracy import load_student_model  # type: ignore
    return {
        "FEATURE_NAMES_25": FEATURE_NAMES_25,
        "CompleteCutFeatureExtractor": CompleteCutFeatureExtractor,
        "materialize_candidate_subgraph": materialize_candidate_subgraph,
        "parse_aiger_ascii_binary": parse_aiger_ascii_binary,
        "parse_library": parse_library,
        "load_student_model": load_student_model,
    }


class LiteCutPredictor:
    def __init__(self, cutlib: Path, original_aig: Path, checkpoint: Path, library: Path):
        bridge = load_mapping_free_bridge(cutlib)
        self.feature_names = bridge["FEATURE_NAMES_25"]
        self.materialize = bridge["materialize_candidate_subgraph"]
        parse_aiger = bridge["parse_aiger_ascii_binary"]
        self.feat_graph, self.max_var, self.n_pi, self.n_po, self.n_and = parse_aiger(original_aig)
        self.lib = bridge["parse_library"](library)
        self.extractor = bridge["CompleteCutFeatureExtractor"](self.feat_graph, inverter_weight=1.5)
        self.model, self.ckpt = bridge["load_student_model"](str(checkpoint), "cpu")
        payload = self.ckpt.get("abc_inference_payload", {})
        cfg = self.ckpt.get("config", {})
        input_dim = int(payload.get("input_dim", cfg.get("input_dim", 19)))
        group_dims = tuple(int(x) for x in payload.get("group_dims", [8, 6, 5]))
        if input_dim != 19 or group_dims != (8, 6, 5):
            raise ValueError(f"unexpected LiteCut checkpoint schema: {input_dim}, {group_dims}")
        self.x_mu = np.asarray(self.ckpt.get("x_mu", payload.get("input_mu")), dtype=np.float32).reshape(-1)
        self.x_inv = np.asarray(self.ckpt.get("x_inv_sigma", payload.get("input_inv_sigma")), dtype=np.float32).reshape(-1)
        self.y_mu = float(np.asarray(self.ckpt.get("y_mu", payload.get("output_mu"))).reshape(-1)[0])
        self.y_inv = float(np.asarray(self.ckpt.get("y_inv_sigma", payload.get("output_inv_sigma"))).reshape(-1)[0])
        if self.x_mu.size != 19 or self.x_inv.size != 19:
            raise ValueError("LiteCut checkpoint normalization is not 19-dimensional")
        self.model.eval()

    def score(self, record: dict[str, Any], subgraph_index: int) -> dict[str, Any]:
        truth = int(record["truth"])
        materialized = dict(record)
        materialized["target_root"] = int(record["root"])
        materialized["subgraph_index"] = int(subgraph_index)
        materialized["phase"] = int(self.lib["p_phases"][truth])
        materialized["permutation"] = [
            int(x) for x in self.lib["perms4"][self.lib["p_perms"][truth]]
        ]
        subgraph, dag_info = self.materialize(materialized, self.lib)
        full, names = self.extractor.extract_all(subgraph)
        if names != self.feature_names:
            raise ValueError(f"canonical feature order mismatch: {names}")
        raw = np.asarray(full, dtype=np.float64)
        if raw.size != 25 or not np.all(np.isfinite(raw)):
            raise ValueError(f"non-finite LiteCut features for {record['root']}:{record['cut_slot']}:{subgraph_index}")
        raw19 = raw[:19].astype(np.float32).reshape(1, 19)
        scaled = (raw19 - self.x_mu[None, :]) * self.x_inv[None, :]
        with torch.inference_mode():
            model_output = float(self.model(torch.from_numpy(scaled)).prediction.detach().cpu().numpy().reshape(-1)[0])
        y_log = max(-10.0, min(10.0, model_output / self.y_inv + self.y_mu))
        prediction = float(math.expm1(y_log))
        feature_hash = hashlib.sha256(np.ascontiguousarray(raw19, dtype=np.float64).tobytes()).hexdigest()
        return {
            "subgraph_index": int(subgraph_index),
            "candidate_type": "GPU_REWRITE_LIBRARY_SUBGRAPH",
            "feature_hash": feature_hash,
            "continuous_features_19": raw[:19].tolist(),
            "predicted_local_abc_delay": prediction,
            "replacement_dag_info": dag_info,
        }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def selected_gpu_choices(trace_path: Path) -> dict[int, dict[str, int]]:
    choices: dict[int, dict[str, int]] = {}
    for item in load_jsonl(trace_path):
        if item.get("preselected"):
            choices[int(item["root"])] = {
                "cut_slot": int(item["cut_slot"]),
                "subgraph_index": int(item["selected_subgraph"]),
            }
    return choices


def select_path(paths: list[dict[str, Any]], decisions: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_root: dict[int, list[dict[str, Any]]] = {}
    for decision in decisions:
        by_root.setdefault(int(decision["root"]), []).append(decision)
    ranked: list[tuple[int, int, int, dict[str, Any], list[dict[str, Any]]]] = []
    for path in paths:
        path_decisions: list[dict[str, Any]] = []
        for node in path["node_path"]:
            path_decisions.extend(by_root.get(int(node), []))
        ranked.append((
            -len(path_decisions),
            -len(path["compressed_region_sequence"]),
            int(path["po_id"]),
            path,
            path_decisions,
        ))
    _, _, _, selected, selected_decisions = min(ranked, key=lambda x: x[:3])
    return selected, selected_decisions


def parse_replay_trace(trace_path: Path) -> list[dict[str, Any]]:
    return load_jsonl(trace_path)


def run_replay(gpuls: Path, original_aig: Path, region_map: Path, config_file: Path,
               replay_dir: Path) -> None:
    cmd = [
        str(gpuls), "-c",
        " ".join([
            "read", str(original_aig), ";",
            "gate3e_execution_replay", "-o", str(replay_dir),
            "--region-map", str(region_map),
            "--config", str(config_file),
        ]),
    ]
    result = subprocess.run(cmd, text=True, capture_output=True)
    (replay_dir / "gpuls_stdout.log").write_text(result.stdout, encoding="utf-8")
    (replay_dir / "gpuls_stderr.log").write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"Gate 3E replay failed with {result.returncode}: {result.stderr[-4000:]}")


def run_cec(abc: Path, original_aig: Path, candidate_aig: Path, log_path: Path) -> bool:
    script = f"read {original_aig}; read {candidate_aig}; cec -n"
    result = subprocess.run([str(abc), "-c", script], text=True, capture_output=True)
    log_path.write_text(result.stdout + "\n--- stderr ---\n" + result.stderr, encoding="utf-8")
    text = (result.stdout + "\n" + result.stderr).lower()
    return result.returncode == 0 and ("equivalent" in text or "cec passes" in text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutlib", type=Path, default=Path("/home/wushangge/cutLibOrigin"))
    ap.add_argument("--original-aig", type=Path, default=Path("/home/wushangge/cutLibOrigin/open-sourceDataset/OpenABC/c1355_orig.aig"))
    ap.add_argument("--checkpoint", type=Path, default=Path("/home/wushangge/cutLibOrigin/best.pt"))
    ap.add_argument("--library", type=Path, default=Path("/home/wushangge/CULS/src/algorithms/rewrite_library.inc"))
    ap.add_argument("--gate3c", type=Path, default=Path("gate3c_out/c1355_orig"))
    ap.add_argument("--gate3b", type=Path, default=Path("gate3b_out/c1355_orig"))
    ap.add_argument("--gate3d", type=Path, default=Path("gate3d_out/c1355_orig"))
    ap.add_argument("--region-map", type=Path, required=True)
    ap.add_argument("--gpuls", type=Path, required=True)
    ap.add_argument("--abc", type=Path, default=Path("/home/share/process/abc/abc"))
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    decisions = sorted(
        load_jsonl(args.gate3c / "executable_rewrite_decisions.jsonl"),
        key=lambda d: (int(d["root"]), int(d["cut_slot"])),
    )
    if len(decisions) != 91:
        raise ValueError(f"expected 91 executable decisions, got {len(decisions)}")
    total_candidates = sum(int(d["candidate_count"]) for d in decisions)
    if total_candidates != 326:
        raise ValueError(f"expected 326 candidates, got {total_candidates}")

    predictor = LiteCutPredictor(args.cutlib, args.original_aig, args.checkpoint, args.library)
    gpu_choices = selected_gpu_choices(args.gate3d / "decision_execution_trace.jsonl")
    candidate_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    score_map: dict[tuple[int, int], dict[int, dict[str, Any]]] = {}
    for decision in decisions:
        root, cut_slot = int(decision["root"]), int(decision["cut_slot"])
        scores: dict[int, dict[str, Any]] = {}
        for subgraph in decision["candidate_subgraph_indices"]:
            score = predictor.score(decision, int(subgraph))
            scores[int(subgraph)] = score
            candidate_rows.append({
                "region_id": int(decision["decision_owner_region"]),
                "root": root,
                "cut_slot": cut_slot,
                **score,
            })
        score_map[(root, cut_slot)] = scores
        ordered = sorted(scores.items(), key=lambda item: (item[1]["predicted_local_abc_delay"], item[0]))
        local_best = ordered[0][0]
        values = np.asarray([x["predicted_local_abc_delay"] for x in scores.values()], dtype=np.float64)
        gpu = gpu_choices.get(root)
        gpu_sub = None
        gpu_matches_decision = bool(gpu and gpu["cut_slot"] == cut_slot)
        if gpu_matches_decision:
            gpu_sub = gpu["subgraph_index"]
        decision_rows.append({
            "region_id": int(decision["decision_owner_region"]),
            "root": root,
            "cut_slot": cut_slot,
            "candidate_count": len(scores),
            "prediction_min": float(values.min()),
            "prediction_max": float(values.max()),
            "prediction_range": float(values.max() - values.min()),
            "prediction_std": float(values.std()),
            "litecut_local_best_subgraph": local_best,
            "original_gpu_selected_subgraph": "" if gpu_sub is None else gpu_sub,
            "gpu_choice_defined_for_decision": gpu_matches_decision,
            "gpu_choice_equals_litecut_local_best": "" if gpu_sub is None else str(gpu_sub == local_best).lower(),
        })
    write_csv(out / "decision_litecut_stats.csv", decision_rows, list(decision_rows[0]))
    with (out / "candidate_litecut_scores.jsonl").open("w", encoding="utf-8") as stream:
        for row in candidate_rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")

    paths = load_jsonl(args.gate3b / "example_pi_po_paths.jsonl")
    selected_path, path_decisions = select_path(paths, decisions)
    path_decisions_info = []
    for decision in path_decisions:
        key = (int(decision["root"]), int(decision["cut_slot"]))
        stats = next(x for x in decision_rows if (x["root"], x["cut_slot"]) == key)
        path_decisions_info.append({**decision, "litecut": stats, "candidates": score_map[key]})
    selected_path_payload = {
        **selected_path,
        "selection_rule": "max executable decisions; tie longest region path; tie smallest PO ID",
        "executable_decision_count": len(path_decisions),
        "decisions": path_decisions_info,
    }
    (out / "selected_pi_po_path.json").write_text(json.dumps(selected_path_payload, indent=2, sort_keys=True), encoding="utf-8")

    differing: list[dict[str, Any]] = []
    used_regions: set[int] = set()
    used_roots: set[int] = set()
    for decision, stats in zip(path_decisions, [next(x for x in decision_rows if x["root"] == int(decision["root"]) and x["cut_slot"] == int(decision["cut_slot"])) for decision in path_decisions]):
        if not stats["gpu_choice_defined_for_decision"]:
            continue
        if int(stats["original_gpu_selected_subgraph"]) == int(stats["litecut_local_best_subgraph"]):
            continue
        region = int(decision["decision_owner_region"])
        root = int(decision["root"])
        if root in used_roots or region in used_regions:
            continue
        differing.append({
            "region_id": region,
            "root": root,
            "cut_slot": int(decision["cut_slot"]),
            "gpu_subgraph_index": int(stats["original_gpu_selected_subgraph"]),
            "litecut_local_best_subgraph_index": int(stats["litecut_local_best_subgraph"]),
        })
        used_regions.add(region)
        used_roots.add(root)
        if len(differing) == 4:
            break
    config_file = out / "path_configurations.input"
    configurations: list[dict[str, Any]] = []
    with config_file.open("w", encoding="utf-8") as stream:
        options = [("G", d["gpu_subgraph_index"]) for d in differing]
        for bits in itertools.product(("G", "L"), repeat=len(differing)):
            config_id = "".join(bits) if bits else "BASELINE"
            choices = []
            for d, bit in zip(differing, bits):
                choices.append({
                    "root": d["root"], "cut_slot": d["cut_slot"],
                    "subgraph_index": d["gpu_subgraph_index"] if bit == "G" else d["litecut_local_best_subgraph_index"],
                    "label": bit,
                })
                stream.write(f"{config_id}\t{d['root']}\t{d['cut_slot']}\t{choices[-1]['subgraph_index']}\n")
            if not choices:
                stream.write("BASELINE\t-1\t-1\t-1\n")
            configurations.append({"config_id": config_id, "requested_choices": choices, "labels": list(bits)})
    (out / "path_decision_candidate_index.json").write_text(
        json.dumps({"selected_path": selected_path_payload, "differing_decisions": differing}, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    replay_dir = out / "execution_aware_replay"
    run_replay(args.gpuls, args.original_aig, args.region_map, config_file, replay_dir)
    manifest = load_jsonl(replay_dir / "replay_manifest.jsonl")
    replay_results: list[dict[str, Any]] = []
    cec_rows: list[dict[str, Any]] = []
    for item, config in zip(manifest, configurations):
        trace_path = replay_dir / item["trace_path"]
        aig_path = replay_dir / item["aig_path"]
        trace = parse_replay_trace(trace_path)
        statuses = [{"root": x["root"], "status": x["final_status"], "cut_slot": x["cut_slot"],
                     "selected_subgraph": x["selected_subgraph"], "damaged_cut_leaves": x["damaged_cut_leaves"],
                     "damaged_by_prior_root": x["damaged_by_prior_root"], "saved": x["saved"], "added": x["added"]}
                    for x in trace if x["preselected"]]
        replay_results.append({**config, "aig_path": item["aig_path"], "trace_path": item["trace_path"], "realized_statuses": statuses})
        cec_log = out / "cec_logs" / f"{config['config_id']}.log"
        cec_log.parent.mkdir(parents=True, exist_ok=True)
        passed = run_cec(args.abc, args.original_aig, aig_path, cec_log)
        cec_rows.append({"config_id": config["config_id"], "cec_pass": str(passed).lower(), "aig_path": item["aig_path"], "trace_path": item["trace_path"]})
    with (out / "execution_aware_replay_results.jsonl").open("w", encoding="utf-8") as stream:
        for row in replay_results:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    write_csv(out / "cec_results.csv", cec_rows, ["config_id", "cec_pass", "aig_path", "trace_path"])

    defined = [r for r in decision_rows if r["gpu_choice_defined_for_decision"]]
    agreement = sum(r["gpu_choice_equals_litecut_local_best"] == "true" for r in defined)
    summary = {
        "gate": "Gate 3E",
        "input_path": str(args.original_aig),
        "input_sha256": sha256(args.original_aig),
        "candidate_count": total_candidates,
        "executable_decisions": len(decisions),
        "decisions_with_at_least_two_candidates": sum(r["candidate_count"] >= 2 for r in decision_rows),
        "gpu_choice_defined_decisions": len(defined),
        "gpu_choice_equals_litecut_local_best": agreement,
        "gpu_choice_differs_from_litecut_local_best": len(defined) - agreement,
        "selected_path_po_id": int(selected_path["po_id"]),
        "selected_path_region_sequence": selected_path["compressed_region_sequence"],
        "selected_path_executable_decisions": len(path_decisions),
        "differing_path_decisions_used": differing,
        "configuration_count": len(configurations),
        "cec_pass_count": sum(row["cec_pass"] == "true" for row in cec_rows),
        "cec_fail_count": sum(row["cec_pass"] != "true" for row in cec_rows),
        "mapping_called": False,
        "sta_called": False,
        "gradient_called": False,
        "training_called": False,
        "region_state_enumeration": False,
        "deterministic_order": "root, cut_slot, subgraph_index; path decisions in node_path order; configs lexicographic product order",
    }
    (out / "gate3e_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (out / "mapping_free_bridge_audit.md").write_text(
        "# Gate 3E mapping-free bridge audit\n\n"
        f"- original AIG: `{args.original_aig}`\n"
        f"- checkpoint: `{args.checkpoint}` ({sha256(args.checkpoint)})\n"
        f"- feature implementation: `{args.cutlib}/src/data_preprocess/feat_extraction_v3.py:CompleteCutFeatureExtractor`\n"
        f"- model implementation: `{args.cutlib}/src/extract_cutlib_student/modeling.py:StudentModel`\n"
        "- replacement materialization: Gate 2A `materialize_candidate_subgraph`\n"
        "- ABC mapping called: `false`\n"
        "- STA called: `false`\n"
        "- training called: `false`\n"
        "- prediction field: `predicted_local_abc_delay`\n",
        encoding="utf-8",
    )
    (out / "gate3e_report.md").write_text(
        "# Gate 3E\n\n"
        f"- executable decisions: {len(decisions)}\n"
        f"- replacement candidates scored: {total_candidates}\n"
        f"- decisions with >=2 candidates: {summary['decisions_with_at_least_two_candidates']}\n"
        f"- GPU-choice comparisons defined: {len(defined)}; agree: {agreement}; differ: {len(defined) - agreement}\n"
        f"- selected PO: {selected_path['po_id']}\n"
        f"- selected region path: {' -> '.join(map(str, selected_path['compressed_region_sequence']))}\n"
        f"- path executable decisions: {len(path_decisions)}\n"
        f"- path differing decisions used: {len(differing)}; configurations: {len(configurations)}\n"
        f"- CEC PASS: {summary['cec_pass_count']}/{len(configurations)}\n\n"
        "Configurations are ordered decision/candidate choices. They are replayed from the same original AIG through the existing replacement-DAG builder and CPU sequential install semantics; no region-state enumeration is used.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
