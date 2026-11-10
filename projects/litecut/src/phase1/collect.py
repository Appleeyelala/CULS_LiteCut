"""Collect a reproducible Phase-1 snapshot for one AIG.

This intentionally does not modify the network.  It records the mapped STA
summary, the critical path printed by ABC, all feasible cuts printed by the
cutLib ABC build, and MFFC membership.  Candidate prediction/replay is a
later stage once this state-level data is available.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
import shutil
from collections import defaultdict
from pathlib import Path


def run_abc(abc: Path, workdir: Path, command: str) -> str:
    p = subprocess.run(
        [str(abc), "-c", command], cwd=workdir, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )
    if p.returncode:
        raise RuntimeError(f"ABC failed ({p.returncode}): {command}\n{p.stdout}")
    return p.stdout


def parse_stime(text: str) -> dict[str, float | None]:
    area = re.search(r"Area\s*=\s*([0-9]+(?:\.[0-9]+)?)", text)
    delay = re.search(r"Delay\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*ps", text, re.I)
    if delay is None:
        # The cutLib ABC fork writes numeric STA summaries to its report.
        delay = re.search(r"(?:Delay|delay)[^0-9]*([0-9]+(?:\.[0-9]+)?)", text)
    return {"area": float(area.group(1)) if area else None,
            "delay_ps": float(delay.group(1)) if delay else None}


def parse_critical_path(text: str) -> dict:
    lines = [line.strip() for line in text.splitlines() if "Arrival" in line or "Primary input" in line or "Primary output" in line]
    nodes = []
    for line in lines:
        m = re.search(r"Level\s+(\d+)\s*:\s*(.*)", line)
        if m:
            node = re.search(r"\b(n\d+)\b", m.group(2))
            arrival = re.search(r"Arrival(?: time)?\s*=\s*([0-9]+(?:\.[0-9]+)?)", m.group(2))
            nodes.append({"level": int(m.group(1)), "node": node.group(1) if node else None,
                          "arrival": float(arrival.group(1)) if arrival else None, "text": m.group(2)})
    source = re.search(r"from PI\s+\"([^\"]+)\"", text)
    sink = re.search(r"to PO\s+\"([^\"]+)\"", text)
    return {"raw": text, "source": source.group(1) if source else None,
            "sink": sink.group(1) if sink else None, "nodes": nodes}


def parse_cuts(text: str) -> list[dict]:
    records: list[dict] = []
    current: int | None = None
    for line in text.splitlines():
        m = re.match(r"Node\s+(\d+)", line.strip())
        if m:
            current = int(m.group(1))
            continue
        m = re.search(r"Truth\s*:\s*(\d+)\s+Leaves:\s*(.*)", line)
        if m and current is not None:
            leaves = [int(x) for x in re.findall(r"\d+", m.group(2))]
            candidate_id = sum(1 for item in records if item["target_id"] == current)
            records.append({"target_id": current, "candidate_id": candidate_id,
                            "candidate_type": "cut", "truth": int(m.group(1)),
                            "leaves": leaves, "cut_size": len(leaves)})
    return records


def parse_mffc(text: str) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for line in text.splitlines():
        m = re.search(r"Node\s*=\s*(n?\d+).*?\((.*?)\)\s*$", line)
        if m:
            result[m.group(1)] = [int(x[1:]) for x in re.findall(r"n\d+", m.group(2))]
    return result


def build_conflict_graph(candidates: list[dict], mffc: dict[str, list[int]]) -> dict:
    """Build conservative conflicts from same-target and MFFC overlap."""
    nodes = [f"{c['target_id']}:{c['candidate_id']}" for c in candidates]
    edges = []
    reasons = {}
    for i, left in enumerate(candidates):
        left_key = nodes[i]
        left_nodes = set(mffc.get(str(left["target_id"]), []))
        for j in range(i + 1, len(candidates)):
            right = candidates[j]
            right_key = nodes[j]
            why = []
            if left["target_id"] == right["target_id"]:
                why.append("same_target")
            right_nodes = set(mffc.get(str(right["target_id"]), []))
            if left_nodes and right_nodes and left_nodes & right_nodes:
                why.append("mffc_overlap")
            if why:
                edge = [left_key, right_key]
                edges.append(edge)
                reasons["|".join(edge)] = why
    return {"nodes": nodes, "edges": edges, "edge_reasons": reasons,
            "source": "target_and_mffc", "conservative": True}


def collect(abc: Path, lib: Path, aig: Path, output: Path, sta_abc: Path | None = None,
            path_k: int = 10, checkpoint: Path | None = None, paths_file: Path | None = None) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="phase1_", dir=output.parent) as tmp:
        workdir = Path(tmp)
        read = f'read "{lib}"; read "{aig}"; '
        sta_runner = sta_abc or abc
        sta_out = run_abc(sta_runner, workdir, read + "map; topo; stime")
        path_out = run_abc(sta_runner, workdir, read + "map; topo; print_delay")
        sta_report = workdir / "abc_stime_report.log"
        path_report = workdir / "abc_logictopo.log"
        if sta_report.exists():
            sta_out += "\n" + sta_report.read_text(errors="replace")
        if path_report.exists():
            path_out += "\n" + path_report.read_text(errors="replace")
        # -c is an extension in the cutLib ABC build.  Keep a partial snapshot
        # when a standard ABC binary is supplied instead of silently treating
        # missing candidate data as an empty candidate set.
        help_out = run_abc(abc, workdir, "rewrite -h")
        cut_mode_available = "-c" in help_out
        if cut_mode_available:
            run_abc(abc, workdir, read + "rewrite -c")
        cut_text = (workdir / "cut_trave.log").read_text(errors="replace") if (workdir / "cut_trave.log").exists() else ""
        mffc_out = run_abc(sta_runner, workdir, read + "map; topo; print_mffc")
        # print_mffc in the fork is emitted in the command log rather than stdout.
        if (workdir / "abc_logictopo.log").exists():
            mffc_out += "\n" + (workdir / "abc_logictopo.log").read_text(errors="replace")
        cec_out = run_abc(abc, workdir, f'read "{aig}"; cec "{aig}"')
        # Keep raw fork reports beside the structured snapshot for later parsers.
        raw_dir = output.parent / (output.stem + "_raw")
        raw_dir.mkdir(parents=True, exist_ok=True)
        for name in ("abc_stime_report.log", "abc_logictopo.log", "cut_trave.log", "abc_map.log"):
            source = workdir / name
            if source.exists():
                shutil.copyfile(source, raw_dir / name)
        payload = {
            "schema_version": 1,
            "abc": str(abc), "sta_abc": str(sta_runner), "library": str(lib), "aig": str(aig),
            "sta": {"raw": sta_out, **parse_stime(sta_out)},
            "critical_path": parse_critical_path(path_out),
            "candidates": parse_cuts(cut_text),
            "candidate_collection": {"available": cut_mode_available, "reason": None if cut_mode_available else "ABC binary does not support rewrite -c"},
            "mffc": parse_mffc(mffc_out),
            "cec": {"raw": cec_out, "passed": "equivalent" in cec_out.lower() or "verifies" in cec_out.lower()},
            "raw_logs_dir": str(raw_dir),
        }
        by_target: dict[str, list[dict]] = {}
        for item in payload["candidates"]:
            by_target.setdefault(str(item["target_id"]), []).append(item)
        payload["candidate_summary"] = {
            "targets": len(by_target),
            "total": len(payload["candidates"]),
            "min_per_target": min((len(v) for v in by_target.values()), default=0),
            "max_per_target": max((len(v) for v in by_target.values()), default=0),
            "mean_per_target": (len(payload["candidates"]) / len(by_target)) if by_target else 0.0,
        }
        payload["conflict_graph"] = build_conflict_graph(payload["candidates"], payload["mffc"])
        path_nodes = [item["node"] for item in payload["critical_path"]["nodes"] if item.get("node")]
        path_node_ids = {int(node[1:]) for node in path_nodes if node[1:].isdigit()}
        grouped = defaultdict(list)
        for item in payload["candidates"]:
            grouped[str(item["target_id"])].append(item)
            item["path_ids"] = ["critical_path_0"] if item["target_id"] in path_node_ids else []
        payload["targets"] = [
            {"target_id": int(target), "path_ids": ["critical_path_0"] if int(target) in path_node_ids else [],
             "candidates": items, "candidate_count": len(items)}
            for target, items in sorted(grouped.items(), key=lambda pair: int(pair[0]))
        ]
        payload["path_collection"] = {
            "requested_k": path_k,
            "available_k": 1 if payload["critical_path"]["nodes"] else 0,
            "status": "top1_only",
            "reason": "Current ABC print_delay exports one critical path; Top-K exporter is not yet available.",
        }
        if paths_file:
            supplied = json.loads(paths_file.read_text(encoding="utf-8"))
            if isinstance(supplied, dict):
                supplied = supplied.get("critical_paths", supplied.get("paths", []))
            if not isinstance(supplied, list):
                raise ValueError("paths file must contain a list or critical_paths/paths list")
            payload["critical_paths"] = supplied[:path_k]
            payload["path_collection"].update({"available_k": len(payload["critical_paths"]), "status": "external_json"})
            node_to_paths = defaultdict(list)
            for path in payload["critical_paths"]:
                pid = path.get("path_id", len(node_to_paths))
                for node in path.get("nodes", []):
                    node_id = node.get("node_id", node.get("node")) if isinstance(node, dict) else node
                    try: node_to_paths[int(str(node_id).lstrip("n"))].append(pid)
                    except (TypeError, ValueError): pass
            for item in payload["candidates"]:
                item["path_ids"] = node_to_paths.get(item["target_id"], [])
            for target in payload["targets"]:
                target["path_ids"] = node_to_paths.get(target["target_id"], [])
        payload["replay"] = {
            "available": False,
            "reason": "The current ABC command line exports cuts but not a candidate-id replay command.",
        }
        payload["prediction"] = {"available": False, "checkpoint": str(checkpoint) if checkpoint else None,
                                  "reason": "candidate feature bridge is not implemented" if checkpoint else "checkpoint_not_provided"}
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        base = output.parent
        (base / "paths").mkdir(exist_ok=True)
        (base / "candidates").mkdir(exist_ok=True)
        (base / "predictions").mkdir(exist_ok=True)
        (base / "conflicts").mkdir(exist_ok=True)
        (base / "paths" / "topk_paths.json").write_text(json.dumps(payload.get("critical_paths", [payload["critical_path"]]), indent=2), encoding="utf-8")
        (base / "candidates" / "candidates.jsonl").write_text("\n".join(json.dumps(x) for x in payload["candidates"]) + "\n", encoding="utf-8")
        (base / "predictions" / "litecut_predictions.jsonl").write_text("\n".join(json.dumps({"state_id": "state_0000", "target_id": x["target_id"], "candidate_id": x["candidate_id"], "litecut_prediction": None, "prediction_status": "unavailable", "reason": payload["prediction"]["reason"]}) for x in payload["candidates"]) + "\n", encoding="utf-8")
        (base / "conflicts" / "conflict_graph.json").write_text(json.dumps(payload["conflict_graph"], indent=2), encoding="utf-8")
        print(f"wrote {output}")
        print(f"delay_ps={payload['sta']['delay_ps']} candidates={len(payload['candidates'])} path_nodes={len(payload['critical_path']['nodes'])}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--abc", type=Path, required=True)
    p.add_argument("--sta-abc", type=Path, default=None,
                   help="Optional ABC binary for parseable map/STA/path output")
    p.add_argument("--lib", type=Path, default=Path("lib/asap7.lib"))
    p.add_argument("--aig", type=Path, required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--output", type=Path)
    group.add_argument("--experiment-root", type=Path,
                   help="Optional root; writes structured output under <root>/<AIG-stem>/")
    p.add_argument("--path-k", type=int, default=10)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--paths-file", type=Path, help="Optional externally generated Top-K path JSON")
    args = p.parse_args()
    output = args.output.resolve() if args.output else None
    if args.experiment_root:
        output = args.experiment_root.resolve() / args.aig.stem / "state_0000.json"
    assert output is not None
    if args.path_k < 1: raise SystemExit("--path-k must be positive")
    collect(args.abc.resolve(), args.lib.resolve(), args.aig.resolve(), output,
            args.sta_abc.resolve() if args.sta_abc else None, args.path_k,
            args.checkpoint.resolve() if args.checkpoint else None,
            args.paths_file.resolve() if args.paths_file else None)


if __name__ == "__main__":
    main()
