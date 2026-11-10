#!/usr/bin/env python3
"""Gate 2C timing-active rewrite decision screening runner.

The runner has three deliberately separate responsibilities:

* replay every candidate of the frozen decision sample with the Gate 1 engine;
* reuse ``gate2a_candidate_inference.py`` for mapping-free Student inference;
* map and time only the two preselected candidates for each decision with the
  exact Gate 2B ABC flow.

No timing value participates in candidate-pair selection.  All expensive work
is intended to run from a data-local Slurm workspace.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


METRIC_RE = __import__("re").compile(
    r"Area\s*=\s*([0-9]+(?:\.[0-9]+)?) .*?"
    r"Delay\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*ps",
    __import__("re").I,
)
METRIC_RE_COMPACT = __import__("re").compile(
    r"Area\s*=\s*([0-9]+(?:\.[0-9]+)?) .*?"
    r"Delay\s*=\s*([0-9]+(?:\.[0-9]+)?)",
    __import__("re").I,
)
GATES_RE = __import__("re").compile(r"Gates\s*=\s*([0-9]+)", __import__("re").I)
STATS_RE = __import__("re").compile(
    r"i/o\s*=\s*([0-9]+)\s*/\s*([0-9]+).*?and\s*=\s*([0-9]+)",
    __import__("re").I,
)
STATS_RE2 = __import__("re").compile(r"And\s*=\s*([0-9]+)", __import__("re").I)


DEFAULT_RUN_ROOT = Path("/data/wushangge/gate2c_runs/gate2c_20260917")
DEFAULT_MANIFEST = DEFAULT_RUN_ROOT / "input" / "sampling_manifest.json"


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def parse_metrics(text: str) -> tuple[float | None, float | None, str | None]:
    for line in text.splitlines():
        match = METRIC_RE.search(line) or METRIC_RE_COMPACT.search(line)
        if match:
            return float(match.group(1)), float(match.group(2)), line.strip()
    match = __import__("re").search(
        r"Area[^0-9]*([0-9]+(?:\.[0-9]+)?).*?"
        r"Delay[^0-9]*([0-9]+(?:\.[0-9]+)?)",
        text,
        __import__("re").I | __import__("re").S,
    )
    if match:
        return (
            float(match.group(1)),
            float(match.group(2)),
            match.group(0).strip().replace("\n", " "),
        )
    return None, None, None


def parse_gate_count(text: str) -> int | None:
    for line in text.splitlines():
        match = GATES_RE.search(line)
        if match:
            return int(match.group(1))
        match = STATS_RE.search(line)
        if match:
            return int(match.group(3))
        match = STATS_RE2.search(line)
        if match:
            return int(match.group(1))
    return None


def run_logged(
    command: list[str],
    cwd: Path,
    log_path: Path,
    env: dict[str, str] | None = None,
    timeout_seconds: int = 600,
) -> tuple[int, str, bool]:
    cwd.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timed_out = False
    try:
        process = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
        output = process.stdout or ""
        return_code = process.returncode
    except subprocess.TimeoutExpired as error:
        timed_out = True
        output = (error.stdout or "")
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        output += f"\nTIMEOUT after {timeout_seconds} seconds\n"
        return_code = 124
    log_path.write_text(output, encoding="utf-8", errors="replace")
    return return_code, output, timed_out


def load_bridge(repo_root: Path):
    repo_root = repo_root.resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    bridge = importlib.import_module("gate2a_candidate_inference")
    return bridge


def stage_file(source: Path, destination: Path) -> Path:
    """Copy a small immutable input into the run workspace and return it."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256(destination) != sha256(source):
            raise ValueError(f"staged file differs from source: {destination}")
        return destination
    shutil.copy2(source, destination)
    return destination


def candidate_id(decision_id: str, subgraph_index: int) -> str:
    return f"{decision_id}_sub{int(subgraph_index):02d}"


def prepare_candidate_record(
    decision: dict[str, Any],
    subgraph_index: int,
    provenance_decision: dict[str, Any],
) -> dict[str, Any]:
    record = dict(decision)
    # Gate 2A accepts the Gate 1 provenance field names below.  Keep the
    # original audit fields as well, so every candidate remains self-auditing.
    record.update(provenance_decision)
    record["design"] = decision.get("design", "c1355.aig")
    record["decision_id"] = decision["decision_id"]
    record["root"] = int(decision["root"])
    record["target_root"] = int(provenance_decision["target_root"])
    record["cut_slot"] = int(provenance_decision["cut_slot"])
    record["candidate_count"] = int(decision["candidate_count"])
    record["candidate_subgraph_index"] = int(subgraph_index)
    record["subgraph_index"] = int(provenance_decision["subgraph_index"])
    record["saved_region_nodes"] = [int(x) for x in decision.get("saved_region_nodes", [])]
    record["removable_region_boundary"] = [
        int(x) for x in decision.get("removable_region_boundary", [])
    ]
    return record


def make_aig_link(record_path: Path, aig_path: Path) -> None:
    link_path = record_path.with_suffix(".aig")
    if link_path.is_symlink() or link_path.exists():
        if link_path.is_symlink() and link_path.resolve() == aig_path.resolve():
            return
        raise FileExistsError(f"unexpected existing inference-side AIG: {link_path}")
    link_path.symlink_to(os.path.relpath(aig_path, link_path.parent))


def feature_hash(features: Iterable[float]) -> str:
    return hashlib.sha256(canonical_json([float(x) for x in features]).encode()).hexdigest()


def infer_candidate(
    bridge: Any,
    record_path: Path,
    original: Path,
    library: Path,
    checkpoint: Path,
    predictions_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    first = bridge.infer_one(record_path, original, library, checkpoint)
    second = bridge.infer_one(record_path, original, library, checkpoint)
    first_features = first["continuous_features_19"]
    second_features = second["continuous_features_19"]
    first_prediction = first["inference"]["predicted_raw_abc_delay"]
    second_prediction = second["inference"]["predicted_raw_abc_delay"]
    raw_features_close = all(
        math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)
        for left, right in zip(first["raw_features_25"], second["raw_features_25"])
    )
    prediction_close = math.isclose(
        float(first_prediction), float(second_prediction), rel_tol=0.0, abs_tol=1e-12
    )
    deterministic = (
        first_features == second_features
        and raw_features_close
        and first["replacement_dag"] == second["replacement_dag"]
        and prediction_close
    )
    finite = (
        all(math.isfinite(float(x)) for x in first["raw_features_25"])
        and math.isfinite(float(first_prediction))
    )
    if not finite:
        raise ValueError(f"non-finite LiteCut inference for {record_path}")
    if not deterministic:
        raise ValueError(f"non-deterministic LiteCut inference for {record_path}")

    result = dict(first)
    result["feature_hash_19"] = feature_hash(first_features)
    result["finite_features_25"] = True
    result["finite_prediction"] = True
    result["deterministic_repeat"] = True
    result["repeat_prediction"] = second_prediction
    result["replacement_dag_structural_hash"] = hashlib.sha256(
        canonical_json(first["replacement_dag"]).encode()
    ).hexdigest()
    result["inference_repeat_check"] = {
        "feature_vectors_exact": first_features == second_features,
        "raw_features_exact": first["raw_features_25"] == second["raw_features_25"],
        "raw_features_close_abs_tol_1e-12": raw_features_close,
        "replacement_dag_exact": first["replacement_dag"] == second["replacement_dag"],
        "prediction_exact": first_prediction == second_prediction,
        "prediction_close_abs_tol_1e-12": prediction_close,
    }
    output_path = predictions_dir / f"{result['candidate_id']}.json"
    write_json(output_path, result)
    return result, {
        "candidate_id": result["candidate_id"],
        "feature_hash_19": result["feature_hash_19"],
        "predicted_raw_abc_delay": first_prediction,
        "finite": True,
        "deterministic": True,
        "replacement_dag_structural_hash": result["replacement_dag_structural_hash"],
        "output": str(output_path),
    }


def replay_candidate(
    decision: dict[str, Any],
    subgraph_index: int,
    run_root: Path,
    original: Path,
    engine: Path,
    abc: Path,
) -> tuple[dict[str, Any], Path, Path]:
    out = run_root / "output"
    records_dir = out / "candidate_records"
    aigs_dir = out / "replayed_aigs"
    work_dir = run_root / "work" / "replay"
    logs_dir = out / "logs"
    cid = candidate_id(decision["decision_id"], subgraph_index)
    record_path = records_dir / f"{cid}.json"
    aig_path = aigs_dir / f"{cid}.aig"
    provenance_path = work_dir / f"{cid}.provenance.json"
    engine_log = logs_dir / f"{cid}.engine.log"
    cec_log = logs_dir / f"{cid}.cec.log"
    for path in (record_path, aig_path, provenance_path):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing to overwrite existing replay artifact: {path}")
    records_dir.mkdir(parents=True, exist_ok=True)
    aigs_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    for name in (
        "GATE1C_SPEC",
        "GATE1A_PROVENANCE",
        "GATE1B_AUDIT",
        "GATE1B_DESIGN",
        "GATE1B_ROOT",
    ):
        env.pop(name, None)
    env.update(
        {
            "GATE1A_ROOT": str(int(decision["root"])),
            "GATE1A_CUT": str(int(decision["cut_slot"])),
            "GATE1A_SUB": str(int(subgraph_index)),
            "GATE1A_PROVENANCE": str(provenance_path),
            "LD_LIBRARY_PATH": str(engine.parent),
        }
    )
    replay_command = [
        str(engine),
        "-c",
        f'gread "{original}"; grw; gwrite "{aig_path}"',
    ]
    exit_code, _, timed_out = run_logged(
        replay_command, work_dir, engine_log, env=env, timeout_seconds=600
    )
    if timed_out or exit_code != 0:
        raise RuntimeError(f"Gate 1 replay failed for {cid}: exit={exit_code}")
    require_file(aig_path, f"replayed AIG {cid}")
    require_file(provenance_path, f"provenance {cid}")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    decisions = provenance.get("decisions")
    if not isinstance(decisions, list) or len(decisions) != 1:
        raise ValueError(f"expected one provenance decision for {cid}")
    pdecision = decisions[0]
    expected = {
        "target_root": int(decision["root"]),
        "cut_slot": int(decision["cut_slot"]),
        "subgraph_index": int(subgraph_index),
    }
    for field, value in expected.items():
        if int(pdecision[field]) != value:
            raise ValueError(
                f"provenance mismatch for {cid}: {field}={pdecision[field]} expected {value}"
            )
    if pdecision.get("installation_kind") not in {
        "NO_OP_FULL_CORRESPONDENCE",
        "PHYSICAL_REPLACEMENT",
    }:
        raise ValueError(f"illegal installation kind for {cid}: {pdecision.get('installation_kind')}")
    if not all(bool(pdecision.get(key)) for key in ("seen", "evaluated", "built", "installed")):
        raise ValueError(f"incomplete Gate 1 provenance for {cid}")

    cec_command = [str(abc), "-c", f'cec "{original}" "{aig_path}"']
    cec_exit, cec_output, cec_timeout = run_logged(
        cec_command, work_dir / "cec", cec_log, timeout_seconds=600
    )
    cec_pass = (
        not cec_timeout
        and cec_exit == 0
        and "Networks are equivalent" in cec_output
    )
    if not cec_pass:
        raise RuntimeError(f"CEC failed for {cid}: exit={cec_exit}")

    record = prepare_candidate_record(decision, subgraph_index, pdecision)
    record.update(
        {
            "candidate_id": cid,
            "replay": {
                "engine": str(engine),
                "command": replay_command,
                "provenance": str(provenance_path),
                "engine_log": str(engine_log),
            },
            "aig": {
                "path": str(aig_path),
                "sha256": sha256(aig_path),
            },
            "cec": {
                "status": "PASS",
                "command": cec_command,
                "log": str(cec_log),
            },
            "provenance_sha256": sha256(provenance_path),
        }
    )
    write_json(record_path, record)
    make_aig_link(record_path, aig_path)
    return record, record_path, aig_path


def run_sta(
    tag: str,
    aig_path: Path,
    run_root: Path,
    original: Path,
    abc: Path,
    library: Path,
    library_sha256: str,
) -> dict[str, Any]:
    out = run_root / "output"
    sta_dir = out / "sta"
    logs_dir = out / "logs"
    work = run_root / "work" / "sta" / tag
    result_path = sta_dir / f"{tag}.json"
    log_path = logs_dir / f"{tag}.sta.log"
    command_text = f'read "{library}"; read "{aig_path}"; map; topo; stime'
    command = [str(abc), "-c", command_text]
    exit_code, stdout, timed_out = run_logged(command, work, log_path, timeout_seconds=600)

    side_parts: list[str] = []
    for side_name in ("abc_stime_report.log", "abc_logictopo.log", "abc_map.log"):
        side_path = work / side_name
        if side_path.is_file():
            side_text = side_path.read_text(errors="replace")
            side_parts.append(f"\n===== {side_name} =====\n{side_text}")
            shutil.copy2(side_path, logs_dir / f"{tag}_{side_name}")
    combined = stdout + "".join(side_parts)
    if side_parts:
        log_path.write_text(combined, encoding="utf-8", errors="replace")
    area, delay, timing_line = parse_metrics(combined)
    gate_count = parse_gate_count(combined)
    result = {
        "tag": tag,
        "input_aig": str(aig_path),
        "input_aig_sha256": sha256(aig_path),
        "abc_binary": str(abc),
        "library": str(library),
        "library_sha256": library_sha256,
        "command": command_text,
        "working_directory": str(work),
        "log": str(log_path),
        "exit_code": exit_code,
        "timed_out": timed_out,
        "area": area,
        "whole_design_delay_ps": delay,
        "gate_count": gate_count,
        "raw_timing_line": timing_line,
        "stdout": stdout,
    }
    write_json(result_path, result)
    if (
        timed_out
        or exit_code != 0
        or area is None
        or delay is None
        or not math.isfinite(float(delay))
    ):
        raise RuntimeError(f"fixed mapping/STA failed or was unparseable for {tag}")
    return result


def choose_candidate_pair(candidates: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], str]:
    physical = sorted(
        (candidate for candidate in candidates
         if candidate["installation_kind"] == "PHYSICAL_REPLACEMENT"),
        key=lambda candidate: int(candidate["subgraph_index"]),
    )
    no_op = sorted(
        (candidate for candidate in candidates
         if candidate["installation_kind"] == "NO_OP_FULL_CORRESPONDENCE"),
        key=lambda candidate: int(candidate["subgraph_index"]),
    )
    if len(physical) >= 2:
        return physical[0], physical[1], "physical-vs-physical"
    if no_op and physical:
        return no_op[0], physical[0], "no-op-vs-physical"
    raise ValueError("decision has no legal pair under the frozen candidate-pair rule")


def local_preference(pred_a: float, pred_b: float) -> str:
    if pred_a < pred_b:
        return "A"
    if pred_b < pred_a:
        return "B"
    return "TIE_OR_UNCERTAIN"


def timing_preference(delay_a: float, delay_b: float, tolerance: float) -> str:
    if delay_a < delay_b - tolerance:
        return "A"
    if delay_b < delay_a - tolerance:
        return "B"
    return "TIE_OR_UNCERTAIN"


def classify_delta(delta: float, tolerance: float) -> str:
    epsilon = 1e-12
    if abs(delta) > tolerance + epsilon:
        return "TIMING_ACTIVE"
    if abs(abs(delta) - tolerance) <= epsilon:
        return "BORDERLINE"
    return "TIMING_INACTIVE"


def compatible_pair(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Apply the Gate 1 batch compatibility test to two selected decisions.

    Gate 1 preserves each root wrapper.  Therefore the removable region is the
    audited saved region with the root itself removed; this also permits the
    documented dependency case where one root is a boundary leaf of the other.
    """
    root_a = int(a["root"])
    root_b = int(b["root"])
    removed_a = set(int(x) for x in a.get("saved_region_nodes", [])) - {root_a}
    removed_b = set(int(x) for x in b.get("saved_region_nodes", [])) - {root_b}
    boundary_a = set(int(x) for x in a.get("cut_leaves", []))
    boundary_b = set(int(x) for x in b.get("cut_leaves", []))
    checks = {
        "removable_regions_disjoint": removed_a.isdisjoint(removed_b),
        "root_A_not_removed_by_B": root_a not in removed_b,
        "root_B_not_removed_by_A": root_b not in removed_a,
        "boundary_A_not_removed_by_B": boundary_a.isdisjoint(removed_b),
        "boundary_B_not_removed_by_A": boundary_b.isdisjoint(removed_a),
    }
    dependencies: list[dict[str, Any]] = []
    if root_a in boundary_b:
        dependencies.append({"from": root_a, "to": root_b, "reason": "root_A_is_boundary_of_B"})
    if root_b in boundary_a:
        dependencies.append({"from": root_b, "to": root_a, "reason": "root_B_is_boundary_of_A"})
    return {
        "compatible": all(checks.values()),
        "checks": checks,
        "dependencies": dependencies,
        "root_A": root_a,
        "root_B": root_b,
        "removed_A": sorted(removed_a),
        "removed_B": sorted(removed_b),
        "boundary_A": sorted(boundary_a),
        "boundary_B": sorted(boundary_b),
        "method": "Gate 1 preserved-root-wrapper compatibility test",
    }


def search_next_gate_pair(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    active = [
        decision for decision in decisions
        if decision["timing_class"] == "TIMING_ACTIVE"
    ]
    physical_active = [
        decision for decision in active
        if decision["candidate_pair_type"] == "physical-vs-physical"
    ]
    # This order is part of the selection policy: active physical pairs first,
    # then active decisions with a no-op/physical pair.  It never uses the
    # magnitude or sign of a timing delta.
    groups = [physical_active, active]
    checked: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for group in groups:
        ordered = sorted(group, key=lambda item: item["decision_id"])
        for index, first in enumerate(ordered):
            for second in ordered[index + 1:]:
                key = (first["decision_id"], second["decision_id"])
                if key in seen:
                    continue
                seen.add(key)
                evidence = compatible_pair(first, second)
                checked.append({
                    "decision_ids": list(key),
                    "both_physical_pairs": (
                        first["candidate_pair_type"] == "physical-vs-physical"
                        and second["candidate_pair_type"] == "physical-vs-physical"
                    ),
                    "compatibility": evidence,
                })
                if evidence["compatible"]:
                    return {
                        "status": "FOUND",
                        "selection_rule": {
                            "only_timing_active": True,
                            "prefer_both_physical_vs_physical": True,
                            "sort": "decision_id ascending",
                            "do_not_use_timing_delta_for_pair_selection": True,
                        },
                        "selected_pair": {
                            "decision_ids": list(key),
                            "candidate_pair_types": [
                                first["candidate_pair_type"],
                                second["candidate_pair_type"],
                            ],
                        },
                        "compatibility": evidence,
                        "checked_pair_count_before_selection": len(checked),
                    }
    return {
        "status": "BLOCKED",
        "selection_rule": {
            "only_timing_active": True,
            "prefer_both_physical_vs_physical": True,
            "sort": "decision_id ascending",
            "do_not_use_timing_delta_for_pair_selection": True,
        },
        "selected_pair": None,
        "compatibility": None,
        "checked_pair_count": len(checked),
        "reason": "No compatible pair exists in the timing-active pool under the fixed test",
        "checked_pairs": checked,
    }


def validate_joint_replay(
    pair_search: dict[str, Any],
    decisions: list[dict[str, Any]],
    run_root: Path,
    original: Path,
    engine: Path,
    abc: Path,
) -> dict[str, Any]:
    """Prove all four selected-candidate combinations jointly replay and CEC.

    This validation deliberately does not run mapping or timing.  It establishes
    only the compatibility prerequisite for a later 2x2 interaction Gate.
    """
    if pair_search.get("status") != "FOUND":
        return pair_search
    selected_ids = pair_search["selected_pair"]["decision_ids"]
    by_id = {decision["decision_id"]: decision for decision in decisions}
    first, second = (by_id[decision_id] for decision_id in selected_ids)
    combinations = {
        "AA": (first["candidate_A"], second["candidate_A"]),
        "AB": (first["candidate_A"], second["candidate_B"]),
        "BA": (first["candidate_B"], second["candidate_A"]),
        "BB": (first["candidate_B"], second["candidate_B"]),
    }
    joint_dir = run_root / "output" / "next_gate_joint_replay"
    logs_dir = run_root / "output" / "logs"
    work_dir = run_root / "work" / "next_gate_joint_replay"
    results: dict[str, dict[str, Any]] = {}
    for label, (candidate_1, candidate_2) in combinations.items():
        aig_path = joint_dir / f"{label}.aig"
        provenance_path = joint_dir / f"{label}.json"
        replay_log = logs_dir / f"next_gate_{label}.engine.log"
        cec_log = logs_dir / f"next_gate_{label}.cec.log"
        if aig_path.exists() or provenance_path.exists():
            raise FileExistsError(f"refusing to overwrite joint replay {label}")
        joint_dir.mkdir(parents=True, exist_ok=True)
        spec = (
            f"{first['root']}:{first['cut_slot']}:{candidate_1['subgraph_index']},"
            f"{second['root']}:{second['cut_slot']}:{candidate_2['subgraph_index']}"
        )
        env = os.environ.copy()
        for name in ("GATE1B_AUDIT", "GATE1B_DESIGN", "GATE1B_ROOT"):
            env.pop(name, None)
        env.update({
            "GATE1C_SPEC": spec,
            "GATE1A_ROOT": str(first["root"]),
            "GATE1A_CUT": str(first["cut_slot"]),
            "GATE1A_SUB": str(candidate_1["subgraph_index"]),
            "GATE1A_PROVENANCE": str(provenance_path),
            "LD_LIBRARY_PATH": str(engine.parent),
        })
        command = [
            str(engine),
            "-c",
            f'gread "{original}"; grw; gwrite "{aig_path}"',
        ]
        exit_code, _, timed_out = run_logged(
            command, work_dir / label, replay_log, env=env, timeout_seconds=600
        )
        replay_ok = not timed_out and exit_code == 0 and aig_path.is_file() and provenance_path.is_file()
        cec_exit = None
        cec_pass = False
        if replay_ok:
            cec_command = [str(abc), "-c", f'cec "{original}" "{aig_path}"']
            cec_exit, cec_output, cec_timed_out = run_logged(
                cec_command, work_dir / label / "cec", cec_log, timeout_seconds=600
            )
            cec_pass = (
                not cec_timed_out
                and cec_exit == 0
                and "Networks are equivalent" in cec_output
            )
        results[label] = {
            "candidate_1": {
                "decision_id": first["decision_id"],
                "label": label[0],
                "subgraph_index": candidate_1["subgraph_index"],
                "installation_kind": candidate_1["installation_kind"],
            },
            "candidate_2": {
                "decision_id": second["decision_id"],
                "label": label[1],
                "subgraph_index": candidate_2["subgraph_index"],
                "installation_kind": candidate_2["installation_kind"],
            },
            "forced_spec": spec,
            "replay_exit_code": exit_code,
            "replay_pass": replay_ok,
            "cec_exit_code": cec_exit,
            "cec_status": "PASS" if cec_pass else "FAIL",
            "aig": str(aig_path),
            "aig_sha256": sha256(aig_path) if aig_path.is_file() else None,
            "provenance": str(provenance_path),
            "replay_log": str(replay_log),
            "cec_log": str(cec_log),
        }
    all_pass = all(
        result["replay_pass"] and result["cec_status"] == "PASS"
        for result in results.values()
    )
    pair_search["joint_replay_validation"] = {
        "status": "PASS" if all_pass else "FAIL",
        "all_four_candidate_combinations_replayed": all_pass,
        "cec_all_four": all_pass,
        "mapping_or_sta_run": False,
        "configurations": results,
    }
    if not all_pass:
        pair_search["status"] = "BLOCKED"
        pair_search["reason"] = "Static-compatible pair failed joint replay or CEC"
    return pair_search


def write_screening_csv(path: Path, decisions: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "decision_id", "design", "root", "cut_slot", "cut_leaves",
        "candidate_count", "candidate_pair_type", "candidate_A_subgraph_index",
        "candidate_A_installation_kind", "candidate_A_litecut_predicted_local_abc_delay",
        "candidate_A_feature_hash_19", "candidate_B_subgraph_index",
        "candidate_B_installation_kind", "candidate_B_litecut_predicted_local_abc_delay",
        "candidate_B_feature_hash_19", "litecut_local_preference",
        "litecut_delta_B_minus_A", "D_A_ps", "D_B_ps", "Delta_D_B_minus_A_ps",
        "timing_class", "whole_design_single_decision_preference", "same_direction",
        "cec_A", "cec_B", "finite_and_deterministic",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for decision in decisions:
            a = decision["candidate_A"]
            b = decision["candidate_B"]
            writer.writerow({
                "decision_id": decision["decision_id"],
                "design": decision["design"],
                "root": decision["root"],
                "cut_slot": decision["cut_slot"],
                "cut_leaves": ":".join(str(x) for x in decision["cut_leaves"]),
                "candidate_count": decision["candidate_count"],
                "candidate_pair_type": decision["candidate_pair_type"],
                "candidate_A_subgraph_index": a["subgraph_index"],
                "candidate_A_installation_kind": a["installation_kind"],
                "candidate_A_litecut_predicted_local_abc_delay": a["litecut_predicted_local_abc_delay"],
                "candidate_A_feature_hash_19": a["feature_hash_19"],
                "candidate_B_subgraph_index": b["subgraph_index"],
                "candidate_B_installation_kind": b["installation_kind"],
                "candidate_B_litecut_predicted_local_abc_delay": b["litecut_predicted_local_abc_delay"],
                "candidate_B_feature_hash_19": b["feature_hash_19"],
                "litecut_local_preference": decision["litecut_local_preference"],
                "litecut_delta_B_minus_A": decision["litecut_delta_B_minus_A"],
                "D_A_ps": decision["whole_design_timing"]["A"],
                "D_B_ps": decision["whole_design_timing"]["B"],
                "Delta_D_B_minus_A_ps": decision["whole_design_timing"]["delta_B_minus_A"],
                "timing_class": decision["timing_class"],
                "whole_design_single_decision_preference": decision["whole_design_single_decision_preference"],
                "same_direction": decision["same_direction"],
                "cec_A": a["cec_status"],
                "cec_B": b["cec_status"],
                "finite_and_deterministic": decision["finite_and_deterministic"],
            })


def build_summary(
    manifest: dict[str, Any],
    candidate_pool: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    original_sta: dict[str, Any],
    next_pair: dict[str, Any],
    run_root: Path,
    paths: dict[str, Path],
    status: str,
    error: str | None = None,
) -> dict[str, Any]:
    tolerance = float(manifest["timing_classification"]["timing_tolerance_ps"])
    counts = {
        "sampled_decisions": len(manifest["selected_records"]),
        "valid_decisions": len(decisions),
        "timing_active_count": sum(d["timing_class"] == "TIMING_ACTIVE" for d in decisions),
        "timing_inactive_count": sum(d["timing_class"] == "TIMING_INACTIVE" for d in decisions),
        "borderline_count": sum(d["timing_class"] == "BORDERLINE" for d in decisions),
    }
    counts["timing_active_fraction"] = (
        counts["timing_active_count"] / counts["valid_decisions"]
        if counts["valid_decisions"] else None
    )
    by_type: dict[str, dict[str, Any]] = {}
    for pair_type in ("physical-vs-physical", "no-op-vs-physical"):
        subset = [d for d in decisions if d["candidate_pair_type"] == pair_type]
        by_type[pair_type] = {
            "sampled_decisions": len(subset),
            "timing_active_count": sum(d["timing_class"] == "TIMING_ACTIVE" for d in subset),
            "timing_inactive_count": sum(d["timing_class"] == "TIMING_INACTIVE" for d in subset),
            "borderline_count": sum(d["timing_class"] == "BORDERLINE" for d in subset),
            "timing_active_fraction": (
                sum(d["timing_class"] == "TIMING_ACTIVE" for d in subset) / len(subset)
                if subset else None
            ),
        }
    active = [d for d in decisions if d["timing_class"] == "TIMING_ACTIVE"]
    same = sum(d["same_direction"] is True for d in active)
    opposite = sum(d["same_direction"] is False for d in active)
    tie = sum(d["same_direction"] is None for d in active)
    local_diff_unchanged = sum(
        abs(float(d["litecut_delta_B_minus_A"])) > float(manifest["litecut"]["local_score_difference_tolerance"])
        and d["timing_class"] == "TIMING_INACTIVE"
        for d in decisions
    )
    local_diff_delay_unchanged = sum(
        abs(float(d["litecut_delta_B_minus_A"])) > float(manifest["litecut"]["local_score_difference_tolerance"])
        and abs(float(d["whole_design_timing"]["delta_B_minus_A"])) <= tolerance
        for d in decisions
    )
    return {
        "gate": "Gate 2C",
        "status": status,
        "run_root": str(run_root),
        "sampling": {
            "manifest": str(paths["manifest"]),
            "sampling_rule": manifest["sampling_rule"],
            "candidate_pair_selection_rule": manifest["candidate_pair_selection_rule"],
            "population_size": manifest["candidate_enumeration"]["frame_size"],
            "engine_records_seen_not_sampled": manifest["candidate_enumeration"]["engine_records_seen"],
            "sampled_decision_ids": manifest["selected_decision_ids"],
        },
        "flow": {
            "abc_binary": manifest["timing_classification"]["abc_binary"],
            "library": manifest["timing_classification"]["library"],
            "library_sha256": manifest["timing_classification"]["library_sha256"],
            "script": manifest["timing_classification"]["flow"],
            "timing_tolerance_ps": tolerance,
            "reported_precision_ps": manifest["timing_classification"]["reported_precision_ps"],
        },
        "litecut": manifest["litecut"],
        "counts": counts,
        "pair_type_statistics": by_type,
        "litecut_vs_whole_design_direction": {
            "timing_active_decisions": len(active),
            "same_direction_count": same,
            "opposite_direction_count": opposite,
            "tie_or_uncertain_count": tie,
            "interpretation": "single-decision perturbation direction comparison, not global accuracy",
        },
        "local_score_difference_but_delay_unchanged": {
            "score_difference_tolerance": manifest["litecut"]["local_score_difference_tolerance"],
            "timing_inactive_count": local_diff_unchanged,
            "within_timing_tolerance_including_borderline_count": local_diff_delay_unchanged,
            "definition": "nonzero selected-pair LiteCut delta while whole-design delta is below the fixed precision tolerance",
        },
        "original_sta": original_sta,
        "next_gate_pair_search": next_pair,
        "artifact_paths": {key: str(value) for key, value in paths.items()},
        "candidate_pool_size": len(candidate_pool),
        "decision_screening": decisions,
        "error": error,
    }


def run(args: argparse.Namespace) -> int:
    run_root = args.run_root.resolve()
    manifest_path = args.manifest.resolve()
    require_file(manifest_path, "sampling manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "FROZEN_BEFORE_TIMING":
        raise ValueError("sampling manifest is not frozen")
    selected = manifest.get("selected_records")
    if not isinstance(selected, list) or not selected:
        raise ValueError("sampling manifest has no selected records")
    if args.decision_limit is not None:
        selected = selected[: args.decision_limit]
    if not selected:
        raise ValueError("--decision-limit selected no records")

    input_dir = run_root / "input"
    output_dir = run_root / "output"
    work_dir = run_root / "work"
    for directory in (input_dir, output_dir, work_dir):
        directory.mkdir(parents=True, exist_ok=True)

    original = args.original.resolve()
    library = args.rewrite_library.resolve()
    checkpoint = args.checkpoint.resolve()
    engine = args.gate1_engine.resolve()
    abc = args.abc.resolve()
    litecut_repo = args.litecut_repo.resolve()
    for path, label in (
        (original, "original snapshot"),
        (library, "rewrite library"),
        (checkpoint, "checkpoint"),
        (engine, "Gate 1 engine"),
        (abc, "ABC binary"),
        (litecut_repo / "gate2a_candidate_inference.py", "Gate 2A bridge"),
    ):
        require_file(path, label)
    bridge = load_bridge(litecut_repo)

    manifest_copy = output_dir / "sampling_manifest.json"
    shutil.copy2(manifest_path, manifest_copy)
    paths: dict[str, Path] = {
        "manifest": manifest_copy,
        "candidate_pool": output_dir / "candidate_pool.json",
        "screening_csv": output_dir / "decision_screening.csv",
        "timing_active": output_dir / "timing_active_decisions.json",
        "timing_inactive": output_dir / "timing_inactive_decisions.json",
        "next_gate_pair": output_dir / "next_gate_pair_candidates.json",
        "summary": output_dir / "gate2c_summary.json",
        "report": output_dir / "gate2c_report.md",
    }
    flow_hash = sha256(library)
    write_json(output_dir / "environment.json", {
        "gate": "Gate 2C",
        "run_root": str(run_root),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
        "abc_binary": str(abc),
        "gate1_engine": str(engine),
        "original_snapshot": str(original),
        "original_sha256": sha256(original),
        "rewrite_library": str(library),
        "rewrite_library_sha256": flow_hash,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "python": sys.executable,
        "python_version": sys.version,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "flow": manifest["timing_classification"]["flow"],
    })
    write_json(output_dir / "library_manifest.json", {
        "source_path": str(library),
        "sha256": flow_hash,
        "library_kind": "Gate 1 rewrite_library.inc for candidate construction",
        "timing_library": manifest["timing_classification"]["library"],
        "timing_library_sha256": manifest["timing_classification"]["library_sha256"],
    })

    # This copy is used for every STA invocation.  The source library hash is
    # retained in the manifest; the runner verifies that the staged bytes are
    # identical before it starts timing.
    timing_library = stage_file(args.timing_library.resolve(), input_dir / "asap7.lib")
    if sha256(timing_library) != manifest["timing_classification"]["library_sha256"]:
        raise ValueError("staged timing library hash does not match frozen manifest")

    candidate_pool: list[dict[str, Any]] = []
    decision_results: list[dict[str, Any]] = []
    for ordinal, decision in enumerate(selected):
        decision_id = decision["decision_id"]
        candidates: list[dict[str, Any]] = []
        for subgraph_index in range(int(decision["candidate_count"])):
            record, record_path, aig_path = replay_candidate(
                decision, subgraph_index, run_root, original, engine, abc
            )
            litecut_full, litecut_summary = infer_candidate(
                bridge,
                record_path,
                original,
                library,
                checkpoint,
                output_dir / "litecut_predictions",
            )
            candidate = {
                "candidate_id": record["candidate_id"],
                "decision_id": decision_id,
                "root": int(record["target_root"]),
                "cut_slot": int(record["cut_slot"]),
                "cut_leaves": [int(x) for x in record["cut_leaves"]],
                "candidate_count": int(decision["candidate_count"]),
                "subgraph_index": int(record["subgraph_index"]),
                "installation_kind": record["installation_kind"],
                "physical_graph_modified": bool(record["physical_graph_modified"]),
                "saved_region_nodes": [int(x) for x in decision.get("saved_region_nodes", [])],
                "removable_region_boundary": [
                    int(x) for x in decision.get("removable_region_boundary", [])
                ],
                "replacement_dag_structural_hash": litecut_summary["replacement_dag_structural_hash"],
                "feature_hash_19": litecut_summary["feature_hash_19"],
                "litecut_predicted_local_abc_delay": litecut_summary["predicted_raw_abc_delay"],
                "litecut_finite": litecut_summary["finite"],
                "litecut_deterministic": litecut_summary["deterministic"],
                "aig_path": str(aig_path),
                "aig_sha256": sha256(aig_path),
                "record_path": str(record_path),
                "cec_status": record["cec"]["status"],
                "cec_log": record["cec"]["log"],
                "provenance": record,
                "litecut_prediction_artifact": litecut_summary["output"],
                "litecut_full_artifact": str(output_dir / "litecut_predictions" / f"{record['candidate_id']}.json"),
                "litecut_full_result": litecut_full,
            }
            candidate_pool.append(candidate)
            candidates.append(candidate)
        if len(candidates) < 2:
            raise ValueError(f"decision {decision_id} has fewer than two replayed candidates")
        chosen_a, chosen_b, pair_type = choose_candidate_pair(candidates)
        decision_result = {
            "decision_id": decision_id,
            "design": decision["design"],
            "root": int(decision["root"]),
            "cut_slot": int(decision["cut_slot"]),
            "cut_leaves": [int(x) for x in decision["cut_leaves"]],
            "truth": int(decision["truth"]),
            "truth_class": int(decision["truth_class"]),
            "candidate_count": int(decision["candidate_count"]),
            "saved_region_nodes": [int(x) for x in decision.get("saved_region_nodes", [])],
            "removable_region_boundary": [
                int(x) for x in decision.get("removable_region_boundary", [])
            ],
            "candidate_pair_selection_rule": manifest["candidate_pair_selection_rule"],
            "candidate_pair_type": pair_type,
            "candidate_A": chosen_a,
            "candidate_B": chosen_b,
        }
        # Make pair-labelled copies explicit while retaining every candidate
        # replay under its subgraph index.
        pair_dir = output_dir / "replayed_aigs"
        pair_aig_a = pair_dir / f"{decision_id}_A.aig"
        pair_aig_b = pair_dir / f"{decision_id}_B.aig"
        shutil.copy2(chosen_a["aig_path"], pair_aig_a)
        shutil.copy2(chosen_b["aig_path"], pair_aig_b)
        chosen_a["selected_label"] = "A"
        chosen_b["selected_label"] = "B"
        chosen_a["selected_pair_aig"] = str(pair_aig_a)
        chosen_b["selected_pair_aig"] = str(pair_aig_b)

        # STA is intentionally delayed until the pair is fixed by installation
        # semantics and subgraph index; LiteCut magnitude is never consulted.
        sta_a = run_sta(
            f"{decision_id}_A", pair_aig_a, run_root, original, abc,
            timing_library,
            manifest["timing_classification"]["library_sha256"],
        )
        sta_b = run_sta(
            f"{decision_id}_B", pair_aig_b, run_root, original, abc,
            timing_library,
            manifest["timing_classification"]["library_sha256"],
        )
        # The execution copy of the timing library is supplied by the job; the
        # manifest path is rewritten by the Slurm wrapper to that data-local
        # copy before this runner starts.
        delay_a = float(sta_a["whole_design_delay_ps"])
        delay_b = float(sta_b["whole_design_delay_ps"])
        delta = delay_b - delay_a
        tolerance = float(manifest["timing_classification"]["timing_tolerance_ps"])
        local_a = float(chosen_a["litecut_predicted_local_abc_delay"])
        local_b = float(chosen_b["litecut_predicted_local_abc_delay"])
        local_delta = local_b - local_a
        local_pref = local_preference(local_a, local_b)
        whole_pref = timing_preference(delay_a, delay_b, tolerance)
        timing_class = classify_delta(delta, tolerance)
        same_direction: bool | None
        if timing_class != "TIMING_ACTIVE":
            same_direction = None
        elif local_pref == "TIE_OR_UNCERTAIN" or whole_pref == "TIE_OR_UNCERTAIN":
            same_direction = None
        else:
            same_direction = local_pref == whole_pref
        decision_result.update({
            "whole_design_timing": {
                "A": delay_a,
                "B": delay_b,
                "delta_B_minus_A": delta,
                "A_sta": sta_a,
                "B_sta": sta_b,
            },
            "litecut_local_preference": local_pref,
            "litecut_delta_B_minus_A": local_delta,
            "whole_design_single_decision_preference": whole_pref,
            "timing_class": timing_class,
            "same_direction": same_direction,
            "finite_and_deterministic": all(
                bool(candidate["litecut_finite"] and candidate["litecut_deterministic"])
                for candidate in candidates
            ),
            "cec": {
                "A": chosen_a["cec_status"],
                "B": chosen_b["cec_status"],
            },
            "candidate_pool_subgraphs": [
                {
                    key: candidate[key]
                    for key in (
                        "candidate_id", "subgraph_index", "installation_kind",
                        "replacement_dag_structural_hash", "feature_hash_19",
                        "litecut_predicted_local_abc_delay", "aig_sha256", "cec_status",
                    )
                }
                for candidate in candidates
            ],
        })
        decision_results.append(decision_result)
        # A compact per-decision file is useful without making the screening
        # CSV carry the full candidate provenance and inference payload.
        write_json(output_dir / "litecut_predictions" / f"{decision_id}.json", {
            "decision_id": decision_id,
            "candidates": [
                {
                    key: candidate[key]
                    for key in (
                        "candidate_id", "subgraph_index", "installation_kind",
                        "feature_hash_19", "replacement_dag_structural_hash",
                        "litecut_predicted_local_abc_delay", "litecut_finite",
                        "litecut_deterministic", "aig_sha256", "cec_status",
                    )
                }
                for candidate in candidates
            ],
            "selected_pair": {
                "type": pair_type,
                "A": chosen_a["subgraph_index"],
                "B": chosen_b["subgraph_index"],
            },
            "litecut_local_preference": local_pref,
            "litecut_delta_B_minus_A": local_delta,
        })
        print(json.dumps({
            "ordinal": ordinal,
            "decision_id": decision_id,
            "candidate_count": len(candidates),
            "pair_type": pair_type,
            "timing_class": timing_class,
            "D_A": delay_a,
            "D_B": delay_b,
            "delta": delta,
        }, ensure_ascii=False), flush=True)

    original_sta = run_sta(
        "original", original, run_root, original, abc,
        timing_library,
        manifest["timing_classification"]["library_sha256"],
    )
    timing_active = [d for d in decision_results if d["timing_class"] == "TIMING_ACTIVE"]
    timing_inactive = [d for d in decision_results if d["timing_class"] == "TIMING_INACTIVE"]
    borderline = [d for d in decision_results if d["timing_class"] == "BORDERLINE"]
    write_json(paths["candidate_pool"], {
        "gate": "Gate 2C",
        "sampling_manifest": str(manifest_copy),
        "decision_count": len(selected),
        "candidate_count": len(candidate_pool),
        "decisions": [
            {
                "decision_id": d["decision_id"],
                "design": d["design"],
                "root": d["root"],
                "cut_slot": d["cut_slot"],
                "cut_leaves": d["cut_leaves"],
                "candidate_count": d["candidate_count"],
                "saved_region_nodes": d["saved_region_nodes"],
                "removable_region_boundary": d["removable_region_boundary"],
                "candidates": d["candidate_pool_subgraphs"],
                "candidate_pair_type": d["candidate_pair_type"],
                "candidate_A": d["candidate_A"]["subgraph_index"],
                "candidate_B": d["candidate_B"]["subgraph_index"],
            }
            for d in decision_results
        ],
    })
    write_json(paths["timing_active"], timing_active)
    write_json(paths["timing_inactive"], timing_inactive + borderline)
    next_pair = validate_joint_replay(
        search_next_gate_pair(decision_results),
        decision_results,
        run_root,
        original,
        engine,
        abc,
    )
    write_json(paths["next_gate_pair"], next_pair)
    write_screening_csv(paths["screening_csv"], decision_results)
    summary_status = "PASS" if len(decision_results) == len(selected) else "BLOCKED"
    summary = build_summary(
        manifest, candidate_pool, decision_results, original_sta, next_pair,
        run_root, paths, summary_status,
    )
    write_json(paths["summary"], summary)

    active_for_direction = [
        decision for decision in decision_results
        if decision["timing_class"] == "TIMING_ACTIVE"
    ]
    same = sum(decision["same_direction"] is True for decision in active_for_direction)
    opposite = sum(decision["same_direction"] is False for decision in active_for_direction)
    tie = sum(decision["same_direction"] is None for decision in active_for_direction)

    report_lines = [
        "# Gate 2C — Timing-Active Rewrite Decision Screening",
        "",
        f"Status: **{summary_status}**",
        "",
        "The sample and candidate-pair rule were frozen before replay timing was run.",
        "This run uses the Gate 1 engine for candidate provenance, the existing Gate 2A mapping-free LiteCut bridge, and the exact Gate 2B `map; topo; stime` flow.",
        "",
        "## Frozen sampling",
        "",
        f"- Captured sampling frame: **{manifest['candidate_enumeration']['frame_size']}** Gate 1B records; engine `records_seen` was **{manifest['candidate_enumeration']['engine_records_seen']}**, but the uncaptured tail was not sampled.",
        f"- Sample rule: `{manifest['sampling_rule']['type']}`, sort by `(root, cut_slot)` ascending, filter `candidate_count >= 2`, take first **{len(selected)}**.",
        f"- Selected candidates replayed: **{len(candidate_pool)}** across **{len(decision_results)}** decisions.",
        "- No STA or LiteCut result was used to select a decision or candidate pair.",
        "",
        "## Results",
        "",
        f"- TIMING_ACTIVE: **{len(timing_active)}**",
        f"- TIMING_INACTIVE: **{len(timing_inactive)}**",
        f"- BORDERLINE: **{len(borderline)}**",
        f"- TIMING_ACTIVE fraction among valid decisions: **{len(timing_active) / len(decision_results) if decision_results else 0:.6f}**",
        f"- Fixed tolerance: **{manifest['timing_classification']['timing_tolerance_ps']} ps**; reported precision: **{manifest['timing_classification']['reported_precision_ps']} ps**.",
        "",
        "| pair type | decisions | active | inactive | borderline |",
        "|---|---:|---:|---:|---:|",
    ]
    for pair_type, stats in build_summary(
        manifest, candidate_pool, decision_results, original_sta, next_pair,
        run_root, paths, summary_status,
    )["pair_type_statistics"].items():
        report_lines.append(
            f"| {pair_type} | {stats['sampled_decisions']} | {stats['timing_active_count']} | {stats['timing_inactive_count']} | {stats['borderline_count']} |"
        )
    report_lines += [
        "",
        "## LiteCut comparison",
        "",
        f"- LiteCut local score difference but timing-inactive (strictly below tolerance): **{summary['local_score_difference_but_delay_unchanged']['timing_inactive_count']}**.",
        f"- LiteCut local score difference but whole-design delta within tolerance including borderline: **{summary['local_score_difference_but_delay_unchanged']['within_timing_tolerance_including_borderline_count']}**.",
        f"- Among TIMING_ACTIVE decisions, same direction: **{same}**, opposite direction: **{opposite}**, tie/uncertain: **{tie}**.",
        "These are single-decision perturbation direction comparisons, not LiteCut global accuracy.",
        "",
        "## Next Gate pair",
        "",
        f"- Pair search status: **{next_pair['status']}**.",
    ]
    if next_pair["status"] == "FOUND":
        report_lines.append(
            f"- Selected compatible timing-active decisions: `{', '.join(next_pair['selected_pair']['decision_ids'])}`."
        )
        report_lines.append(
            "- Compatibility was established by all four no-STA joint replays and CEC checks; no joint timing or interaction experiment was run in Gate 2C."
        )
    else:
        report_lines.append("- No compatible timing-active pair was found; this does not block the screening result itself.")
    report_lines += [
        "",
        "## Required answers",
        "",
        f"1. Q1: **{len(timing_active)}** of **{len(decision_results)}** valid sampled decisions are TIMING_ACTIVE.",
        f"2. Q2: TIMING_ACTIVE fraction is **{len(timing_active) / len(decision_results) if decision_results else 0:.6f}**.",
        f"3. Q3: **{summary['local_score_difference_but_delay_unchanged']['timing_inactive_count']}** have a nonzero selected-pair LiteCut local delta while whole-design timing is TIMING_INACTIVE; **{summary['local_score_difference_but_delay_unchanged']['within_timing_tolerance_including_borderline_count']}** are within tolerance including borderline.",
        f"4. Q4: Among TIMING_ACTIVE decisions, LiteCut and whole-design direction agree **{same}**, disagree **{opposite}**, and are tie/uncertain **{tie}** times.",
        f"5. Q5: A compatible pair for the next Gate was **{'found' if next_pair['status'] == 'FOUND' else 'not found'}**.",
        "",
        "## Artifacts",
        "",
        f"- Summary: `{paths['summary']}`",
        f"- Sampling manifest: `{paths['manifest']}`",
        f"- Candidate pool: `{paths['candidate_pool']}`",
        f"- Screening CSV: `{paths['screening_csv']}`",
        f"- Active decisions: `{paths['timing_active']}`",
        f"- Next-Gate pair candidates: `{paths['next_gate_pair']}`",
    ]
    (paths["report"]).write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": summary_status,
        "sampled_decisions": len(selected),
        "valid_decisions": len(decision_results),
        "timing_active": len(timing_active),
        "timing_inactive": len(timing_inactive),
        "borderline": len(borderline),
        "next_gate_pair": next_pair["status"],
        "summary": str(paths["summary"]),
    }, ensure_ascii=False, indent=2))
    return 0 if summary_status == "PASS" else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--rewrite-library", type=Path, required=True)
    parser.add_argument("--timing-library", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gate1-engine", type=Path, required=True)
    parser.add_argument("--abc", type=Path, required=True)
    parser.add_argument("--litecut-repo", type=Path, required=True)
    parser.add_argument(
        "--decision-limit", type=int, default=None,
        help="validation-only prefix limit; the full Gate requires the manifest sample size",
    )
    args = parser.parse_args()
    if args.decision_limit is not None and args.decision_limit <= 0:
        parser.error("--decision-limit must be positive")
    try:
        return run(args)
    except Exception as error:
        run_root = args.run_root.resolve()
        output_dir = run_root / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        failure = {
            "gate": "Gate 2C",
            "status": "BLOCKED",
            "error": f"{type(error).__name__}: {error}",
            "run_root": str(run_root),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        }
        write_json(output_dir / "gate2c_summary.json", failure)
        print(json.dumps(failure, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
