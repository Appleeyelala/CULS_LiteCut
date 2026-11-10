#!/usr/bin/env python3
"""Freeze the Gate 2C decision sample before any timing result is observed.

The source audit is produced by the Gate 1 rewrite engine.  Gate 1B keeps the
first 512 candidate-bearing records in its JSON audit, while also reporting the
larger number of records seen by the engine.  Gate 2C deliberately defines its
sampling frame as those captured records, so the frame is finite and auditable
and is not confused with the uncaptured tail.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_AUDIT = Path(
    "/data/wushangge/gate1a_AIGRewrite/gate1c_out/c1355_probe.json"
)
DEFAULT_ORIGINAL = Path(
    "/data/wushangge/gate1a_AIGRewrite/gate1c_r_out/normalized_original.aig"
)
DEFAULT_REWRITE_LIBRARY = Path(
    "/data/wushangge/gate1a_AIGRewrite/src/algorithms/rewrite_library.inc"
)
DEFAULT_TIMING_LIBRARY = Path("/home/wushangge/cutLibOrigin/lib/asap7.lib")
DEFAULT_CHECKPOINT = Path("/home/wushangge/cutLibOrigin/best.pt")
DEFAULT_ENGINE = Path(
    "/data/wushangge/gate1a_AIGRewrite/build_gate1c/abcg"
)
DEFAULT_ABC = Path("/data/public/AIGRewrite/abc/abc")
DEFAULT_OUTPUT = Path("gate2c_out/sampling_manifest.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def decision_id(record: dict[str, Any], ordinal: int) -> str:
    return f"c1355_r{int(record['root']):06d}_c{int(record['cut_slot']):03d}_n{ordinal:03d}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--original", type=Path, default=DEFAULT_ORIGINAL)
    parser.add_argument("--rewrite-library", type=Path, default=DEFAULT_REWRITE_LIBRARY)
    parser.add_argument("--timing-library", type=Path, default=DEFAULT_TIMING_LIBRARY)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--gate1-engine", type=Path, default=DEFAULT_ENGINE)
    parser.add_argument("--abc", type=Path, default=DEFAULT_ABC)
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    for path, label in (
        (args.audit, "Gate 1B audit"),
        (args.original, "normalized original snapshot"),
        (args.rewrite_library, "rewrite library"),
        (args.timing_library, "timing library"),
        (args.checkpoint, "LiteCut checkpoint"),
        (args.gate1_engine, "Gate 1 engine"),
        (args.abc, "ABC binary"),
    ):
        require_file(path, label)
    if args.sample_size <= 0:
        raise ValueError("--sample-size must be positive")

    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    records = audit.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("Gate 1B audit has no records")

    # The audit itself is the frozen frame.  Do not infer a larger population
    # from records_seen: the engine's 512-record cap means those extra records
    # are not available for deterministic replay in this Gate.
    frame = [
        record
        for record in records
        if int(record.get("number_of_available_subgraphs", 0)) >= 2
    ]
    frame.sort(key=lambda record: (int(record["root"]), int(record["cut_slot"])))
    if len(frame) < args.sample_size:
        raise ValueError(
            f"Only {len(frame)} captured decisions meet candidate_count >= 2; "
            f"cannot select {args.sample_size}"
        )

    selected = frame[: args.sample_size]
    selected_ids: list[str] = []
    selected_records: list[dict[str, Any]] = []
    seen_ids: set[tuple[int, int]] = set()
    for ordinal, source_record in enumerate(selected):
        root_cut = (int(source_record["root"]), int(source_record["cut_slot"]))
        if root_cut in seen_ids:
            raise ValueError(f"duplicate decision in frozen frame: {root_cut}")
        seen_ids.add(root_cut)
        record = dict(source_record)
        record["design"] = "c1355.aig"
        record["decision_id"] = decision_id(record, ordinal)
        record["candidate_count"] = int(record["number_of_available_subgraphs"])
        record["source_audit_record_ordinal_after_sort"] = ordinal
        selected_ids.append(record["decision_id"])
        selected_records.append(record)

    manifest = {
        "gate": "Gate 2C",
        "status": "FROZEN_BEFORE_TIMING",
        "purpose": "Timing-active rewrite decision screening",
        "design": "c1355.aig",
        "original_snapshot": {
            "source_path": str(args.original),
            "sha256": sha256(args.original),
            "normalization": "Gate 1C-R normalized original snapshot",
        },
        "candidate_enumeration": {
            "engine": str(args.gate1_engine),
            "engine_sha256": sha256(args.gate1_engine),
            "source_audit": str(args.audit),
            "source_audit_sha256": sha256(args.audit),
            "captured_audit_records": len(records),
            "engine_records_seen": audit.get("records_seen"),
            "audit_record_cap_note": (
                "Gate 1B engine audit stores at most 512 records; the 512 "
                "captured records, not records_seen, define this frame."
            ),
            "frame_filter": "number_of_available_subgraphs >= 2",
            "frame_size": len(frame),
        },
        "sampling_rule": {
            "type": "deterministic_sorted_prefix",
            "filter": "candidate_count >= 2",
            "sort_keys": ["root ascending", "cut_slot ascending"],
            "take_first": args.sample_size,
            "seed": None,
            "random_sampling_used": False,
            "timing_results_observed_before_selection": False,
        },
        "sampled_decisions": len(selected_records),
        "selected_decision_ids": selected_ids,
        "candidate_pair_selection_rule": {
            "predefined": True,
            "rule": [
                "After candidate installation kinds are enumerated, if at least "
                "two PHYSICAL_REPLACEMENT candidates exist, choose the two "
                "smallest subgraph_index values among them.",
                "Otherwise choose the smallest subgraph_index "
                "NO_OP_FULL_CORRESPONDENCE candidate and the smallest "
                "subgraph_index PHYSICAL_REPLACEMENT candidate.",
                "For physical-vs-physical, label the lower subgraph_index A "
                "and the higher B; for no-op-vs-physical, label NO_OP as A "
                "and PHYSICAL_REPLACEMENT as B.",
                "LiteCut scores, STA, and structural hash differences are not "
                "used for pair selection.",
            ],
        },
        "timing_classification": {
            "flow": 'read "<technology_library>"; read "<input_aig>"; map; topo; stime',
            "abc_binary": str(args.abc),
            "library": str(args.timing_library),
            "library_sha256": sha256(args.timing_library),
            "reported_precision_ps": 0.01,
            "timing_tolerance_ps": 0.01,
            "borderline_rule": "abs(delta) == tolerance within 1e-12 ps",
            "active_rule": "abs(delta) > tolerance + 1e-12 ps",
            "inactive_rule": "abs(delta) < tolerance - 1e-12 ps",
        },
        "litecut": {
            "bridge": "gate2a_candidate_inference.py",
            "rewrite_library": str(args.rewrite_library),
            "rewrite_library_sha256": sha256(args.rewrite_library),
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": sha256(args.checkpoint),
            "model_family": "student_group_fusion_soft_moe",
            "input_dimension": 19,
            "prediction_semantics": "LiteCut-predicted local abc_delay",
            "local_score_difference_tolerance": 1e-12,
        },
        "replay_requirements": {
            "same_original_snapshot_for_A_and_B": True,
            "other_decisions_changed": False,
            "cec_required": True,
            "physical_candidate_semantics_source": "Gate 1 engine provenance",
        },
        "selected_records": selected_records,
        "provenance": {
            "prepare_script": "gate2c_prepare.py",
            "gate1_engine_source_commit": "1870e7218d323774a2b4e1fe8d6bbea05b33ed5a",
            "gate1_engine_source_tree_dirty": True,
            "litecut_repo_commit": "077aa1e49b8b97ed6d76013a7272c06bfacabf23",
            "candidate_semantics_unchanged": True,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "frame_size": len(frame),
        "sampled_decisions": len(selected_records),
        "candidate_total": sum(
            int(record["candidate_count"]) for record in selected_records
        ),
        "selected_decision_ids": selected_ids,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
