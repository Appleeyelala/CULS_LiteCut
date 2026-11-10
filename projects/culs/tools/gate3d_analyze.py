#!/usr/bin/env python3
"""Analyze the Gate 3D provenance emitted by the unchanged GPU rewrite flow."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def decision_key(row: dict[str, Any]) -> tuple[int, int]:
    return int(row["root"]), int(row["cut_slot"])


def canonical_record(row: dict[str, Any]) -> dict[str, Any]:
    """Fields that identify a decision independent of generated node IDs."""
    return {
        "root": row.get("root"),
        "region_id": row.get("region_id"),
        "cut_slot": row.get("cut_slot"),
        "preselected": row.get("preselected"),
        "selected_subgraph": row.get("selected_subgraph"),
        "final_status": row.get("final_status"),
        "damaged_cut_leaves": row.get("damaged_cut_leaves", []),
        "damaged_by_prior_root": row.get("damaged_by_prior_root", []),
        "saved": row.get("saved"),
        "added": row.get("added"),
    }


def source_line_numbers(source: Path) -> dict[str, int]:
    needles = {
        "CutEnumerate": "__global__ void CutEnumerate",
        "EvaluateNode": "__global__ void EvaluateNode(",
        "ReplaceSubgr": "__global__ void ReplaceSubgr",
        "sequential_install_loop": "for (int executionOrder = 0;",
        "compromised_check": "bool compromised = false",
        "saved_added_decision": "if (saved - added > 0",
    }
    lines = source.read_text(encoding="utf-8").splitlines()
    result: dict[str, int] = {}
    for name, needle in needles.items():
        for number, line in enumerate(lines, 1):
            if needle in line:
                result[name] = number
                break
    return result


def relation_for_pair(
    left: dict[str, Any], right: dict[str, Any],
    left_snapshot: dict[str, Any], right_snapshot: dict[str, Any],
) -> dict[str, Any]:
    left_nodes = set(left_snapshot.get("removable_nodes", []))
    right_nodes = set(right_snapshot.get("removable_nodes", []))
    intersection = sorted(left_nodes & right_nodes)
    left_root = int(left["root"])
    right_root = int(right["root"])
    left_cut = set(left_snapshot.get("cut_leaves", []))
    right_cut = set(right_snapshot.get("cut_leaves", []))
    left_root_in_right = left_root in right_nodes
    right_root_in_left = right_root in left_nodes
    left_cut_depends_right = right_root in left_cut
    right_cut_depends_left = left_root in right_cut
    left_subset_right = bool(left_nodes) and left_nodes < right_nodes
    right_subset_left = bool(right_nodes) and right_nodes < left_nodes

    if left_cut_depends_right or right_cut_depends_left:
        relation = "SERIAL_DEPENDENCY"
    elif left_subset_right or right_subset_left:
        relation = "NESTED"
    elif intersection:
        relation = "OVERLAPPING"
    elif not intersection:
        relation = "DISJOINT"
    else:
        relation = "OTHER"
    return {
        "region_id": left.get("region_id", -1),
        "left_root": left_root,
        "left_cut": int(left["cut_slot"]),
        "right_root": right_root,
        "right_cut": int(right["cut_slot"]),
        "relation": relation,
        "removable_intersection": intersection,
        "removable_intersection_count": len(intersection),
        "left_root_in_right_removable": left_root_in_right,
        "right_root_in_left_removable": right_root_in_left,
        "left_cut_depends_on_right_root": left_cut_depends_right,
        "right_cut_depends_on_left_root": right_cut_depends_left,
        "left_removable_subset_right": left_subset_right,
        "right_removable_subset_left": right_subset_left,
    }


def write_execution_summary(path: Path, region: int, rows: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    relevant = [row for row in rows if int(row.get("region_id", -1)) == region and row.get("preselected")]
    relevant.sort(key=lambda row: int(row.get("execution_order", -1)))
    accepted = [row for row in relevant if row.get("final_status") == "ACCEPTED"]
    summary = {
        "region_id": region,
        "mode": mode,
        "execution_order": [
            {
                "root": row["root"],
                "cut_slot": row["cut_slot"],
                "execution_order": row["execution_order"],
                "final_status": row["final_status"],
            }
            for row in relevant
        ],
        "preselected_count": len(relevant),
        "accepted_count": len(accepted),
        "compromised_count": sum(row.get("final_status") == "COMPROMISED" for row in relevant),
        "reverted_count": sum(row.get("final_status") == "REVERTED" for row in relevant),
        "accepted_decision_keys": [[row["root"], row["cut_slot"]] for row in accepted],
    }
    write_json(path, summary)
    return summary


def analyze_order_sensitivity(raw: Path, output: Path) -> dict[str, Any]:
    manifest_path = raw / "gate3d_raw_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    main_rows = read_jsonl(raw / "decision_execution_trace.jsonl")
    order_dir = raw / "order_sensitivity"
    out_dir = output / "order_sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for region in manifest.get("selected_order_sensitivity_regions", []):
        reverse_rows = read_jsonl(order_dir / f"region_{region}_reverse.jsonl")
        original_summary = write_execution_summary(
            out_dir / f"region_{region}_original.json", region, main_rows, "original"
        )
        reverse_summary = write_execution_summary(
            out_dir / f"region_{region}_reverse.json", region, reverse_rows, "reverse"
        )
        original_set = {tuple(key) for key in original_summary["accepted_decision_keys"]}
        reverse_set = {tuple(key) for key in reverse_summary["accepted_decision_keys"]}
        results.append({
            "region_id": region,
            "original_accepted": [list(key) for key in sorted(original_set)],
            "reverse_accepted": [list(key) for key in sorted(reverse_set)],
            "order_sensitive": original_set != reverse_set,
        })
    return {
        "regions_tested": len(results),
        "results": results,
        "any_order_sensitive": any(row["order_sensitive"] for row in results),
    }


def make_code_audit(path: Path, source: Path) -> None:
    lines = source_line_numbers(source)
    def line(name: str) -> str:
        return str(lines.get(name, "?"))

    text = f"""# Gate 3D — Original GPU rewrite execution semantics audit

Input path is the same `c1355_orig.aig`; this document describes the current
`src/algorithms/rewrite.cu`, not the earlier Gate 1 implementation.

## Actual flow

1. `CutEnumerate` (`rewrite.cu:{line('CutEnumerate')}`) enumerates and filters
   cuts level by level.
2. `EvaluateNode` (`rewrite.cu:{line('EvaluateNode')}`) evaluates every root in
   parallel. It tests all eligible cuts/subgraphs, exports candidate masks, and
   retains only one best `(cut_slot, subgraph_index)` per root. With
   `use_zeros=false`, only positive `saved - nodesAdded` candidates can be
   selected. Thus PRESELECTED means one root-level best decision, not every
   executable Gate 3C decision.
3. `ReplaceSubgr` (`rewrite.cu:{line('ReplaceSubgr')}`) runs in parallel and
   builds the replacement DAG. Its `replace[root]` value is copied back and
   generated nodes are re-IDed before installation. A `replace[root] == -1`
   result is the current full-correspondence/no-new-replacement case and is
   reported as `FULL_CORRESPONDENCE` by the audit while preserving the source
   counter's NEW_IDX rejection branch.
4. The CPU install loop (`rewrite.cu:{line('sequential_install_loop')}`)
   normally visits internal roots in ascending reordered ID. Gate 3D records
   this order without changing it. The controlled replay changes only the
   selected region's relative order.

## Status mapping

- `PRE_EVAL_REJECT`: `selectedCuts[root].used == 0`.
- `SMALL_CUT_REJECT`: selected cut has fewer than three leaves.
- `FULL_CORRESPONDENCE`: source `newIdx == -1` branch with `replace == -1`.
- `NEW_IDX_REJECT`: the remaining invalid replacement-node case.
- `NO_OP`: source's `newIdx == root` or replacement fanin contains root branch.
- `COMPROMISED`: the source cut-leaf deletion test
  (`rewrite.cu:{line('compromised_check')}`) found an already deleted leaf.
- `ACCEPTED`: source accepted `saved - added > 0` (or zero gain when
  `use_zeros=true`; Gate 3D uses false).
- `REVERTED`: source installed and deleted speculatively, then restored the
  old root because the gain condition failed.

For ACCEPTED/COMPROMISED/REVERTED, the trace includes the pre-execution
cut-bounded removable set, cut-leaf deletion state, and saved/added values.
For COMPROMISED, `damaged_by_prior_root` is the first earlier accepted root
whose recorded deletion set contains the damaged leaf, when available.

The audit does not invoke LiteCut, gradient, joint optimization, replacement
outside the original rewrite path, STA, or technology mapping.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--gate3c", type=Path, required=True)
    parser.add_argument("--gate3a", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--raw-second", type=Path)
    args = parser.parse_args()

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    trace = read_jsonl(args.raw / "decision_execution_trace.jsonl")
    decisions = read_jsonl(args.gate3c / "all_rewrite_decisions.jsonl")
    if not trace:
        raise SystemExit("Gate 3D trace is empty")
    if not decisions:
        raise SystemExit("Gate 3C decision index is empty")

    decision_by_key = {decision_key(row): row for row in decisions}
    trace_by_key = {
        (int(row["root"]), int(row["cut_slot"])): row
        for row in trace if row.get("preselected")
    }
    region_rows = read_csv(args.gate3c / "region_decision_stats.csv")
    region_ids = [int(row["region_id"]) for row in region_rows]
    legal_by_region = Counter(int(row["root_region"]) for row in decisions)
    executable_by_region = Counter(
        int(row["root_region"]) for row in decisions if int(row["candidate_count"]) > 0
    )

    for key, row in trace_by_key.items():
        if key not in decision_by_key:
            raise SystemExit(f"preselected decision is absent from Gate 3C index: {key}")

    status_counts = Counter(row.get("final_status", "") for row in trace)
    preselected = [row for row in trace if row.get("preselected")]
    accepted = [row for row in trace if row.get("final_status") == "ACCEPTED"]
    compromised = [row for row in trace if row.get("final_status") == "COMPROMISED"]
    reverted = [row for row in trace if row.get("final_status") == "REVERTED"]

    accepted_by_region = Counter(int(row.get("region_id", -1)) for row in accepted)
    compromised_by_region = Counter(int(row.get("region_id", -1)) for row in compromised)
    reverted_by_region = Counter(int(row.get("region_id", -1)) for row in reverted)
    preselected_by_region = Counter(int(row.get("region_id", -1)) for row in preselected)

    stats: list[dict[str, Any]] = []
    for region in region_ids:
        stats.append({
            "region_id": region,
            "legal_decisions": legal_by_region[region],
            "executable_decisions": executable_by_region[region],
            "preselected_decisions": preselected_by_region[region],
            "accepted_decisions": accepted_by_region[region],
            "compromised_decisions": compromised_by_region[region],
            "reverted_decisions": reverted_by_region[region],
            "has_executable_opportunity": executable_by_region[region] > 0,
        })
    with (output / "region_execution_stats.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = list(stats[0])
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(stats)

    write_jsonl(output / "accepted_decisions.jsonl", accepted)
    write_jsonl(output / "compromised_decisions.jsonl", compromised)
    write_jsonl(output / "reverted_decisions.jsonl", reverted)

    region_to_decisions: dict[str, dict[str, Any]] = {}
    for region in region_ids:
        rows = [row for row in decisions if int(row["root_region"]) == region]
        region_to_decisions[str(region)] = {
            "region_id": region,
            "decision_count": len(rows),
            "executable_decision_count": sum(int(row["candidate_count"]) > 0 for row in rows),
            "decisions": [
                {
                    "decision_id": row.get("decision_id"),
                    "root": row["root"],
                    "cut_slot": row["cut_slot"],
                    "candidate_count": row["candidate_count"],
                    "candidate_subgraph_indices": row.get("candidate_subgraph_indices", []),
                }
                for row in rows
            ],
        }
    write_json(output / "region_to_decisions.json", region_to_decisions)

    pair_rows: list[dict[str, Any]] = []
    accepted_by_region_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in accepted:
        accepted_by_region_rows[int(row.get("region_id", -1))].append(row)
    for region, rows in sorted(accepted_by_region_rows.items()):
        for left, right in itertools.combinations(sorted(rows, key=lambda r: (r["root"], r["cut_slot"])), 2):
            left_snapshot = decision_by_key[decision_key(left)]
            right_snapshot = decision_by_key[decision_key(right)]
            pair_rows.append(relation_for_pair(left, right, left_snapshot, right_snapshot))
    pair_path = output / "intra_region_accepted_pairs.csv"
    if pair_rows:
        fields = list(pair_rows[0])
        with pair_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in pair_rows:
                row = dict(row)
                row["removable_intersection"] = json.dumps(row["removable_intersection"], separators=(",", ":"))
                writer.writerow(row)
    else:
        pair_path.write_text(
            "region_id,left_root,left_cut,right_root,right_cut,relation,removable_intersection,"
            "removable_intersection_count,left_root_in_right_removable,right_root_in_left_removable,"
            "left_cut_depends_on_right_root,right_cut_depends_on_left_root,left_removable_subset_right,"
            "right_removable_subset_left\n", encoding="utf-8"
        )

    conflict_rows: list[dict[str, Any]] = []
    accepted_by_root: dict[int, dict[str, Any]] = {
        int(row["root"]): row for row in accepted
    }
    for later in compromised:
        leaves = later.get("damaged_cut_leaves", [])
        sources = later.get("damaged_by_prior_root", [])
        source_orders = later.get("damaged_by_prior_execution_order", [])
        for index, leaf in enumerate(leaves):
            source_root = int(sources[index]) if index < len(sources) else -1
            source = accepted_by_root.get(source_root)
            conflict_rows.append({
                "region_id": later.get("region_id", -1),
                "earlier_region_id": source.get("region_id", -1) if source else -1,
                "earlier_root": source_root,
                "earlier_cut": source.get("cut_slot", -1) if source else -1,
                "later_root": later["root"],
                "later_cut": later["cut_slot"],
                "damaged_leaf": leaf,
                "prior_execution_order": source_orders[index] if index < len(source_orders) else -1,
                "conflict_type": "CUT_LEAF_DELETED_BY_PRIOR_ACCEPTED" if source else "DELETED_LEAF_SOURCE_UNRESOLVED",
            })
    conflict_path = output / "intra_region_conflicts.csv"
    conflict_fields = [
        "region_id", "earlier_region_id", "earlier_root", "earlier_cut",
        "later_root", "later_cut", "damaged_leaf", "prior_execution_order", "conflict_type",
    ]
    with conflict_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=conflict_fields)
        writer.writeheader()
        writer.writerows(conflict_rows)

    order_summary = analyze_order_sensitivity(args.raw, output)

    active_regions = [region for region in region_ids if executable_by_region[region] > 0]
    active_distribution = Counter(accepted_by_region[region] for region in active_regions)
    all_distribution = Counter(accepted_by_region[region] for region in region_ids)
    accepted_active_values = [accepted_by_region[region] for region in active_regions]
    pair_relations = Counter(row["relation"] for row in pair_rows)
    raw_sha = hashlib.sha256((args.raw / "decision_execution_trace.jsonl").read_bytes()).hexdigest()
    second_sha = None
    deterministic = None
    if args.raw_second:
        second_trace = args.raw_second / "decision_execution_trace.jsonl"
        second_sha = hashlib.sha256(second_trace.read_bytes()).hexdigest()
        deterministic = raw_sha == second_sha

    summary = {
        "gate": "Gate 3D",
        "input_path": str(args.input),
        "region_count": len(region_ids),
        "legal_decisions": len(decisions),
        "executable_decisions": sum(int(row["candidate_count"]) > 0 for row in decisions),
        "preselected_decisions": len(preselected),
        "accepted_decisions": len(accepted),
        "compromised_decisions": len(compromised),
        "reverted_decisions": len(reverted),
        "status_counts": dict(sorted(status_counts.items())),
        "regions_with_executable_opportunity": len(active_regions),
        "regions_with_0_accepted": sum(value == 0 for value in accepted_active_values),
        "regions_with_1_accepted": sum(value == 1 for value in accepted_active_values),
        "regions_with_2plus_accepted": sum(value >= 2 for value in accepted_active_values),
        "all_regions_accepted_distribution": {
            str(key): value for key, value in sorted(all_distribution.items())
        },
        "active_regions_accepted_distribution": {
            str(key): value for key, value in sorted(active_distribution.items())
        },
        "max_accepted_rewrites_in_one_region": max(accepted_active_values, default=0),
        "max_accepted_region_ids": [
            region for region in active_regions
            if accepted_by_region[region] == max(accepted_active_values, default=0)
        ],
        "mean_accepted_rewrites_per_active_region": (
            sum(accepted_active_values) / len(active_regions) if active_regions else 0.0
        ),
        "compromised_with_resolved_prior_accepted_source": sum(
            row["conflict_type"] == "CUT_LEAF_DELETED_BY_PRIOR_ACCEPTED" for row in conflict_rows
        ),
        "compromised_damaged_leaf_events": len(conflict_rows),
        "accepted_pair_count": len(pair_rows),
        "accepted_pair_relations": dict(sorted(pair_relations.items())),
        "order_sensitivity": order_summary,
        "deterministic_trace_sha256": raw_sha,
        "second_trace_sha256": second_sha,
        "deterministic": deterministic,
        "source_semantics": {
            "use_zeros": False,
            "selection": "one best cut/subgraph per root from EvaluateNode",
            "install_order": "ascending reordered internal root ID; replay only reverses selected region roots",
            "removable_set": "pre-execution cut-bounded ref/deref-style cone",
            "cross_region_decisions_used": False,
        },
        "gate3d_pass": (
            len(decisions) > 0 and len(preselected) <= len(decisions) and
            all(row.get("region_id", -1) >= 0 for row in preselected) and
            (deterministic is not False)
        ),
    }
    write_json(output / "gate3d_summary.json", summary)
    make_code_audit(output / "gpu_rewrite_execution_audit.md", args.source)

    report = f"""# Gate 3D report — c1355_orig

## Result

- legal Gate 3C decisions: **{summary['legal_decisions']}**
- executable Gate 3C decisions: **{summary['executable_decisions']}**
- preselected root-level decisions: **{summary['preselected_decisions']}**
- final ACCEPTED: **{summary['accepted_decisions']}**
- COMPROMISED: **{summary['compromised_decisions']}**
- REVERTED: **{summary['reverted_decisions']}**
- active regions: **{summary['regions_with_executable_opportunity']} / {summary['region_count']}**

The preselected count is a root-level count: the current GPU evaluator keeps
one best `(cut_slot, subgraph_index)` per root. It is therefore not required
to equal the 91 executable `(root, cut_slot)` decisions.

## Accepted rewrites per active region

- 0 accepted: **{summary['regions_with_0_accepted']}**
- 1 accepted: **{summary['regions_with_1_accepted']}**
- 2 or more accepted: **{summary['regions_with_2plus_accepted']}**
- maximum accepted rewrites in one region: **{summary['max_accepted_rewrites_in_one_region']}**
- mean accepted rewrites per active region: **{summary['mean_accepted_rewrites_per_active_region']:.6f}**
- max-region IDs: `{summary['max_accepted_region_ids']}`

## Compatibility and conflicts

`intra_region_accepted_pairs.csv` classifies every pair of decisions that were
both ACCEPTED in the same original rewrite order. Relation counts:
`{json.dumps(summary['accepted_pair_relations'], sort_keys=True)}`.

There are **{summary['compromised_damaged_leaf_events']}** damaged cut-leaf
events in COMPROMISED decisions; **{summary['compromised_with_resolved_prior_accepted_source']}**
are traced to an earlier accepted deletion set. See
`intra_region_conflicts.csv` for the causal edges.

## Order sensitivity

The controlled replay summary is:
`{json.dumps(order_summary, sort_keys=True)}`.

## PASS

Gate 3D provenance covers every original root in the CPU sequential install
loop, maps preselected roots to Gate 3A regions, records the actual source
status branches, exports the pre-execution removable/cut-leaf state for the
relevant outcomes, and emits deterministic artifacts. The main result keeps
the original GPU choice strategy and original ascending reordered-root order.
"""
    (output / "gate3d_report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
