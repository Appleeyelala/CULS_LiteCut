#!/usr/bin/env python3
"""Gate 3C: align GPU rewrite decisions with the Gate 3A MFFC partition.

The CUDA exporter writes every eligible GPU rewrite cut, including cuts with
zero legal replacement candidates.  This script turns the positive-candidate
rows into Rewrite Decisions and classifies them using only their actual
cut-bounded removable node sets.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set


def dump_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def dump_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_owner_map(path: Path) -> Dict[int, int]:
    owners: Dict[int, int] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            node = int(row["node_id"])
            region = int(row["region_id"])
            if node in owners and owners[node] != region:
                raise RuntimeError(f"node {node} has multiple strict-MFFC owners")
            owners[node] = region
    return owners


def load_regions(path: Path) -> List[dict]:
    payload = load_json(path)
    if payload.get("partition") != "STRICT_MAXIMAL_MFFC_PARTITION":
        raise RuntimeError(f"unexpected partition in {path}")
    regions = sorted(payload["regions"], key=lambda row: int(row["region_id"]))
    expected = list(range(len(regions)))
    actual = [int(row["region_id"]) for row in regions]
    if actual != expected:
        raise RuntimeError(f"region IDs are not contiguous: {actual}")
    for row in regions:
        if row.get("semantics") != "STANDARD_MFFC":
            raise RuntimeError(f"region {row['region_id']} is not STANDARD_MFFC")
    return regions


def region_label(region_id: int) -> str:
    return f"R{region_id}"


def normalize_decision(raw: dict, decision_id: str, owners: Dict[int, int], region_ids: Set[int]) -> dict:
    candidate_indices = [int(value) for value in raw["candidate_subgraph_indices"]]
    candidate_count = int(raw["candidate_count"])
    if candidate_count != len(candidate_indices) or candidate_count < 0:
        raise RuntimeError(f"raw row has an invalid candidate list: {raw}")
    if candidate_indices != sorted(set(candidate_indices)):
        raise RuntimeError(f"candidate indices are not deterministic at {raw['root']}:{raw['cut_slot']}")

    removable = sorted(set(int(node) for node in raw["removable_nodes"]))
    if not removable:
        raise RuntimeError(f"decision {raw['root']}:{raw['cut_slot']} has no removable nodes")
    missing = [node for node in removable if node not in owners]
    if missing:
        raise RuntimeError(f"decision {raw['root']}:{raw['cut_slot']} has unowned nodes {missing}")
    owner_regions = sorted(set(owners[node] for node in removable))
    if any(region not in region_ids for region in owner_regions):
        raise RuntimeError(f"decision {raw['root']}:{raw['cut_slot']} has unknown owner region")

    root = int(raw["root"])
    if root not in owners:
        raise RuntimeError(f"decision root {root} is not covered by the Gate 3A owner map")
    classification = "INTRA_REGION" if len(owner_regions) == 1 else "CROSS_REGION"
    decision_owner = owner_regions[0] if classification == "INTRA_REGION" else None
    candidates = [{"subgraph_index": index} for index in candidate_indices]
    return {
        "decision_id": decision_id,
        "root": root,
        "cut_slot": int(raw["cut_slot"]),
        "cut_leaves": [int(node) for node in raw["cut_leaves"]],
        "truth": int(raw["truth"]),
        "truth_class": int(raw["truth_class"]),
        "candidate_count": candidate_count,
        "candidate_subgraph_indices": candidate_indices,
        "replacement_candidates": candidates,
        "removable_nodes": removable,
        "removable_node_count": len(removable),
        "saved_count": int(raw["saved_count"]),
        "root_region": owners[root],
        "owner_region_count": len(owner_regions),
        "owner_regions": owner_regions,
        "decision_owner_region": decision_owner,
        "classification": classification,
    }


def write_alignment_csv(path: Path, decisions: Sequence[dict]) -> None:
    fields = [
        "decision_id", "root", "cut_slot", "candidate_count",
        "removable_node_count", "root_region", "owner_region_count",
        "owner_regions", "classification", "decision_owner_region",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for decision in decisions:
            writer.writerow({
                "decision_id": decision["decision_id"],
                "root": decision["root"],
                "cut_slot": decision["cut_slot"],
                "candidate_count": decision["candidate_count"],
                "removable_node_count": decision["removable_node_count"],
                "root_region": decision["root_region"],
                "owner_region_count": decision["owner_region_count"],
                "owner_regions": ",".join(region_label(r) for r in decision["owner_regions"]),
                "classification": decision["classification"],
                "decision_owner_region": "" if decision["decision_owner_region"] is None
                else decision["decision_owner_region"],
            })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--maximal-mffcs", type=Path, required=True)
    parser.add_argument("--owner-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    raw_manifest = load_json(args.raw_dir / "gate3c_raw_manifest.json")
    raw_rows = load_jsonl(args.raw_dir / "gpu_rewrite_cuts.jsonl")
    regions = load_regions(args.maximal_mffcs)
    owners = load_owner_map(args.owner_map)
    region_ids = {int(row["region_id"]) for row in regions}

    raw_rows.sort(key=lambda row: (int(row["root"]), int(row["cut_slot"])))
    decisions = [
        normalize_decision(row, f"d{index}", owners, region_ids)
        for index, row in enumerate(raw_rows)
    ]

    executable = [row for row in decisions if row["candidate_count"] > 0]
    intra = [row for row in decisions if row["classification"] == "INTRA_REGION"]
    cross = [row for row in decisions if row["classification"] == "CROSS_REGION"]
    executable_intra = [row for row in executable if row["classification"] == "INTRA_REGION"]
    executable_cross = [row for row in executable if row["classification"] == "CROSS_REGION"]
    total_candidates = sum(row["candidate_count"] for row in decisions)

    dump_jsonl(args.output / "all_rewrite_decisions.jsonl", decisions)
    dump_jsonl(args.output / "executable_rewrite_decisions.jsonl", executable)
    dump_jsonl(args.output / "intra_region_decisions.jsonl", intra)
    dump_jsonl(args.output / "cross_region_decisions.jsonl", cross)
    write_alignment_csv(args.output / "decision_region_alignment.csv", decisions)

    indexed: Dict[int, List[dict]] = {region_id: [] for region_id in sorted(region_ids)}
    for decision in decisions:
        for region_id in decision["owner_regions"]:
            indexed[region_id].append({
                "decision_id": decision["decision_id"],
                "root": decision["root"],
                "cut_slot": decision["cut_slot"],
                "classification": decision["classification"],
                "candidate_count": decision["candidate_count"],
                "candidate_subgraph_indices": decision["candidate_subgraph_indices"],
            })
    region_to_decisions = {
        "partition": "STRICT_MAXIMAL_MFFC_PARTITION",
        "region_count": len(regions),
        "regions": [
            {
                "region_id": region_id,
                "decision_count": len(indexed[region_id]),
                "candidate_count": sum(row["candidate_count"] for row in indexed[region_id]),
                "decisions": indexed[region_id],
            }
            for region_id in sorted(region_ids)
        ],
    }
    dump_json(args.output / "region_to_decisions.json", region_to_decisions)

    stats_fields = [
        "region_id", "region_root", "decision_count", "candidate_count",
        "intra_decision_count", "cross_decision_count", "executable_decision_count",
        "has_rewrite_opportunity",
    ]
    with (args.output / "region_decision_stats.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=stats_fields)
        writer.writeheader()
        for region in regions:
            region_id = int(region["region_id"])
            rows = indexed[region_id]
            writer.writerow({
                "region_id": region_id,
                "region_root": int(region["root"]),
                "decision_count": len(rows),
                "candidate_count": sum(row["candidate_count"] for row in rows),
                "intra_decision_count": sum(row["classification"] == "INTRA_REGION" for row in rows),
                "cross_decision_count": sum(row["classification"] == "CROSS_REGION" for row in rows),
                "executable_decision_count": sum(row["candidate_count"] > 0 for row in rows),
                "has_rewrite_opportunity": any(row["candidate_count"] > 0 for row in rows),
            })

    regions_with_opportunity = [
        region_id for region_id, rows in indexed.items()
        if any(row["candidate_count"] > 0 for row in rows)
    ]
    regions_without_opportunity = [
        region_id for region_id, rows in indexed.items()
        if not any(row["candidate_count"] > 0 for row in rows)
    ]
    input_manifest = load_json(args.input_manifest) if args.input_manifest else {}
    summary = {
        "gate": "Gate 3C",
        "gate3c_pass": True,
        "partition": "STRICT_MAXIMAL_MFFC_PARTITION",
        "region_count": len(regions),
        "total_eligible_gpu_cuts": len(raw_rows),
        "zero_candidate_gpu_cuts": sum(int(row["candidate_count"]) == 0 for row in raw_rows),
        "total_decisions": len(decisions),
        "executable_decisions": len(executable),
        "intra_region_count": len(intra),
        "cross_region_count": len(cross),
        "executable_intra_region_count": len(executable_intra),
        "executable_cross_region_count": len(executable_cross),
        "total_replacement_candidates": total_candidates,
        "regions_with_rewrite_opportunity": regions_with_opportunity,
        "regions_without_rewrite_opportunity": regions_without_opportunity,
        "regions_with_rewrite_opportunity_count": len(regions_with_opportunity),
        "regions_without_rewrite_opportunity_count": len(regions_without_opportunity),
        "candidate_semantics": raw_manifest["candidate_rule"],
        "candidate_identity_source": "GPU rewrite EvaluateNode candidate mask export",
        "removable_set_semantics": "GPU rewrite CalcMFFC/Decrease ref-deref semantics",
        "id_space": raw_manifest["id_space"],
        "input_path": input_manifest.get("absolute_input_path", raw_manifest["input_path"]),
        "input_sha256": input_manifest.get("sha256"),
        "partition_source": str(args.maximal_mffcs),
        "owner_map_source": str(args.owner_map),
        "deterministic_order": "root ascending, cut_slot ascending, subgraph_index ascending",
        "deterministic": True,
        "forbidden_flows_executed": {
            "gradient": False,
            "joint_optimization": False,
            "litecut": False,
            "replacement": False,
        },
        "artifact_sha256": {},
    }
    dump_json(args.output / "gate3c_summary.json", summary)

    report = [
        "# Gate 3C report - c1355_orig",
        "",
        "## Scope",
        "",
        "The same c1355_orig AIG was used. The raw enumeration calls the existing GPU rewrite cut enumeration and candidate matching path with `use_zeros=false`; candidate identities come from the GPU `EvaluateNode` mask export, and no replacement is performed.",
        "",
        "Terminology is fixed as follows:",
        "",
        "- MFFC Region: a Gate 3A strict maximal standard MFFC.",
        "- Rewrite Decision: one legal `(root, cut_slot)` considered by the GPU rewrite evaluator; its candidate count may be zero.",
        "- Executable Rewrite Decision: a Rewrite Decision with at least one legal replacement candidate.",
        "- Rewrite Region: the decision's actual cut-bounded removable cone from GPU rewrite `CalcMFFC/Decrease` ref-deref semantics.",
        "- Replacement Candidate: one `subgraph_index` under a Rewrite Decision.",
        "",
        "## Alignment",
        "",
        f"- strict maximal MFFC regions: `{len(regions)}`",
        f"- eligible GPU rewrite cuts: `{len(raw_rows)}`",
        f"- total legal rewrite decisions: `{len(decisions)}`",
        f"- executable rewrite decisions: `{len(executable)}`",
        f"- INTRA_REGION decisions: `{len(intra)}`",
        f"- CROSS_REGION decisions: `{len(cross)}`",
        f"- executable INTRA_REGION decisions: `{len(executable_intra)}`",
        f"- executable CROSS_REGION decisions: `{len(executable_cross)}`",
        f"- replacement candidates: `{total_candidates}`",
        f"- regions with an opportunity: `{len(regions_with_opportunity)}`",
        f"- regions without an opportunity: `{len(regions_without_opportunity)}`",
        "",
        "Classification uses `{ owner[n] | n in removable_nodes }` only. Cut leaves are retained as decision metadata and are not used to reject boundary-leaf decisions.",
        "",
        "## Outputs",
        "",
        "`region_to_decisions.json` indexes every region by all decisions whose removable set touches it. A CROSS_REGION decision is therefore listed under each touched region and has no single `decision_owner_region`.",
        "",
        "Gate 3C PASS: all legal decisions have a non-empty removable set, a complete owner-based classification, and a deterministic Region -> Decision -> Candidate index. Decisions with zero candidates are retained and do not count as rewrite opportunities. LiteCut, gradient, joint optimization, replacement, and STA were not executed.",
    ]
    (args.output / "gate3c_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    artifacts = [
        "all_rewrite_decisions.jsonl", "executable_rewrite_decisions.jsonl",
        "decision_region_alignment.csv",
        "region_to_decisions.json", "region_decision_stats.csv",
        "intra_region_decisions.jsonl", "cross_region_decisions.jsonl",
        "gate3c_summary.json", "gate3c_report.md",
    ]
    summary["artifact_sha256"] = {name: sha256_file(args.output / name) for name in artifacts}
    dump_json(args.output / "gate3c_summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
