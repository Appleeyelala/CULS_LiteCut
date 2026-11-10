#!/usr/bin/env python3
"""Gate 3A post-processing and independent standard-MFFC oracle.

The CUDA command writes only the graph representation used by CULS and the
raw PO->PI discovery regions.  This script deliberately computes the strict
oracle and all ownership checks independently on the CPU.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple


def dump_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_hash(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


def graph_node_id(literal: int) -> int:
    return literal >> 1


def strict_standard_mffc(root: int, fanin0: Sequence[int], fanin1: Sequence[int],
                         fanouts: Sequence[int], n_pis: int) -> List[int]:
    """Compute the standard root-deref MFFC with no size/depth heuristic.

    The root is retained as the region root.  Its two incoming references are
    decremented once, and a fanin is recursively dereferenced only when its
    reference count reaches zero.  PO fanouts are already included in
    ``fanouts`` by the CULS reader and therefore remain external references.
    """

    refs = list(fanouts)
    region: Set[int] = {root}
    pending: List[int] = []

    def decrement(literal: int) -> None:
        node = graph_node_id(literal)
        refs[node] -= 1
        if node > n_pis and refs[node] == 0 and node not in region:
            region.add(node)
            pending.append(node)

    decrement(fanin0[root])
    decrement(fanin1[root])
    while pending:
        node = pending.pop()
        decrement(fanin0[node])
        decrement(fanin1[node])
    return sorted(region)


def normalize_raw_regions(rows: Sequence[dict]) -> List[dict]:
    normalized = []
    for row in rows:
        internal = sorted(int(node) for node in row["internal_nodes"])
        cut = sorted(int(node) for node in row["cut_nodes"])
        if int(row.get("traversal_internal_size", len(internal))) != len(internal):
            raise RuntimeError(
                f"root {row['root']}: traversal size does not match dumped node list"
            )
        if int(row["cone_size"]) != len(internal):
            raise RuntimeError(
                f"root {row['root']}: CULS cone_size does not match dumped node list"
            )
        normalized.append({
            "root": int(row["root"]),
            "cut_nodes": cut,
            "internal_nodes": internal,
            "cone_size": int(row["cone_size"]),
            "traversal_internal_size": len(internal),
            "cut_size": int(row["cut_size"]),
            "discovery_level": int(row["discovery_level"]),
        })
    normalized.sort(key=lambda row: (row["discovery_level"], row["root"]))
    for region_id, row in enumerate(normalized):
        row["region_id"] = region_id
    return normalized


def partition_checks(name: str, regions: Sequence[dict], and_nodes: Set[int]) -> dict:
    owners: Dict[int, List[int]] = {}
    for region_id, region in enumerate(regions):
        for node in set(region["internal_nodes"]):
            owners.setdefault(node, []).append(region_id)

    covered = set(owners)
    unowned = sorted(and_nodes - covered)
    multiply = {
        node: sorted(region_ids)
        for node, region_ids in owners.items()
        if len(region_ids) > 1
    }

    overlap_pairs = []
    for left in range(len(regions)):
        left_nodes = set(regions[left]["internal_nodes"])
        for right in range(left + 1, len(regions)):
            overlap = sorted(left_nodes.intersection(regions[right]["internal_nodes"]))
            if overlap:
                overlap_pairs.append({
                    "region_a": left,
                    "region_b": right,
                    "root_a": regions[left]["root"],
                    "root_b": regions[right]["root"],
                    "overlap_nodes": overlap,
                })

    return {
        "partition": name,
        "region_count": len(regions),
        "total_and_nodes": len(and_nodes),
        "covered_and_nodes": len(covered.intersection(and_nodes)),
        "uncovered_and_nodes": unowned,
        "uncovered_and_count": len(unowned),
        "coverage_ratio": (len(covered.intersection(and_nodes)) / len(and_nodes)
                            if and_nodes else 1.0),
        "full_coverage": not unowned,
        "multiply_owned_nodes": [
            {"node": node, "region_ids": region_ids}
            for node, region_ids in sorted(multiply.items())
        ],
        "multiply_owned_count": len(multiply),
        "unowned_nodes": unowned,
        "overlap_pair_count": len(overlap_pairs),
        "overlap_pairs": overlap_pairs,
        "disjoint": not multiply,
        "unique_ownership": not unowned and not multiply,
        "owner_map": [
            {"node": node, "region_ids": sorted(owners.get(node, []))}
            for node in sorted(and_nodes)
        ],
    }


def write_node_map(path: Path, regions: Sequence[dict], and_nodes: Set[int], region_type: str) -> None:
    owners: Dict[int, List[int]] = {}
    for region_id, region in enumerate(regions):
        for node in set(region["internal_nodes"]):
            owners.setdefault(node, []).append(region_id)

    with path.open("w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out)
        writer.writerow(["node_id", "region_id", "region_root", "region_type", "owner_count"])
        for node in sorted(and_nodes):
            node_owners = owners.get(node, [])
            if len(node_owners) == 1:
                region_id = node_owners[0]
                root = regions[region_id]["root"]
                type_value = regions[region_id].get("semantics", region_type)
            else:
                region_id = -1
                root = -1
                type_value = "UNOWNED_OR_MULTIPLY_OWNED"
            writer.writerow([node, region_id, root, type_value, len(node_owners)])


def region_size_stats(regions: Sequence[dict]) -> dict:
    sizes = [len(set(region["internal_nodes"])) for region in regions]
    if not sizes:
        return {
            "region_count": 0,
            "min_region_size": 0,
            "max_region_size": 0,
            "mean_region_size": 0.0,
            "median_region_size": 0.0,
            "singleton_regions": 0,
        }
    return {
        "region_count": len(sizes),
        "min_region_size": min(sizes),
        "max_region_size": max(sizes),
        "mean_region_size": statistics.fmean(sizes),
        "median_region_size": statistics.median(sizes),
        "singleton_regions": sum(size == 1 for size in sizes),
    }


def write_code_audit(path: Path) -> None:
    path.write_text(
        "# CULS MFFC code audit\n\n"
        "This Gate 3A audit uses the repository's original PO-to-PI discovery\n"
        "front half and does not call the refactoring back half.\n\n"
        "## Actual call chain\n\n"
        "```text\n"
        "PO drivers\n"
        "  -> initial roots (AND PO drivers, node-id sort/unique)\n"
        "  -> recordMFFC()\n"
        "       -> Aig::findReconvMFFCCut()\n"
        "            -> findReconvMFFCCut_iter()\n"
        "            -> ref-count-guided cut expansion\n"
        "  -> setStatus()\n"
        "       -> AND nodes in the current cut whose vCutSizes == -1\n"
        "  -> next roots\n"
        "  -> repeat toward PI/constant boundaries\n"
        "```\n\n"
        "Relevant implementation points:\n\n"
        "- `src/algorithms/refactor_mffc.cu`: `recordMFFC`, `setStatus`, and the\n"
        "  discovery loop in `refactorMFFCPerform`.\n"
        "- `src/aig/mffc.cuh`: `findReconvMFFCCut` initializes the root and its\n"
        "  fanins, then `findReconvMFFCCut_iter` expands only a zero-reference\n"
        "  cut node.\n"
        "- `src/aig/truth.cuh`: the shared `collectCutConeNodes` traversal is used\n"
        "  by the existing truth path and by the new audit-only dump kernel.\n\n"
        "## Cut limit semantics\n\n"
        "`findReconvMFFCCut_iter` returns without expanding when\n"
        "`vCutSizes[root] - 1 + bestCost > nMaxCutSize`. Thus `nMaxCutSize` is\n"
        "a real early-stop condition. The audit retains every discovered\n"
        "discovery cone, including singleton cones; it does not apply the later\n"
        "refactor candidate filter `isSmallMFFC` (`cone_size < 2`).\n\n"
        "Accordingly, a dumped region is not called a complete MFFC until it is\n"
        "compared with the independent unlimited CPU deref/ref oracle. An exact\n"
        "match is `STANDARD_MFFC`; a strict subset is classified as\n"
        "`CUT_LIMITED_FFC`; any non-subset is a `MISMATCH` and blocks the gate.\n\n"
        "## Audit boundary\n\n"
        "The Gate 3A command stops after cut discovery and the shared\n"
        "cut-bounded internal-node traversal. It does not execute truth-table\n"
        "factorization, candidate evaluation, resynthesis, replacement, LiteCut,\n"
        "STA, rewrite, or gradient code.\n",
        encoding="utf-8",
    )


def scan_dataset(dataset_root: Path) -> List[Tuple[Path, int]]:
    files = [
        path.resolve()
        for path in dataset_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".aig", ".aiger"}
    ]
    files.sort(key=lambda path: str(path))
    return [(path, path.stat().st_size) for path in files]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--original-input", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--design", default="")
    parser.add_argument("--culs-head", required=True)
    parser.add_argument("--culs-log", required=True)
    parser.add_argument("--culs-status", required=True)
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    original_input = args.original_input.resolve()
    output_root = args.output_root.resolve()
    design = args.design or original_input.stem
    design_root = output_root / design
    run1 = design_root / "run1"
    run2 = design_root / "run2"
    output_root.mkdir(parents=True, exist_ok=True)
    design_root.mkdir(parents=True, exist_ok=True)

    files = scan_dataset(dataset_root)
    manifest_lines = [
        "AIG dataset root: " + str(dataset_root),
        "AIG file count: " + str(len(files)),
        "",
        "file path\tfile size (bytes)",
    ]
    manifest_lines.extend(f"{path}\t{size}" for path, size in files)
    (output_root / "dataset_manifest.txt").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    culs_manifest = {
        "repository": "/home/wushangge/CULS",
        "git_status_at_audit_start": args.culs_status,
        "git_rev_parse_HEAD_at_audit_start": args.culs_head,
        "git_log_1_at_audit_start": args.culs_log,
        "audit_input_policy": "only /home/wushangge/cutLibOrigin/open-sourceDataset",
        "gate_scope": "CULS MFFC/FFC discovery only; no LiteCut/rewrite/STA/gradient",
    }
    dump_json(output_root / "culs_manifest.json", culs_manifest)
    write_code_audit(output_root / "culs_mffc_code_audit.md")

    if not original_input.is_file():
        raise RuntimeError(f"selected input does not exist: {original_input}")
    if not any(path == original_input for path, _ in files):
        raise RuntimeError("selected input is not present in the audited dataset manifest")

    graph1 = json.loads((run1 / "graph.json").read_text(encoding="utf-8"))
    graph2 = json.loads((run2 / "graph.json").read_text(encoding="utf-8"))
    graph_compare_fields = ["n_objs", "n_pis", "n_pos", "n_nodes", "fanin0", "fanin1", "outs", "num_fanouts"]
    if any(graph1[field] != graph2[field] for field in graph_compare_fields):
        raise RuntimeError("run1 and run2 graph dumps differ")

    raw1 = normalize_raw_regions(load_jsonl(run1 / "culs_regions.jsonl"))
    raw2 = normalize_raw_regions(load_jsonl(run2 / "culs_regions.jsonl"))
    raw_hash1 = canonical_json_hash(raw1)
    raw_hash2 = canonical_json_hash(raw2)

    graph = graph1
    n_objs = int(graph["n_objs"])
    n_pis = int(graph["n_pis"])
    n_pos = int(graph["n_pos"])
    n_nodes = int(graph["n_nodes"])
    fanin0 = graph["fanin0"]
    fanin1 = graph["fanin1"]
    fanouts = graph["num_fanouts"]
    and_nodes = set(range(n_pis + 1, n_objs))
    if len(and_nodes) != n_nodes:
        raise RuntimeError("graph node count is inconsistent with n_pis/n_objs/n_nodes")

    original_header = original_input.read_bytes().splitlines()[0].decode("ascii")
    header_parts = original_header.split()
    if len(header_parts) < 6 or header_parts[0] != "aig":
        raise RuntimeError(f"unexpected AIG header: {original_header}")

    input_manifest = {
        "selected_design": design,
        "selection_rule": "use c1355 corresponding AIG when present",
        "dataset_root": str(dataset_root),
        "absolute_input_path": str(original_input),
        "file_size_bytes": original_input.stat().st_size,
        "sha256": sha256_file(original_input),
        "format": "aig",
        "header": original_header,
        "pi_count": n_pis,
        "po_count": n_pos,
        "and_count": n_nodes,
        "culs_graph_counts": {
            "n_objs_including_const": n_objs,
            "n_pis": n_pis,
            "n_pos": n_pos,
            "n_nodes": n_nodes,
        },
    }
    dump_json(design_root / "input_manifest.json", input_manifest)

    # The strict oracle is intentionally evaluated for every AND root, not
    # only for roots selected by the CULS PO->PI discovery.
    standard_by_root = {
        root: strict_standard_mffc(root, fanin0, fanin1, fanouts, n_pis)
        for root in sorted(and_nodes)
    }
    all_mffcs = [
        {
            "root": root,
            "internal_nodes": standard_by_root[root],
            "cone_size": len(standard_by_root[root]),
            "semantics": "STANDARD_MFFC",
        }
        for root in sorted(and_nodes)
    ]

    regions = []
    comparisons = []
    mismatch_roots = []
    for culs_region in raw1:
        root = culs_region["root"]
        standard_nodes = standard_by_root[root]
        culs_nodes = sorted(set(culs_region["internal_nodes"]))
        standard_set = set(standard_nodes)
        culs_set = set(culs_nodes)
        if culs_set == standard_set:
            relation = "EXACT_MFFC"
            semantics = "STANDARD_MFFC"
        elif culs_set < standard_set:
            relation = "STRICT_SUBSET"
            semantics = "CUT_LIMITED_FFC"
        else:
            relation = "MISMATCH"
            semantics = "MISMATCH"
            mismatch_roots.append(root)

        final_region = dict(culs_region)
        final_region["region_id"] = len(regions)
        final_region["internal_nodes"] = culs_nodes
        final_region["semantics"] = semantics
        final_region["classification"] = relation
        final_region["standard_mffc_size"] = len(standard_nodes)
        final_region["standard_mffc_nodes"] = standard_nodes
        regions.append(final_region)

        comparisons.append({
            "root": root,
            "culs_region_id": final_region["region_id"],
            "culs_cone_size": len(culs_nodes),
            "standard_mffc_size": len(standard_nodes),
            "culs_cut_size": final_region["cut_size"],
            "relation": relation,
            "semantics": semantics,
            "strict_extra_count": len(standard_set - culs_set),
            "culs_missing_count": len(culs_set - standard_set),
        })

    culs_checks = partition_checks("CULS_FFC_MFFC_PARTITION", regions, and_nodes)
    culs_checks["classification_mismatch_roots"] = mismatch_roots
    culs_checks["classification_valid"] = not mismatch_roots
    dump_json(design_root / "culs_partition_checks.json", culs_checks)

    maximal_indices = []
    mffc_sets = [set(row["internal_nodes"]) for row in all_mffcs]
    for index, current in enumerate(mffc_sets):
        if any(current < other for other in mffc_sets if other is not current):
            continue
        maximal_indices.append(index)

    strict_regions = []
    for strict_region_id, index in enumerate(maximal_indices):
        source = all_mffcs[index]
        strict_regions.append({
            "region_id": strict_region_id,
            "root": source["root"],
            "internal_nodes": source["internal_nodes"],
            "cone_size": source["cone_size"],
            "semantics": "STANDARD_MFFC",
        })
    strict_checks = partition_checks("STRICT_MAXIMAL_MFFC_PARTITION", strict_regions, and_nodes)
    dump_json(design_root / "strict_mffc_partition_checks.json", strict_checks)

    jsonl(design_root / "culs_regions.jsonl", regions)
    jsonl(design_root / "all_standard_mffcs.jsonl", all_mffcs)
    dump_json(design_root / "maximal_standard_mffcs.json", {
        "partition": "STRICT_MAXIMAL_MFFC_PARTITION",
        "regions": strict_regions,
        "source_all_standard_mffc_count": len(all_mffcs),
        "maximal_region_count": len(strict_regions),
    })

    with (design_root / "culs_vs_standard_mffc.csv").open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=list(comparisons[0].keys()) if comparisons else ["root"])
        writer.writeheader()
        writer.writerows(comparisons)

    write_node_map(design_root / "node_to_culs_region.csv", regions, and_nodes, "CULS_REGION")
    write_node_map(design_root / "node_to_strict_mffc_region.csv", strict_regions, and_nodes, "STANDARD_MFFC")

    culs_stats = region_size_stats(regions)
    culs_stats.update({
        "partition": "CULS_FFC_MFFC_PARTITION",
        "full_standard_mffc_regions": sum(row["semantics"] == "STANDARD_MFFC" for row in regions),
        "cut_limited_ffc_regions": sum(row["semantics"] == "CUT_LIMITED_FFC" for row in regions),
        "mismatch_regions": sum(row["semantics"] == "MISMATCH" for row in regions),
    })
    strict_stats = region_size_stats(strict_regions)
    strict_stats["partition"] = "STRICT_MAXIMAL_MFFC_PARTITION"
    dump_json(design_root / "region_stats.json", {
        "and_node_count": n_nodes,
        "number_of_culs_regions": len(regions),
        "number_of_all_standard_mffcs": len(all_mffcs),
        "number_of_strict_maximal_mffcs": len(strict_regions),
        "culs": culs_stats,
        "strict_all_standard_mffcs": region_size_stats(all_mffcs),
        "strict_maximal": strict_stats,
    })

    deterministic = (
        raw_hash1 == raw_hash2
        and len(raw1) == len(raw2)
        and canonical_json_hash(culs_checks["owner_map"]) == canonical_json_hash(
            partition_checks("CULS_FFC_MFFC_PARTITION", raw2, and_nodes)["owner_map"]
        )
    )
    cut_size_limit = json.loads((run1 / "culs_run_manifest.json").read_text(encoding="utf-8"))["cut_size_limit"]
    culs_valid_partition = culs_checks["classification_valid"] and culs_checks["unique_ownership"]
    strict_valid_partition = strict_checks["unique_ownership"]
    if strict_valid_partition:
        recommendation = "STRICT_MAXIMAL_MFFC_PARTITION"
    elif culs_valid_partition:
        recommendation = "CULS_FFC_MFFC_PARTITION"
    else:
        recommendation = "NEITHER_FULL_DISJOINT_PARTITION_BLOCKED"

    summary = {
        "gate": "Gate 3A",
        "gate3a_pass": bool(deterministic and not mismatch_roots),
        "selected_design": design,
        "selected_input": str(original_input),
        "input_sha256": input_manifest["sha256"],
        "cut_size_limit": cut_size_limit,
        "deterministic": deterministic,
        "run1_culs_regions_sha256": raw_hash1,
        "run2_culs_regions_sha256": raw_hash2,
        "q1": {"pi": n_pis, "po": n_pos, "and": n_nodes},
        "q2_culs_region_count": len(regions),
        "q3_culs_full_coverage": culs_checks["full_coverage"],
        "q4_culs_disjoint": culs_checks["disjoint"],
        "q4_culs_unique_ownership": culs_checks["unique_ownership"],
        "q5_full_standard_mffc_regions": culs_stats["full_standard_mffc_regions"],
        "q5_cut_limited_ffc_regions": culs_stats["cut_limited_ffc_regions"],
        "q5_mismatch_regions": culs_stats["mismatch_regions"],
        "q6_strict_maximal_full_coverage": strict_checks["full_coverage"],
        "q6_strict_maximal_disjoint": strict_checks["disjoint"],
        "q6_strict_maximal_unique_ownership": strict_checks["unique_ownership"],
        "strict_all_standard_mffc_count": len(all_mffcs),
        "strict_maximal_mffc_count": len(strict_regions),
        "recommended_partition": recommendation,
        "culs_partition_checks": culs_checks,
        "strict_mffc_partition_checks": strict_checks,
    }
    dump_json(output_root / "gate3a_summary.json", summary)

    report = f"# Gate 3A report — {design}\n\n"
    report += "## 固定输入与规模\n\n"
    report += f"- 输入：`{original_input}`\n- SHA256：`{input_manifest['sha256']}`\n"
    report += f"- PI / PO / AND：`{n_pis} / {n_pos} / {n_nodes}`\n- CULS cut limit K：`{cut_size_limit}`\n\n"
    report += "## 7 个问题的结论\n\n"
    report += f"1. Q1：选择 `{design}`；PI={n_pis}，PO={n_pos}，AND={n_nodes}。\n"
    report += f"2. Q2：CULS PO→PI discovery 得到 `{len(regions)}` 个 regions。\n"
    report += f"3. Q3：CULS 覆盖率 `{culs_checks['coverage_ratio']:.6f}`；100% = `{culs_checks['full_coverage']}`。\n"
    report += f"4. Q4：disjoint=`{culs_checks['disjoint']}`，unique owner=`{culs_checks['unique_ownership']}`，multiply-owned={culs_checks['multiply_owned_count']}，unowned={culs_checks['uncovered_and_count']}。\n"
    report += f"5. Q5：完整 Standard MFFC `{culs_stats['full_standard_mffc_regions']}` 个；cut-limited FFC `{culs_stats['cut_limited_ffc_regions']}` 个；mismatch `{culs_stats['mismatch_regions']}` 个。\n"
    report += f"6. Q6：inclusion-maximal standard MFFC 数量 `{len(strict_regions)}`；100% coverage=`{strict_checks['full_coverage']}`，disjoint=`{strict_checks['disjoint']}`。\n"
    report += f"7. Q7：基于实测结果建议 `{recommendation}`。\n\n"
    report += "## 语义与 determinism\n\n"
    report += "CULS raw discovery 保留 singleton cone，没有人为补 singleton；后续原始 refactor 的 `cone_size < 2` filter 未执行。\n\n"
    report += f"run1 canonical SHA256=`{raw_hash1}`；run2 canonical SHA256=`{raw_hash2}`；deterministic=`{deterministic}`。\n\n"
    report += "若出现 `MISMATCH`，说明 CULS internal-node extraction 不是 strict standard MFFC 的子集，Gate 应阻断；本报告会保留所有 counterexample。\n"
    (output_root / "gate3a_report.md").write_text(report, encoding="utf-8")

    print(json.dumps({
        "output_root": str(output_root),
        "design": design,
        "culs_regions": len(regions),
        "culs_full_coverage": culs_checks["full_coverage"],
        "culs_unique_ownership": culs_checks["unique_ownership"],
        "strict_maximal_regions": len(strict_regions),
        "strict_maximal_full_coverage": strict_checks["full_coverage"],
        "strict_maximal_disjoint": strict_checks["disjoint"],
        "deterministic": deterministic,
        "recommended_partition": recommendation,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
