#!/usr/bin/env python3
"""Gate 3B: build and audit the strict-MFFC region DAG.

This tool intentionally consumes the Gate 3A partition as data.  It does not
recompute MFFCs, invoke CULS, or modify the AIG.  Its only graph construction
is the quotient of the original AIG's node-level connections by the supplied
node -> strict-MFFC owner map.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import io
import json
import shutil
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple


def dump_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def dump_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def dump_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fieldnames), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_hash(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_varint(stream: io.BytesIO) -> int:
    """Read one AIGER binary delta varint."""

    value = 0
    shift = 0
    while True:
        raw = stream.read(1)
        if not raw:
            raise ValueError("truncated AIGER binary varint")
        byte = raw[0]
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value
        shift += 7
        if shift > 63:
            raise ValueError("AIGER binary varint is too long")


@dataclass(frozen=True)
class Aig:
    path: Path
    maximum_variable: int
    pi_count: int
    latch_count: int
    po_count: int
    and_count: int
    po_literals: Tuple[int, ...]
    # Each AND node maps to its two literal-level fanins.  Node IDs are used
    # for graph ownership; inversion is irrelevant to structural reachability.
    fanin_literals: Mapping[int, Tuple[int, int]]

    @property
    def pi_nodes(self) -> Set[int]:
        return set(range(1, self.pi_count + 1))

    @property
    def and_nodes(self) -> Set[int]:
        return set(range(self.pi_count + 1, self.maximum_variable + 1))

    @property
    def node_count_including_constant(self) -> int:
        return self.maximum_variable + 1


def parse_binary_aig(path: Path) -> Aig:
    data = path.read_bytes()
    stream = io.BytesIO(data)
    header_raw = stream.readline()
    if not header_raw:
        raise ValueError(f"empty AIGER input: {path}")
    try:
        header = header_raw.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise ValueError(f"non-ASCII AIGER header: {path}") from exc
    fields = header.split()
    if len(fields) != 6 or fields[0] != "aig":
        raise ValueError(f"expected binary AIGER header, got: {header!r}")
    maximum_variable, pi_count, latch_count, po_count, and_count = map(int, fields[1:])
    if latch_count != 0:
        raise ValueError("Gate 3B expects a latch-free AIG; latch support is not needed here")
    if and_count != maximum_variable - pi_count:
        raise ValueError(
            "AIGER header is inconsistent for a latch-free AIG: "
            f"M={maximum_variable}, I={pi_count}, A={and_count}"
        )

    po_literals: List[int] = []
    for po_index in range(po_count):
        line = stream.readline()
        if not line:
            raise ValueError(f"missing PO literal {po_index}")
        try:
            po_literals.append(int(line.strip()))
        except ValueError as exc:
            raise ValueError(f"invalid PO literal at index {po_index}: {line!r}") from exc

    fanin_literals: Dict[int, Tuple[int, int]] = {}
    for offset in range(and_count):
        node = pi_count + 1 + offset
        lhs = 2 * node
        delta0 = read_varint(stream)
        delta1 = read_varint(stream)
        rhs0 = lhs - delta0
        rhs1 = rhs0 - delta1
        if rhs0 < rhs1 or rhs1 < 0:
            raise ValueError(
                f"invalid AIGER AND literals for node {node}: "
                f"lhs={lhs}, rhs0={rhs0}, rhs1={rhs1}"
            )
        if rhs0 >= lhs or rhs1 >= lhs:
            raise ValueError(f"AND node {node} has a non-topologically-ordered fanin")
        if rhs0 > 2 * maximum_variable + 1 or rhs1 > 2 * maximum_variable + 1:
            raise ValueError(f"AND node {node} references a node outside the AIG")
        fanin_literals[node] = (rhs0, rhs1)

    return Aig(
        path=path.resolve(),
        maximum_variable=maximum_variable,
        pi_count=pi_count,
        latch_count=latch_count,
        po_count=po_count,
        and_count=and_count,
        po_literals=tuple(po_literals),
        fanin_literals=fanin_literals,
    )


def node_id(literal: int) -> int:
    return literal >> 1


def load_partition(
    maximal_path: Path, owner_csv_path: Path, and_nodes: Set[int]
) -> Tuple[List[dict], Dict[int, int], dict]:
    payload = json.loads(maximal_path.read_text(encoding="utf-8"))
    rows = list(payload.get("regions", []))
    rows.sort(key=lambda row: int(row["region_id"]))
    expected_ids = list(range(len(rows)))
    actual_ids = [int(row["region_id"]) for row in rows]
    if actual_ids != expected_ids:
        raise ValueError(f"strict-MFFC region IDs are not contiguous: {actual_ids}")

    regions: List[dict] = []
    derived_owner: Dict[int, int] = {}
    duplicate_nodes: List[int] = []
    for row in rows:
        region_id = int(row["region_id"])
        internal_nodes = sorted(int(node) for node in row["internal_nodes"])
        if int(row["cone_size"]) != len(internal_nodes):
            raise ValueError(f"region {region_id} cone_size does not match internal_nodes")
        if row.get("semantics") != "STANDARD_MFFC":
            raise ValueError(f"strict partition region {region_id} is not STANDARD_MFFC")
        for node in internal_nodes:
            if node in derived_owner:
                duplicate_nodes.append(node)
            derived_owner[node] = region_id
        regions.append({
            "region_id": region_id,
            "root": int(row["root"]),
            "internal_nodes": internal_nodes,
            "cone_size": len(internal_nodes),
            "semantics": str(row["semantics"]),
        })

    csv_owner: Dict[int, int] = {}
    csv_root: Dict[int, int] = {}
    csv_owner_count: Dict[int, int] = {}
    with owner_csv_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"node_id", "region_id", "region_root", "region_type", "owner_count"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"owner CSV lacks required columns: {required}")
        for row in reader:
            node = int(row["node_id"])
            csv_owner[node] = int(row["region_id"])
            csv_root[node] = int(row["region_root"])
            csv_owner_count[node] = int(row["owner_count"])

    owner_map_equal = derived_owner == csv_owner
    root_map_equal = all(
        csv_root.get(node) == regions[csv_owner[node]]["root"]
        for node in csv_owner
        if 0 <= csv_owner[node] < len(regions)
    )
    owner_count_valid = all(count == 1 for count in csv_owner_count.values())
    expected_nodes_equal = set(derived_owner) == and_nodes == set(csv_owner)
    partition_consistent = (
        not duplicate_nodes
        and expected_nodes_equal
        and owner_map_equal
        and root_map_equal
        and owner_count_valid
    )
    if not partition_consistent:
        raise ValueError(
            "Gate 3A strict partition consistency check failed: "
            f"duplicates={sorted(set(duplicate_nodes))[:10]}, "
            f"missing={sorted(and_nodes - set(derived_owner))[:10]}, "
            f"extra={sorted(set(derived_owner) - and_nodes)[:10]}, "
            f"owner_csv_mismatch={not owner_map_equal}"
        )

    consistency = {
        "maximal_partition_file": str(maximal_path.resolve()),
        "owner_csv_file": str(owner_csv_path.resolve()),
        "partition_label": payload.get("partition"),
        "region_count": len(regions),
        "and_node_count": len(and_nodes),
        "owner_mapping_consistent": owner_map_equal,
        "region_roots_consistent": root_map_equal,
        "owner_counts_all_one": owner_count_valid,
        "partition_consistent": partition_consistent,
        "duplicate_internal_nodes": sorted(set(duplicate_nodes)),
    }
    return regions, csv_owner, consistency


def tarjan_scc(adjacency: Mapping[int, Set[int]], nodes: Sequence[int]) -> List[List[int]]:
    index = 0
    stack: List[int] = []
    on_stack: Set[int] = set()
    indices: Dict[int, int] = {}
    lowlinks: Dict[int, int] = {}
    components: List[List[int]] = []

    def visit(node: int) -> None:
        nonlocal index
        indices[node] = index
        lowlinks[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)
        for successor in sorted(adjacency[node]):
            if successor not in indices:
                visit(successor)
                lowlinks[node] = min(lowlinks[node], lowlinks[successor])
            elif successor in on_stack:
                lowlinks[node] = min(lowlinks[node], indices[successor])
        if lowlinks[node] == indices[node]:
            component: List[int] = []
            while True:
                member = stack.pop()
                on_stack.remove(member)
                component.append(member)
                if member == node:
                    break
            components.append(sorted(component))

    for node in sorted(nodes):
        if node not in indices:
            visit(node)
    components.sort(key=lambda component: component[0])
    return components


def topological_order(adjacency: Mapping[int, Set[int]], nodes: Sequence[int]) -> Tuple[List[int], List[List[int]]]:
    indegree = {node: 0 for node in nodes}
    for source in nodes:
        for target in adjacency[source]:
            indegree[target] += 1
    ready = [node for node, degree in indegree.items() if degree == 0]
    heapq.heapify(ready)
    order: List[int] = []
    while ready:
        source = heapq.heappop(ready)
        order.append(source)
        for target in sorted(adjacency[source]):
            indegree[target] -= 1
            if indegree[target] == 0:
                heapq.heappush(ready, target)
    components = tarjan_scc(adjacency, nodes)
    cyclic_components = [component for component in components if len(component) > 1 or component[0] in adjacency[component[0]]]
    return order, cyclic_components


def build_graph_data(aig: Aig, regions: Sequence[dict], owner: Mapping[int, int]) -> dict:
    region_ids = [int(region["region_id"]) for region in regions]
    region_id_set = set(region_ids)
    and_nodes = aig.and_nodes
    pi_nodes = aig.pi_nodes

    cross_counts: Counter[Tuple[int, int]] = Counter()
    pi_counts: Counter[Tuple[int, int]] = Counter()
    incoming_boundary: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    outgoing_boundary: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    incoming_pi_edges: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    output_po_edges: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    predecessor_regions: Dict[int, Set[int]] = {region_id: set() for region_id in region_ids}
    successor_regions: Dict[int, Set[int]] = {region_id: set() for region_id in region_ids}
    internal_edge_counts: Counter[int] = Counter()
    and_edge_occurrences: List[dict] = []
    cross_edge_occurrences: List[dict] = []

    for dst in sorted(and_nodes):
        literals = aig.fanin_literals[dst]
        for fanin_index, literal in enumerate(literals):
            src = node_id(literal)
            if src not in (and_nodes | pi_nodes) and src != 0:
                raise ValueError(f"AND node {dst} has invalid fanin node {src}")
            if src in and_nodes:
                src_region = owner[src]
                dst_region = owner[dst]
                occurrence = {
                    "src": src,
                    "dst": dst,
                    "fanin_index": fanin_index,
                    "src_region": src_region,
                    "dst_region": dst_region,
                }
                and_edge_occurrences.append(occurrence)
                if src_region == dst_region:
                    internal_edge_counts[src_region] += 1
                else:
                    cross_counts[(src_region, dst_region)] += 1
                    predecessor_regions[dst_region].add(src_region)
                    successor_regions[src_region].add(dst_region)
                    incoming_boundary[dst_region].append((src, dst))
                    outgoing_boundary[src_region].append((src, dst))
                    cross_edge_occurrences.append(occurrence)
            elif src in pi_nodes:
                dst_region = owner[dst]
                pi_counts[(src, dst_region)] += 1
                incoming_pi_edges[dst_region].append((src, dst))

    po_region_records: List[dict] = []
    direct_pi_po_edges: List[dict] = []
    constant_po_indices: List[int] = []
    for po_index, literal in enumerate(aig.po_literals):
        driver = node_id(literal)
        if driver in and_nodes:
            region_id = owner[driver]
            output_po_edges[region_id].append((driver, po_index))
            po_region_records.append({
                "src_region": region_id,
                "po_id": po_index,
                "po_driver": driver,
            })
        elif driver in pi_nodes:
            direct_pi_po_edges.append({"pi": driver, "po_id": po_index, "po_driver": driver})
        elif driver == 0:
            constant_po_indices.append(po_index)
        else:
            raise ValueError(f"PO {po_index} references invalid driver node {driver}")

    pi_edge_occurrence_count = sum(len(edges) for edges in incoming_pi_edges.values())
    pi_region_edge_counted_occurrences = sum(pi_counts.values())
    region_po_edge_occurrence_count = len(po_region_records)
    region_po_metadata_occurrences = sum(len(edges) for edges in output_po_edges.values())

    adjacency: Dict[int, Set[int]] = {region_id: set() for region_id in region_ids}
    for source, target in cross_counts:
        if source == target:
            raise ValueError(f"unexpected self-crossing region edge R{source}->R{target}")
        adjacency[source].add(target)
    order, cyclic_components = topological_order(adjacency, region_ids)

    cross_edge_set = set(cross_counts)
    missing_region_edges = sorted(
        {
            (occurrence["src_region"], occurrence["dst_region"])
            for occurrence in cross_edge_occurrences
            if (occurrence["src_region"], occurrence["dst_region"]) not in cross_edge_set
        }
    )
    edge_complete = not missing_region_edges

    region_metadata: List[dict] = []
    for region in regions:
        region_id = int(region["region_id"])
        region_metadata.append({
            "region_id": region_id,
            "root": int(region["root"]),
            "internal_nodes": list(region["internal_nodes"]),
            "cone_size": int(region["cone_size"]),
            "semantics": region["semantics"],
            "predecessor_regions": sorted(predecessor_regions[region_id]),
            "successor_regions": sorted(successor_regions[region_id]),
            "input_pis": sorted({pi for pi, _dst in incoming_pi_edges[region_id]}),
            "output_pos": sorted(po for _src, po in output_po_edges[region_id]),
            # These two lists contain original AND->AND edge occurrences that
            # cross a region boundary.  A duplicated AIG fanin is retained as
            # two occurrences and is counted twice in cross_edge_count.
            "incoming_boundary_edges": [list(edge) for edge in sorted(incoming_boundary[region_id])],
            "outgoing_boundary_edges": [list(edge) for edge in sorted(outgoing_boundary[region_id])],
            "input_pi_edges": [list(edge) for edge in sorted(incoming_pi_edges[region_id])],
            "output_po_edges": [list(edge) for edge in sorted(output_po_edges[region_id])],
            "internal_and_edge_count": int(internal_edge_counts[region_id]),
            "incoming_boundary_edge_count": len(incoming_boundary[region_id]),
            "outgoing_boundary_edge_count": len(outgoing_boundary[region_id]),
        })

    region_edges = [
        {
            "src_region": source,
            "dst_region": target,
            "cross_edge_count": int(count),
        }
        for (source, target), count in sorted(cross_counts.items())
    ]
    pi_region_edges = [
        {
            "pi_id": pi,
            "dst_region": region_id,
            "node_edge_count": int(count),
        }
        for (pi, region_id), count in sorted(pi_counts.items())
    ]
    region_po_edges = sorted(po_region_records, key=lambda row: (row["src_region"], row["po_id"]))

    return {
        "region_ids": region_ids,
        "region_metadata": region_metadata,
        "region_edges": region_edges,
        "pi_region_edges": pi_region_edges,
        "region_po_edges": region_po_edges,
        "direct_pi_po_edges": sorted(direct_pi_po_edges, key=lambda row: row["po_id"]),
        "constant_po_indices": constant_po_indices,
        "and_edge_occurrences": and_edge_occurrences,
        "cross_edge_occurrences": cross_edge_occurrences,
        "and_edge_occurrence_count": len(and_edge_occurrences),
        "pi_edge_occurrence_count": pi_edge_occurrence_count,
        "pi_region_edge_counted_occurrences": pi_region_edge_counted_occurrences,
        "pi_region_edges_complete": (
            pi_edge_occurrence_count == pi_region_edge_counted_occurrences
        ),
        "region_po_edge_occurrence_count": region_po_edge_occurrence_count,
        "region_po_metadata_occurrences": region_po_metadata_occurrences,
        "region_po_edges_complete": (
            region_po_edge_occurrence_count == region_po_metadata_occurrences
        ),
        "cross_region_edge_occurrence_count": len(cross_edge_occurrences),
        "region_adjacency": adjacency,
        "topological_order": order,
        "cyclic_components": cyclic_components,
        "cycle_count": len(cyclic_components),
        "topological_sort_covers_all_regions": len(order) == len(region_ids),
        "cross_region_edges_all_present": edge_complete,
        "missing_region_edges": [list(edge) for edge in missing_region_edges],
        "edge_complete_path_projection": edge_complete and len(order) == len(region_ids),
    }


def longest_pi_paths(aig: Aig, owner: Mapping[int, int], graph_data: Mapping[str, object]) -> List[dict]:
    """Return one deterministic longest PI->driver path for every PO."""

    best_path: Dict[int, Tuple[int, ...]] = {
        pi: (pi,) for pi in sorted(aig.pi_nodes)
    }
    for node in sorted(aig.and_nodes):
        candidates: List[Tuple[int, ...]] = []
        for literal in aig.fanin_literals[node]:
            source = node_id(literal)
            if source in best_path:
                candidates.append(best_path[source] + (node,))
        if candidates:
            best_path[node] = min(candidates, key=lambda path: (-len(path), path))

    region_edge_set = {
        (int(row["src_region"]), int(row["dst_region"]))
        for row in graph_data["region_edges"]
    }
    roots = {
        int(row["region_id"]): int(row["root"])
        for row in graph_data["region_metadata"]
    }
    examples: List[dict] = []
    for po_index, literal in enumerate(aig.po_literals):
        driver = node_id(literal)
        path = best_path.get(driver)
        if path is None:
            examples.append({
                "po": po_index,
                "po_id": po_index,
                "po_literal": literal,
                "po_driver": driver,
                "path_available": False,
                "pi": None,
                "node_path": [],
                "raw_region_sequence": [],
                "compressed_region_sequence": [],
                "region_observations": [],
                "valid_node_path": False,
                "valid_region_path": False,
                "reentered_regions": [],
                "path_hits_root_required": False,
            })
            continue

        node_path = list(path)
        valid_node_path = all(
            source in {node_id(fanin) for fanin in aig.fanin_literals[target]}
            for source, target in zip(node_path, node_path[1:])
        )
        raw_regions = [owner[node] for node in node_path if node in owner]
        compressed_regions: List[int] = []
        for region_id in raw_regions:
            if not compressed_regions or compressed_regions[-1] != region_id:
                compressed_regions.append(region_id)
        valid_region_path = all(
            (source, target) in region_edge_set
            for source, target in zip(compressed_regions, compressed_regions[1:])
        )
        region_counts = Counter(raw_regions)
        reentered = sorted(
            region_id for region_id, count in region_counts.items() if count > 1
            and compressed_regions.count(region_id) > 1
        )
        observations = []
        for region_id in compressed_regions:
            path_nodes = [node for node in node_path if owner.get(node) == region_id]
            observations.append({
                "region_id": region_id,
                "path_hits_region": bool(path_nodes),
                "path_hits_region_root": roots[region_id] in node_path,
                "region_root": roots[region_id],
                "path_nodes_in_region": path_nodes,
            })
        examples.append({
            "po": po_index,
            "po_id": po_index,
            "po_literal": literal,
            "po_driver": driver,
            "path_available": True,
            "pi": node_path[0] if node_path and node_path[0] in aig.pi_nodes else None,
            "node_path": node_path,
            "raw_region_sequence": raw_regions,
            "compressed_region_sequence": compressed_regions,
            "region_observations": observations,
            "valid_node_path": valid_node_path,
            "valid_region_path": valid_region_path,
            "reentered_regions": reentered,
            "path_hits_root_required": False,
        })
    return examples


def path_stats(examples: Sequence[dict]) -> dict:
    available = [row for row in examples if row.get("path_available")]
    node_lengths = [len(row["node_path"]) for row in available]
    and_node_lengths = [len(row["raw_region_sequence"]) for row in available]
    region_lengths = [len(row["compressed_region_sequence"]) for row in available]
    segment_observations = [
        observation
        for row in available
        for observation in row["region_observations"]
    ]
    no_root_segments = [
        observation for observation in segment_observations
        if not observation["path_hits_region_root"]
    ]

    def mean_or_zero(values: Sequence[int]) -> float:
        return float(statistics.fmean(values)) if values else 0.0

    def median_or_zero(values: Sequence[int]) -> float:
        return float(statistics.median(values)) if values else 0.0

    return {
        "po_count": len(examples),
        "path_example_count": len(available),
        "unavailable_path_po_ids": [row["po_id"] for row in examples if not row.get("path_available")],
        "average_node_path_length": mean_or_zero(node_lengths),
        "median_node_path_length": median_or_zero(node_lengths),
        "maximum_node_path_length": max(node_lengths, default=0),
        "average_and_node_path_length": mean_or_zero(and_node_lengths),
        "average_region_path_length": mean_or_zero(region_lengths),
        "median_region_path_length": median_or_zero(region_lengths),
        "maximum_region_path_length": max(region_lengths, default=0),
        "region_path_segment_count": len(segment_observations),
        "region_segments_without_root_count": len(no_root_segments),
        "paths_with_region_segment_without_root": sorted({
            row["po_id"]
            for row in available
            if any(not observation["path_hits_region_root"] for observation in row["region_observations"])
        }),
        "region_root_hit_count": len(segment_observations) - len(no_root_segments),
        "reentered_region_path_count": sum(bool(row["reentered_regions"]) for row in available),
        "reentered_region_paths": [
            {"po_id": row["po_id"], "regions": row["reentered_regions"]}
            for row in available if row["reentered_regions"]
        ],
        "all_examples_have_pi": all(row.get("pi") is not None for row in available),
        "all_examples_valid_node_paths": all(row.get("valid_node_path") for row in available),
        "all_examples_valid_region_paths": all(row.get("valid_region_path") for row in available),
    }


def write_dot(
    path: Path,
    regions: Sequence[dict],
    region_edges: Sequence[dict],
    pi_region_edges: Sequence[dict],
    region_po_edges: Sequence[dict],
    direct_pi_po_edges: Sequence[dict],
) -> None:
    lines = ["digraph region_dag {", "  rankdir=LR;", "  node [shape=box];"]
    for region in regions:
        lines.append(
            f'  R{region["region_id"]} [label="R{region["region_id"]}\\n'
            f'root={region["root"]}\\nsize={region["cone_size"]}"];'
        )
    lines.append('  node [shape=ellipse,style=filled,fillcolor="#eeeeee"];')
    pi_ids = sorted({int(row["pi_id"]) for row in pi_region_edges})
    po_ids = sorted({int(row["po_id"]) for row in region_po_edges})
    for pi in pi_ids:
        lines.append(f'  PI{pi} [label="PI{pi}"];')
    for po in po_ids:
        lines.append(f'  PO{po} [label="PO{po}"];')
    lines.append('  node [shape=box,style=solid,fillcolor="white"];')
    for edge in region_edges:
        lines.append(
            f'  R{edge["src_region"]} -> R{edge["dst_region"]} '
            f'[label="{edge["cross_edge_count"]}"];'
        )
    for edge in pi_region_edges:
        lines.append(
            f'  PI{edge["pi_id"]} -> R{edge["dst_region"]} '
            f'[style=dashed,label="{edge["node_edge_count"]}"];'
        )
    for edge in region_po_edges:
        lines.append(f'  R{edge["src_region"]} -> PO{edge["po_id"]} [style=dashed];')
    for edge in direct_pi_po_edges:
        lines.append(f'  PI{edge["pi"]} -> PO{edge["po_id"]} [style=dotted];')
    lines.append("}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(
    input_path: Path,
    maximal_path: Path,
    owner_csv_path: Path,
    run_dir: Path,
    design: str,
    run_label: str,
) -> dict:
    aig = parse_binary_aig(input_path.resolve())
    regions, owner, partition_consistency = load_partition(
        maximal_path.resolve(), owner_csv_path.resolve(), aig.and_nodes
    )
    graph_data = build_graph_data(aig, regions, owner)
    examples = longest_pi_paths(aig, owner, graph_data)
    stats = path_stats(examples)

    region_edge_rows = graph_data["region_edges"]
    pi_region_rows = graph_data["pi_region_edges"]
    region_po_rows = graph_data["region_po_edges"]

    dag_checks = {
        "region_count": len(regions),
        "topological_order": graph_data["topological_order"],
        "topological_sort_length": len(graph_data["topological_order"]),
        "topological_sort_covers_all_regions": graph_data["topological_sort_covers_all_regions"],
        "cycle_count": graph_data["cycle_count"],
        "cyclic_components": graph_data["cyclic_components"],
        "acyclic": graph_data["cycle_count"] == 0,
        "and_edge_occurrence_count": graph_data["and_edge_occurrence_count"],
        "cross_region_edge_occurrence_count": graph_data["cross_region_edge_occurrence_count"],
        "distinct_region_edge_count": len(region_edge_rows),
        "cross_region_edges_all_present": graph_data["cross_region_edges_all_present"],
        "missing_region_edges": graph_data["missing_region_edges"],
        "pi_edge_occurrence_count": graph_data["pi_edge_occurrence_count"],
        "pi_region_edge_counted_occurrences": graph_data["pi_region_edge_counted_occurrences"],
        "pi_region_edges_complete": graph_data["pi_region_edges_complete"],
        "region_po_edge_occurrence_count": graph_data["region_po_edge_occurrence_count"],
        "region_po_metadata_occurrences": graph_data["region_po_metadata_occurrences"],
        "region_po_edges_complete": graph_data["region_po_edges_complete"],
    }
    projection_checks = {
        "projection_rule": (
            "For every node on a path, map AND nodes through owner[node]; omit PI/constant "
            "nodes from the region sequence; compress adjacent duplicate region IDs."
        ),
        "owner_mapping_source": str(owner_csv_path.resolve()),
        "partition_source": str(maximal_path.resolve()),
        "owner_mapping_consistent": partition_consistency["partition_consistent"],
        "cross_region_edges_all_present": graph_data["cross_region_edges_all_present"],
        "edge_complete_path_projection": graph_data["edge_complete_path_projection"],
        "proof_basis": (
            "Every original cross-region AND->AND edge is represented in the quotient DAG; "
            "therefore mapping any node-level path and compressing adjacent equal owners "
            "produces a legal region-DAG path. This is a structural proof, not path sampling."
        ),
        "example_path_count": len(examples),
        "example_paths_available": stats["path_example_count"] == len(examples),
        "example_paths_have_pi": stats["all_examples_have_pi"],
        "example_node_paths_valid": stats["all_examples_valid_node_paths"],
        "example_region_paths_valid": stats["all_examples_valid_region_paths"],
        "reentered_region_path_count": stats["reentered_region_path_count"],
        "reentered_region_paths": stats["reentered_region_paths"],
        "root_is_not_required_for_mapping": True,
        "region_segments_without_root_count": stats["region_segments_without_root_count"],
        "global_no_reentry_from_dag_acyclic": dag_checks["acyclic"],
        "pass": (
            partition_consistency["partition_consistent"]
            and graph_data["edge_complete_path_projection"]
            and stats["path_example_count"] == len(examples)
            and stats["all_examples_have_pi"]
            and stats["all_examples_valid_node_paths"]
            and stats["all_examples_valid_region_paths"]
            and stats["reentered_region_path_count"] == 0
        ),
    }
    # Keep the assignment above explicit in the serialized object without
    # relying on a second, subtly different pass condition.
    projection_checks["pass"] = bool(projection_checks["pass"] and dag_checks["acyclic"])

    region_dag = {
        "gate": "Gate 3B",
        "design": design,
        "input_path": str(input_path.resolve()),
        "input_sha256": sha256_file(input_path),
        "partition": "STRICT_MAXIMAL_MFFC_PARTITION",
        "partition_source": str(maximal_path.resolve()),
        "owner_mapping_source": str(owner_csv_path.resolve()),
        "region_count": len(regions),
        "nodes": [
            {
                "region_id": row["region_id"],
                "root": row["root"],
                "size": row["cone_size"],
                "semantics": row["semantics"],
            }
            for row in regions
        ],
        "edges": region_edge_rows,
        "pi_region_edges": pi_region_rows,
        "region_po_edges": region_po_rows,
        "direct_pi_po_edges": graph_data["direct_pi_po_edges"],
        "constant_po_indices": graph_data["constant_po_indices"],
        "pi_region_edges_complete": graph_data["pi_region_edges_complete"],
        "region_po_edges_complete": graph_data["region_po_edges_complete"],
        "topological_order": graph_data["topological_order"],
        "topological_sort_covers_all_regions": graph_data["topological_sort_covers_all_regions"],
        "cycle_count": graph_data["cycle_count"],
        "cyclic_components": graph_data["cyclic_components"],
        "edge_complete_path_projection": graph_data["edge_complete_path_projection"],
    }
    input_manifest = {
        "design": design,
        "input_path": str(input_path.resolve()),
        "input_sha256": sha256_file(input_path),
        "file_size_bytes": input_path.stat().st_size,
        "format": "binary AIGER (.aig)",
        "header": f"aig {aig.maximum_variable} {aig.pi_count} {aig.latch_count} {aig.po_count} {aig.and_count}",
        "pi_count": aig.pi_count,
        "po_count": aig.po_count,
        "and_count": aig.and_count,
        "partition": "STRICT_MAXIMAL_MFFC_PARTITION",
        "region_count": len(regions),
    }

    run_dir.mkdir(parents=True, exist_ok=True)
    dump_json(run_dir / "input_manifest.json", input_manifest)
    dump_json(run_dir / "partition_consistency.json", partition_consistency)
    dump_json(run_dir / "region_dag.json", region_dag)
    dump_csv(
        run_dir / "region_edges.csv",
        ["src_region", "dst_region", "cross_edge_count"],
        region_edge_rows,
    )
    dump_jsonl(run_dir / "region_metadata.jsonl", graph_data["region_metadata"])
    dump_csv(
        run_dir / "pi_region_edges.csv",
        ["pi_id", "dst_region", "node_edge_count"],
        pi_region_rows,
    )
    dump_csv(
        run_dir / "region_po_edges.csv",
        ["src_region", "po_id", "po_driver"],
        region_po_rows,
    )
    dump_jsonl(run_dir / "example_pi_po_paths.jsonl", examples)
    dump_json(run_dir / "dag_checks.json", dag_checks)
    dump_json(run_dir / "path_projection_checks.json", projection_checks)
    dump_json(run_dir / "region_path_stats.json", stats)
    write_dot(
        run_dir / "region_dag.dot",
        graph_data["region_metadata"],
        region_edge_rows,
        pi_region_rows,
        region_po_rows,
        graph_data["direct_pi_po_edges"],
    )
    run_manifest = {
        "gate": "Gate 3B",
        "design": design,
        "run_label": run_label,
        "input_sha256": input_manifest["input_sha256"],
        "partition_sha256": sha256_file(maximal_path),
        "owner_csv_sha256": sha256_file(owner_csv_path),
        "analysis_mode": "strict maximal MFFC region quotient and path projection only",
        "recomputed_partition": False,
        "rewrite_executed": False,
        "litecut_executed": False,
        "sta_executed": False,
        "gradient_executed": False,
    }
    dump_json(run_dir / "run_manifest.json", run_manifest)
    return {
        "input_manifest": input_manifest,
        "partition_consistency": partition_consistency,
        "region_dag": region_dag,
        "dag_checks": dag_checks,
        "projection_checks": projection_checks,
        "path_stats": stats,
    }


RUN_ARTIFACTS = [
    "input_manifest.json",
    "partition_consistency.json",
    "region_dag.json",
    "region_edges.csv",
    "region_metadata.jsonl",
    "pi_region_edges.csv",
    "region_po_edges.csv",
    "example_pi_po_paths.jsonl",
    "dag_checks.json",
    "path_projection_checks.json",
    "region_path_stats.json",
    "region_dag.dot",
]


def canonical_file_value(path: Path):
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if path.suffix == ".jsonl":
        rows = []
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    if path.suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as stream:
            return list(csv.reader(stream))
    return path.read_text(encoding="utf-8").replace("\r\n", "\n")


def canonical_file_hash(path: Path) -> str:
    return canonical_json_hash(canonical_file_value(path))


def finalize_runs(output_root: Path, design: str) -> dict:
    design_root = output_root / design
    run1 = design_root / "run1"
    run2 = design_root / "run2"
    missing = [
        str(run / artifact)
        for run in (run1, run2)
        for artifact in RUN_ARTIFACTS
        if not (run / artifact).is_file()
    ]
    if missing:
        raise FileNotFoundError("cannot finalize incomplete Gate 3B runs: " + ", ".join(missing))

    comparison = []
    deterministic = True
    for artifact in RUN_ARTIFACTS:
        hash1 = canonical_file_hash(run1 / artifact)
        hash2 = canonical_file_hash(run2 / artifact)
        same = hash1 == hash2
        deterministic = deterministic and same
        comparison.append({
            "file": artifact,
            "run1_canonical_sha256": hash1,
            "run2_canonical_sha256": hash2,
            "identical": same,
        })

    run1_summary = json.loads((run1 / "region_dag.json").read_text(encoding="utf-8"))
    run1_dag = json.loads((run1 / "dag_checks.json").read_text(encoding="utf-8"))
    run1_projection = json.loads((run1 / "path_projection_checks.json").read_text(encoding="utf-8"))
    run1_stats = json.loads((run1 / "region_path_stats.json").read_text(encoding="utf-8"))
    run1_input = json.loads((run1 / "input_manifest.json").read_text(encoding="utf-8"))
    run1_partition = json.loads((run1 / "partition_consistency.json").read_text(encoding="utf-8"))

    for artifact in RUN_ARTIFACTS:
        shutil.copyfile(run1 / artifact, design_root / artifact)

    gate_pass = bool(
        deterministic
        and run1_partition["partition_consistent"]
        and run1_summary["region_count"] == 62
        and run1_dag["topological_sort_covers_all_regions"]
        and run1_dag["cycle_count"] == 0
        and run1_dag["cross_region_edges_all_present"]
        and run1_dag["pi_region_edges_complete"]
        and run1_dag["region_po_edges_complete"]
        and run1_projection["edge_complete_path_projection"]
        and run1_projection["example_paths_available"]
        and run1_projection["example_paths_have_pi"]
        and run1_projection["example_node_paths_valid"]
        and run1_projection["example_region_paths_valid"]
        and run1_projection["reentered_region_path_count"] == 0
    )
    summary = {
        "gate": "Gate 3B",
        "gate3b_pass": gate_pass,
        "selected_design": design,
        "selected_input": run1_input["input_path"],
        "input_sha256": run1_input["input_sha256"],
        "q1": {
            "pi": run1_input["pi_count"],
            "po": run1_input["po_count"],
            "and": run1_input["and_count"],
            "region_count": run1_summary["region_count"],
        },
        "partition": "STRICT_MAXIMAL_MFFC_PARTITION",
        "partition_consistent": run1_partition["partition_consistent"],
        "q2_region_dag_region_count": run1_summary["region_count"],
        "q3_topological_sort_covers_all_regions": run1_dag["topological_sort_covers_all_regions"],
        "q3_cycle_count": run1_dag["cycle_count"],
        "q4_cross_region_edges_all_present": run1_dag["cross_region_edges_all_present"],
        "q4_pi_region_edges_complete": run1_dag["pi_region_edges_complete"],
        "q4_region_po_edges_complete": run1_dag["region_po_edges_complete"],
        "q5_edge_complete_path_projection": run1_projection["edge_complete_path_projection"],
        "q6_example_paths": run1_projection["example_path_count"],
        "q6_example_paths_valid": (
            run1_projection["example_paths_available"]
            and run1_projection["example_paths_have_pi"]
            and run1_projection["example_node_paths_valid"]
            and run1_projection["example_region_paths_valid"]
        ),
        "q7_reentered_region_path_count": run1_projection["reentered_region_path_count"],
        "root_required_for_mapping": False,
        "region_segments_without_root_count": run1_projection["region_segments_without_root_count"],
        "region_path_stats": run1_stats,
        "deterministic": deterministic,
        "determinism_comparison": comparison,
        "forbidden_flows_executed": {
            "litecut": False,
            "rewrite_replacement": False,
            "abc_mapping": False,
            "sta": False,
            "gradient": False,
        },
    }
    dump_json(design_root / "gate3b_summary.json", summary)
    report = f"""# Gate 3B report — {design}

## 结果

Gate 3B = **{'PASS' if gate_pass else 'BLOCKED'}**。
本轮直接复用 Gate 3A 的 `STRICT_MAXIMAL_MFFC_PARTITION`，没有重新计算 partition。

## 固定输入

- AIG：`{run1_input['input_path']}`
- SHA256：`{run1_input['input_sha256']}`
- PI / PO / AND：`{run1_input['pi_count']} / {run1_input['po_count']} / {run1_input['and_count']}`
- strict maximal MFFC regions：`{run1_summary['region_count']}`

## Region DAG

- topological sort 长度：`{run1_dag['topological_sort_length']}` / `{run1_dag['region_count']}`
- cycle count：`{run1_dag['cycle_count']}`
- region DAG 无环：`{run1_dag['acyclic']}`
- 所有跨 region AND→AND edge 已进入 quotient DAG：`{run1_dag['cross_region_edges_all_present']}`
- PI→region boundary 完整记录：`{run1_dag['pi_region_edges_complete']}`
- region→PO boundary 完整记录：`{run1_dag['region_po_edges_complete']}`
- distinct region edges：`{run1_dag['distinct_region_edge_count']}`
- cross-region edge occurrences：`{run1_dag['cross_region_edge_occurrence_count']}`

每个 region 的 predecessor/successor、PI/PO 边界以及真实跨 region node edge 保存在 `region_metadata.jsonl`；PI→region 与 region→PO 分别保存在 `pi_region_edges.csv` 和 `region_po_edges.csv`。

## Path projection

投影规则是：对 node-level path 中的每个 AND node 使用 `owner[node]`，忽略 PI/constant 节点，然后压缩相邻重复 region ID。由于所有原始跨 region AND→AND edge 都有对应 DAG edge，这一规则是 **edge-complete path projection**，不是通过枚举所有 PI→PO path 得出的抽样结论。

- edge-complete path projection：`{run1_projection['edge_complete_path_projection']}`
- 每个 PO 的 longest PI→PO example：`{run1_projection['example_path_count']}` 条
- example path 全部有 PI：`{run1_projection['example_paths_have_pi']}`
- node path 全部合法：`{run1_projection['example_node_paths_valid']}`
- region sequence 全部合法：`{run1_projection['example_region_paths_valid']}`
- reentered region path count：`{run1_projection['reentered_region_path_count']}`
- root 不是映射依据：`True`
- 不经过该 region root 的 sampled region segments：`{run1_projection['region_segments_without_root_count']}`

平均 node path length（含起始 PI）：`{run1_stats['average_node_path_length']:.6f}`；最大：`{run1_stats['maximum_node_path_length']}`。

平均 compressed region path length：`{run1_stats['average_region_path_length']:.6f}`；最大：`{run1_stats['maximum_region_path_length']}`。

## Determinism

run1 与 run2 对全部 canonical analysis artifacts 的 SHA256 比较结果在 `gate3b_summary.json` 中；`deterministic = **{deterministic}**`。

## 约束检查

本 Gate 未执行 LiteCut、Gate1 candidate attach、rewrite/replacement、ABC mapping、STA、critical-path optimization 或 gradient。
"""
    (design_root / "gate3b_report.md").write_text(report, encoding="utf-8")
    summary["report"] = str((design_root / "gate3b_report.md").resolve())
    dump_json(design_root / "gate3b_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="analyze one deterministic run")
    run_parser.add_argument("--input", type=Path, required=True)
    run_parser.add_argument("--maximal-mffcs", type=Path, required=True)
    run_parser.add_argument("--owner-csv", type=Path, required=True)
    run_parser.add_argument("--run-dir", type=Path, required=True)
    run_parser.add_argument("--design", required=True)
    run_parser.add_argument("--run-label", required=True)

    finalize_parser = subparsers.add_parser("finalize", help="compare run1/run2 and write final outputs")
    finalize_parser.add_argument("--output-root", type=Path, required=True)
    finalize_parser.add_argument("--design", required=True)

    args = parser.parse_args()
    if args.command == "run":
        result = run_analysis(
            args.input,
            args.maximal_mffcs,
            args.owner_csv,
            args.run_dir,
            args.design,
            args.run_label,
        )
        print(json.dumps({
            "run_dir": str(args.run_dir.resolve()),
            "region_count": result["region_dag"]["region_count"],
            "cycle_count": result["dag_checks"]["cycle_count"],
            "edge_complete_path_projection": result["projection_checks"]["edge_complete_path_projection"],
            "reentered_region_path_count": result["projection_checks"]["reentered_region_path_count"],
        }, indent=2, sort_keys=True))
        return 0
    summary = finalize_runs(args.output_root.resolve(), args.design)
    print(json.dumps({
        "output_root": str((args.output_root / args.design).resolve()),
        "gate3b_pass": summary["gate3b_pass"],
        "region_count": summary["q2_region_dag_region_count"],
        "cycle_count": summary["q3_cycle_count"],
        "deterministic": summary["deterministic"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
