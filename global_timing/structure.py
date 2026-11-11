"""Strict maximal MFFC partition and port-aware structural Region DAG.

The sparse dereference traversal is equivalent to Gate3A's full refcount copy.
Reverse topological roots avoid enumerating every nested cone. The historical
all-root oracle is retained as an independent regression check.
"""
from __future__ import annotations

from collections import Counter
import csv
from dataclasses import dataclass
import heapq
import hashlib
import struct
from pathlib import Path

from .common import legacy, save_json


@dataclass
class Partition:
    aig: object
    regions: list[dict]
    owner: dict[int, int]
    edges: list[tuple[int, int]]
    order: list[int]
    # Each root is a single output; every boundary port has its own transfer.
    transfers: list[dict[int, int]]

    def write(self, output: Path):
        output.mkdir(parents=True, exist_ok=True)
        save_json(output / 'partition.json', {'regions': self.regions, 'edges': self.edges,
                  'topological_order': self.order, 'port_structural_depths': self.transfers,
                  'metric': 'AND depth, not mapped timing'})
        with (output / 'owners.csv').open('x', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['node_id', 'region_id', 'region_root', 'region_type', 'owner_count'])
            for node, region in sorted(self.owner.items()):
                writer.writerow([node, region, self.regions[region]['root'], 'STANDARD_MFFC', 1])


def parse_aig(path: Path):
    aig = legacy('gate3b_analyze').parse_binary_aig(path)
    if min(aig.pi_count, aig.and_count, aig.po_count, aig.maximum_variable) < 0:
        raise ValueError('Negative AIG header count.')
    if any(lit < 0 or lit > 2 * aig.maximum_variable + 1 for lit in aig.po_literals):
        raise ValueError('PO literal is outside the input AIG.')
    return aig


def aig_structure_digest(path: Path):
    """Hash exact node IDs, PI/PO order and literals, excluding symbols/comments.

    This is not a hash of Boolean equivalence or graph isomorphism. Raw file
    SHA256 remains separately recorded, including ABC's timestamp comments.
    """
    aig = parse_aig(path)
    checksum = hashlib.sha256(struct.pack('<QQQQ', aig.maximum_variable, aig.pi_count,
                                          aig.po_count, aig.and_count))
    for literal in aig.po_literals:
        checksum.update(struct.pack('<Q', literal))
    for node, (left, right) in sorted(aig.fanin_literals.items()):
        checksum.update(struct.pack('<QQQ', node, left, right))
    return checksum.hexdigest()


def references(aig):
    refs = Counter(lit >> 1 for fanins in aig.fanin_literals.values() for lit in fanins)
    refs.update(lit >> 1 for lit in aig.po_literals)
    return refs


def dereference(root, aig, refs):
    remaining = {}
    members = {root}
    pending = [root]
    while pending:
        node = pending.pop()
        for lit in aig.fanin_literals[node]:
            source = lit >> 1
            remaining[source] = remaining.get(source, refs[source]) - 1
            if remaining[source] < 0:
                raise ValueError('Reference count underflow.')
            if source > aig.pi_count and remaining[source] == 0:
                members.add(source)
                pending.append(source)
    return members


def make_partition(aig) -> Partition:
    refs = references(aig)
    assigned = set()
    cones = []
    for root in sorted(aig.and_nodes, reverse=True):
        if root in assigned:
            continue
        members = dereference(root, aig, refs)
        if assigned & members:
            raise ValueError('Maximal MFFC overlap: refusing an invalid partition.')
        cones.append((root, sorted(members)))
        assigned.update(members)
    if assigned != aig.and_nodes:
        raise ValueError('Partition misses AND nodes.')
    regions = [{'region_id': i, 'root': root, 'internal_nodes': members,
                'cone_size': len(members), 'semantics': 'STANDARD_MFFC'}
               for i, (root, members) in enumerate(sorted(cones))]
    owner = {node: r['region_id'] for r in regions for node in r['internal_nodes']}
    edge_set = set()
    transfers = []
    for region in regions:
        rid = region['region_id']
        distances = {}
        for node in region['internal_nodes']:
            paths = {}
            for literal in aig.fanin_literals[node]:
                source = literal >> 1
                if owner.get(source) == rid:
                    source_paths = distances[source]
                else:
                    source_paths = {source: 0}
                    if source in owner:
                        if regions[owner[source]]['root'] != source:
                            raise ValueError('A maximal MFFC exposes a non-root node.')
                        edge_set.add((owner[source], rid))
                for port, length in source_paths.items():
                    paths[port] = max(paths.get(port, 0), length + 1)
            distances[node] = paths
        transfers.append(distances[region['root']])
    successors = {r['region_id']: [] for r in regions}
    indegree = Counter(dst for _, dst in edge_set)
    for src, dst in sorted(edge_set):
        successors[src].append(dst)
    ready = [r for r in successors if indegree[r] == 0]
    heapq.heapify(ready)
    order = []
    while ready:
        region = heapq.heappop(ready)
        order.append(region)
        for successor in successors[region]:
            indegree[successor] -= 1
            if indegree[successor] == 0:
                heapq.heappush(ready, successor)
    if len(order) != len(regions):
        raise ValueError('Region graph is cyclic.')
    for lit in aig.po_literals:
        node = lit >> 1
        if node in owner and regions[owner[node]]['root'] != node:
            raise ValueError('A PO exposes a non-root region member.')
    return Partition(aig, regions, owner, sorted(edge_set), order, transfers)
