"""Exact strashed Binary AIGER replay, conflict resolution, and global PI->PO sensitivity."""
from __future__ import annotations

import math
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .level_weights import Levels, propagate_levels
from .structure import parse_aig, references


def _write_varint(buf: bytearray, value: int) -> None:
    if value < 0:
        raise ValueError(f"Negative AIGER varint delta: {value}")
    while value >= 0x80:
        buf.append((value & 0x7F) | 0x80)
        value >>= 7
    buf.append(value)


def write_binary_aig(path: Path, pi_count: int, po_literals: list[int],
                     fanin_literals: dict[int, tuple[int, int]]) -> None:
    """Write a combinational Binary AIGER file (`aig M I 0 O A`)."""
    and_count = len(fanin_literals)
    max_var = pi_count + and_count
    header = f"aig {max_var} {pi_count} 0 {len(po_literals)} {and_count}\n".encode("ascii")
    out = bytearray(header)
    for lit in po_literals:
        out.extend(f"{lit}\n".encode("ascii"))
    for node in range(pi_count + 1, max_var + 1):
        l0, l1 = fanin_literals[node]
        rhs0, rhs1 = (l0, l1) if l0 >= l1 else (l1, l0)
        lhs = 2 * node
        if rhs0 >= lhs or rhs1 < 0:
            raise ValueError(f"Non-topological or negative fanin at node {node}: ({rhs0}, {rhs1})")
        _write_varint(out, lhs - rhs0)
        _write_varint(out, rhs0 - rhs1)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(out))


class StrashedAigBuilder:
    """Incremental strashed combinational AIG builder with dead-node sweep."""

    def __init__(self, pi_count: int):
        self.pi_count = pi_count
        self.next_var = pi_count + 1
        self.nodes: list[tuple[int, int]] = []  # index = var - (pi_count + 1)
        self.strash: dict[tuple[int, int], int] = {}

    def and_lit(self, lit0: int, lit1: int) -> int:
        if lit0 == 0 or lit1 == 0 or lit0 == (lit1 ^ 1):
            return 0
        if lit0 == 1 or lit0 == lit1:
            return lit1
        if lit1 == 1:
            return lit0
        pair = (lit1, lit0) if lit1 < lit0 else (lit0, lit1)
        var = self.strash.get(pair)
        if var is not None:
            return var << 1
        var = self.next_var
        self.next_var += 1
        self.nodes.append((pair[1], pair[0]))  # store larger first (AIGER convention)
        self.strash[pair] = var
        return var << 1

    def build_and_sweep(self, po_literals: list[int]) -> tuple[list[int], dict[int, tuple[int, int]]]:
        """Keep only nodes reachable from POs and compactly renumber in topological order."""
        needed: set[int] = set()
        stack = [lit >> 1 for lit in po_literals if (lit >> 1) > self.pi_count]
        while stack:
            v = stack.pop()
            if v in needed or v <= self.pi_count:
                continue
            needed.add(v)
            l0, l1 = self.nodes[v - self.pi_count - 1]
            if (l0 >> 1) > self.pi_count:
                stack.append(l0 >> 1)
            if (l1 >> 1) > self.pi_count:
                stack.append(l1 >> 1)

        remap: dict[int, int] = {i: i for i in range(self.pi_count + 1)}
        new_fanins: dict[int, tuple[int, int]] = {}
        new_var = self.pi_count + 1
        for old_var in sorted(needed):
            l0, l1 = self.nodes[old_var - self.pi_count - 1]
            nl0 = (remap[l0 >> 1] << 1) ^ (l0 & 1)
            nl1 = (remap[l1 >> 1] << 1) ^ (l1 & 1)
            remap[old_var] = new_var
            new_fanins[new_var] = (nl0, nl1) if nl0 >= nl1 else (nl1, nl0)
            new_var += 1
        new_pos = [(remap[lit >> 1] << 1) ^ (lit & 1) for lit in po_literals]
        return new_pos, new_fanins


def bounded_mffc_nodes(root: int, leaves: Iterable[int], aig, fanouts: dict[int, int]) -> tuple[frozenset[int], frozenset[int]]:
    """Return (all_removable_nodes_including_root, strict_internal_nodes_excluding_root)."""
    boundary = set(leaves)
    remaining: dict[int, int] = {}
    all_nodes: set[int] = set()
    pending = [root]
    while pending:
        node = pending.pop()
        if node in boundary or node <= aig.pi_count:
            continue
        all_nodes.add(node)
        for lit in aig.fanin_literals[node]:
            src = lit >> 1
            if src in boundary:
                continue
            val = remaining.get(src, fanouts[src]) - 1
            remaining[src] = val
            if val == 0:
                pending.append(src)
    strict_internal = all_nodes - {root}
    return frozenset(all_nodes), frozenset(strict_internal)


def original_cone_truth(aig, root: int, leaves: list[int]) -> int:
    """Compute the 16-bit truth table of `root` in terms of the 4 ordered `leaves`."""
    if len(leaves) != 4 or len(set(leaves)) != 4:
        raise ValueError(f"Expected 4 distinct leaves, got {leaves}")
    pats = [0xAAAA, 0xCCCC, 0xF0F0, 0xFF00]
    vals = {lf: pats[i] for i, lf in enumerate(leaves)}
    vals[0] = 0
    pending = [root]
    interior = set()
    while pending:
        n = pending.pop()
        if n in vals or n in interior:
            continue
        if n <= aig.pi_count:
            raise ValueError(f"Cut {leaves} at root {root} leaks to PI {n}")
        interior.add(n)
        for lit in aig.fanin_literals[n]:
            pending.append(lit >> 1)
    for n in sorted(interior):
        l0, l1 = aig.fanin_literals[n]
        v0 = vals[l0 >> 1] ^ (0xFFFF if (l0 & 1) else 0)
        v1 = vals[l1 >> 1] ^ (0xFFFF if (l1 & 1) else 0)
        vals[n] = v0 & v1
    return vals[root]


def candidate_output_complement(orig_truth: int, leaves_compl: list[int], cand: dict) -> int:
    """Verify candidate matches `orig_truth` (or its complement) and return extra flip bit (0 or 1)."""
    pats = [0xAAAA, 0xCCCC, 0xF0F0, 0xFF00]
    cv = [pats[i] ^ (0xFFFF if leaves_compl[i] else 0) for i in range(4)]
    for e0n, e0c, e1n, e1c in cand["edges"]:
        v0 = cv[e0n] ^ (0xFFFF if e0c else 0)
        v1 = cv[e1n] ^ (0xFFFF if e1c else 0)
        cv.append(v0 & v1)
    if cand["n_size"] == 0:
        raw = 0xFFFF
    else:
        raw = cv[cand["root_node"]]
    out_tt = raw ^ (0xFFFF if cand["root_compl"] else 0)
    if out_tt == orig_truth:
        return 0
    if out_tt == (orig_truth ^ 0xFFFF):
        return 1
    raise ValueError(f"Candidate subgraph {cand.get('subgraph')} truth {out_tt:#06x} != orig {orig_truth:#06x}")


@dataclass(frozen=True)
class ReplacementOp:
    root: int
    leaves: tuple[int, ...]
    leaves_compl: tuple[int, ...]
    cand: dict
    rem_all: frozenset[int]
    rem_strict: frozenset[int]
    flip: int


def select_nonconflicting(candidates_with_priority: list[tuple[tuple, ReplacementOp]]) -> list[ReplacementOp]:
    """Greedy non-conflicting selection sorted by descending priority key (higher first).

    Two replacements conflict iff:
    - same root, OR
    - either's strictly internal deleted nodes touch the other's cone or leaves.
    Note: an upstream replaced `root_1` CAN appear in `leaves_2` because `root_1`
    remains a valid boundary node whose function is mapped to its new literal.
    """
    ordered = sorted(candidates_with_priority, key=lambda x: x[0], reverse=True)
    selected: list[ReplacementOp] = []
    used_roots: set[int] = set()
    deleted_strict: set[int] = set()
    protected_nodes: set[int] = set()  # all nodes referenced as leaves or cones of selected ops

    for _, op in ordered:
        if op.root in used_roots or op.root in deleted_strict:
            continue
        if op.rem_strict & protected_nodes:
            continue
        if op.rem_all & deleted_strict:
            continue
        if set(op.leaves) & deleted_strict:
            continue
        # Double check against already-selected ops' leaves/cones
        selected.append(op)
        used_roots.add(op.root)
        deleted_strict.update(op.rem_strict)
        protected_nodes.update(op.rem_all)
        protected_nodes.update(op.leaves)
    return selected


def apply_replacements(aig, ops: list[ReplacementOp]) -> tuple[list[int], dict[int, tuple[int, int]]]:
    """Apply a verified non-conflicting set of replacements to `aig` and return swept (po_lits, fanins)."""
    by_root = {op.root: op for op in ops}
    if len(by_root) != len(ops):
        raise ValueError("Duplicate root in replacement ops")
    deleted_strict: set[int] = set()
    for op in ops:
        deleted_strict.update(op.rem_strict)
    for op in ops:
        if op.root in deleted_strict or (set(op.leaves) & deleted_strict):
            raise ValueError(f"Conflicting ops passed to apply_replacements at root {op.root}")

    builder = StrashedAigBuilder(aig.pi_count)
    lit_map: dict[int, int] = {0: 0}
    for pi in range(1, aig.pi_count + 1):
        lit_map[pi] = pi << 1

    for node in range(aig.pi_count + 1, aig.maximum_variable + 1):
        if node in by_root:
            op = by_root[node]
            cand = op.cand
            wire = [lit_map[lf] ^ c for lf, c in zip(op.leaves, op.leaves_compl)]
            for e0n, e0c, e1n, e1c in cand["edges"]:
                l0 = wire[e0n] ^ e0c
                l1 = wire[e1n] ^ e1c
                wire.append(builder.and_lit(l0, l1))
            if cand["n_size"] == 0:
                raw_lit = 1  # ABC Dec_GraphIsConst base is const1
            else:
                raw_lit = wire[cand["root_node"]]
            lit_map[node] = raw_lit ^ cand["root_compl"] ^ op.flip
        elif node in deleted_strict:
            # Strictly internal node of a replaced cone; no surviving node may reference it.
            continue
        else:
            l0, l1 = aig.fanin_literals[node]
            nl0 = lit_map[l0 >> 1] ^ (l0 & 1)
            nl1 = lit_map[l1 >> 1] ^ (l1 & 1)
            lit_map[node] = builder.and_lit(nl0, nl1)

    remapped_pos = [lit_map[lit >> 1] ^ (lit & 1) for lit in aig.po_literals]
    return builder.build_and_sweep(remapped_pos)


def apply_and_write_aig(aig, ops: list[ReplacementOp], out_path: Path):
    """Apply ops, write Binary AIGER, parse back, and return parsed AIG."""
    po_lits, fanins = apply_replacements(aig, ops)
    write_binary_aig(out_path, aig.pi_count, po_lits, fanins)
    return parse_aig(out_path)


@dataclass
class DelayProfile:
    """Continuous fanout-aware arrival/required/depth profile duck-typing `Levels`."""

    level: list[float]
    required: list[float | None]
    depth: float
    gate_delay: list[float]
    fanouts: Counter[int]

    def slack(self, node: int) -> float | None:
        req = self.required[node]
        return None if req is None else req - self.level[node]


def propagate_fanout_delays(
    aig,
    fanouts: Counter[int] | None = None,
    fanout_alpha: float = 0.0,
    pi_fanout_alpha: float = 0.0,
) -> DelayProfile:
    """Propagate arrival/required delays with linear fanout load `(1 + alpha * max(0, fo - 1))`.

    With `fanout_alpha=0.0` and `pi_fanout_alpha=0.0`, arrivals and required bounds
    match `propagate_levels(aig)` numerically.
    """
    if fanouts is None:
        fanouts = references(aig)
    n_vars = aig.maximum_variable + 1
    gate_delay = [0.0] * n_vars
    if pi_fanout_alpha > 0.0:
        for node in range(1, aig.pi_count + 1):
            gate_delay[node] = pi_fanout_alpha * max(0, fanouts[node] - 1)
    level = list(gate_delay)
    for node in range(aig.pi_count + 1, n_vars):
        d = 1.0 + fanout_alpha * max(0, fanouts[node] - 1)
        gate_delay[node] = d
        l0, l1 = aig.fanin_literals[node]
        a0, a1 = level[l0 >> 1], level[l1 >> 1]
        level[node] = d + (a0 if a0 >= a1 else a1)

    outputs = [lit >> 1 for lit in aig.po_literals]
    depth = max((level[node] for node in outputs), default=0.0)
    required: list[float | None] = [None] * n_vars
    for node in outputs:
        required[node] = depth
    for node in range(aig.maximum_variable, aig.pi_count, -1):
        req_node = required[node]
        if req_node is None:
            continue
        bound = req_node - gate_delay[node]
        for lit in aig.fanin_literals[node]:
            pred = lit >> 1
            prev = required[pred]
            if prev is None or bound < prev:
                required[pred] = bound
    return DelayProfile(level, required, depth, gate_delay, fanouts)


def original_cone_timing_and_refs(
    aig,
    root: int,
    leaves: list[int],
    delay: DelayProfile,
    rem_all: frozenset[int],
    sens: list[float],
    tau: float = 1.0,
) -> tuple[tuple[float | None, ...], tuple[int, ...], tuple[float, ...]]:
    """Return `(leaf_path_delays, orig_leaf_refs_in_mffc, external_leaf_sensitivities)`."""
    boundary = set(leaves)
    lowest = min(leaves)
    interior: set[int] = set()
    pending = [root]
    while pending:
        node = pending.pop()
        if node in boundary or node < lowest or node <= aig.pi_count or node in interior:
            continue
        interior.add(node)
        for lit in aig.fanin_literals[node]:
            pending.append(lit >> 1)

    distances: dict[int, float] = {root: delay.gate_delay[root]}
    for node in sorted(interior, reverse=True):
        d_base = distances.pop(node)
        for lit in aig.fanin_literals[node]:
            pred = lit >> 1
            d_step = d_base + (delay.gate_delay[pred] if pred in interior else 0.0)
            if d_step > distances.get(pred, -1.0):
                distances[pred] = d_step
    leaf_delays = tuple(distances.get(lf) for lf in leaves)

    leaf_idx = {lf: idx for idx, lf in enumerate(leaves)}
    orig_refs = [0] * len(leaves)
    for node in rem_all:
        for lit in aig.fanin_literals[node]:
            idx = leaf_idx.get(lit >> 1)
            if idx is not None:
                orig_refs[idx] += 1

    local_sens: dict[int, float] = {root: sens[root]}
    for node in sorted(interior, reverse=True):
        su = local_sens.get(node, 0.0)
        if su == 0.0:
            continue
        l0, l1 = aig.fanin_literals[node]
        n0, n1 = l0 >> 1, l1 >> 1
        lv0, lv1 = delay.level[n0], delay.level[n1]
        m = lv0 if lv0 >= lv1 else lv1
        w0 = math.exp((lv0 - m) / tau)
        w1 = math.exp((lv1 - m) / tau)
        denom = w0 + w1
        local_sens[n0] = local_sens.get(n0, 0.0) + su * (w0 / denom)
        local_sens[n1] = local_sens.get(n1, 0.0) + su * (w1 / denom)

    sens_other = tuple(max(0.0, sens[lf] - local_sens.get(lf, 0.0)) for lf in leaves)
    return leaf_delays, tuple(orig_refs), sens_other


def global_po_sensitivity(aig, levels: Levels | DelayProfile, tau: float = 1.0) -> tuple[float, list[float]]:
    """O(|V|) forward/backward pass aggregating all PI->PO paths into per-node global sensitivity.

    Global objective: Phi = sum_{po} exp((level[po] - D) / tau).
    Backward split at each AND node distributes sensitivity across its two fanins
    proportional to exp((level[fanin_i] - max_fanin_level) / tau).
    """
    depth = levels.depth
    sens = [0.0] * (aig.maximum_variable + 1)
    phi = 0.0
    for lit in aig.po_literals:
        drv = lit >> 1
        w = math.exp((levels.level[drv] - depth) / tau)
        phi += w
        sens[drv] += w

    for node in range(aig.maximum_variable, aig.pi_count, -1):
        s = sens[node]
        if s == 0.0:
            continue
        l0, l1 = aig.fanin_literals[node]
        n0, n1 = l0 >> 1, l1 >> 1
        lv0, lv1 = levels.level[n0], levels.level[n1]
        m = lv0 if lv0 >= lv1 else lv1
        w0 = math.exp((lv0 - m) / tau)
        w1 = math.exp((lv1 - m) / tau)
        denom = w0 + w1
        sens[n0] += s * (w0 / denom)
        sens[n1] += s * (w1 / denom)
    return phi, sens


def output_tns_metrics(aig, levels: Levels, ref_depth: int, ks: tuple[int, ...] = (1, 2, 3)) -> dict:
    drivers = [lit >> 1 for lit in aig.po_literals]
    return {
        "depth": levels.depth,
        "ands": aig.and_count,
        "tns": {k: sum(max(0, levels.level[d] - (ref_depth - k)) for d in drivers) for k in ks},
        "over": {k: sum(1 for d in drivers if levels.level[d] > ref_depth - k) for k in ks},
    }


def verify_cec(abc_bin: Path, orig_aig: Path, new_aig: Path, log_path: Path, timeout: int = 300) -> bool:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = f"read_aiger {orig_aig}; cec -n {new_aig}"
    proc = subprocess.run([str(abc_bin), "-c", cmd], capture_output=True, text=True, timeout=timeout)
    text = proc.stdout + proc.stderr
    log_path.write_text(text)
    low = text.lower()
    return proc.returncode == 0 and ("equivalent" in low) and ("not equivalent" not in low)
