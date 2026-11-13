"""Canonical rewrite-library and Boolean checks without loading an ML runtime.

The parser is copied verbatim from projects/litecut/gate2a_candidate_inference.py;
truth checks are shared by gradient and non-ML baselines. Original sources stay
unchanged and full circuit CEC still verifies every installed combination.
"""
from pathlib import Path
import re
from typing import Any

from .truth import ContextProofRequired, support_cut_truth


def parse_library(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"LIBRARY\[\] = \{(.*?)\};", text, re.S)
    if not match:
        raise ValueError(f"LIBRARY array not found in {path}")
    values = [int(x.strip()) for x in match.group(1).split(",") if x.strip()]
    cur = 0
    p_phases = values[cur:cur + 65536]; cur += 65536
    p_perms = values[cur:cur + 65536]; cur += 65536
    perms4 = [values[cur + 4 * i:cur + 4 * (i + 1)] for i in range(24)]; cur += 96
    p_map = values[cur:cur + 65536]; cur += 65536
    n_nodes = values[cur:cur + 222]; cur += 222
    n_subgraphs = values[cur:cur + 222]; cur += 222
    classes: dict[int, dict[str, Any]] = {}
    for cls in range(222):
        fanins = []
        for _ in range(n_nodes[cls]):
            fanins.append(tuple(values[cur:cur + 4]))
            cur += 4
        roots = values[cur:cur + n_subgraphs[cls]]
        cur += n_subgraphs[cls]
        classes[cls] = {"fanins": fanins, "roots": roots}
    if cur != len(values):
        raise ValueError(f"Library parse ended at {cur}, expected {len(values)}")
    return {"p_phases": p_phases, "p_perms": p_perms, "perms4": perms4,
            "p_map": p_map, "classes": classes}


def verify_library_truth(decision, subgraph, library):
    """Exhaustive four-input truth check, allowing the engine's output phase.

    Engine ReplaceSubgr aligns zero-input phases when installing a DAG. The
    cut truth and complement are the only admissible library functions.
    End-to-end CEC independently checks actual installation and global phase.
    """
    truth = int(decision['truth']) & 0xffff
    phase = library['p_phases'][truth]
    permutation = library['perms4'][library['p_perms'][truth]]
    data = library['classes'][decision['truth_class']]
    patterns = [0xaaaa, 0xcccc, 0xf0f0, 0xff00]
    values = [(patterns[permutation[i]] ^ (0xffff if (phase >> i) & 1 else 0)) for i in range(4)]
    for a, b, ca, cb in data['fanins']:
        values.append((values[a] ^ (0xffff if ca else 0)) & (values[b] ^ (0xffff if cb else 0)))
    actual = values[data['roots'][subgraph]]
    if actual not in (truth, truth ^ 0xffff):
        raise ValueError(f'Library candidate function mismatch at {decision["root"]}:{decision["cut_slot"]}:{subgraph}')
    return {'truth_verified': True, 'output_phase_complement': actual != truth}


def verify_cut_truth(aig, decision):
    """Validate the exported cut's boundary and truth against original AIG IDs."""
    leaves = decision['cut_leaves']
    if len(leaves) not in {3, 4} or len(set(leaves)) != len(leaves):
        raise ValueError('Expected distinct 3/4-leaf rewrite cut.')
    values = {node: pattern for node, pattern in zip(leaves, [0xaaaa, 0xcccc, 0xf0f0, 0xff00])}
    values.setdefault(0, 0)
    pending, interior, external_pis = [decision['root']], set(), set()
    while pending:
        node = pending.pop()
        if node in values or node in interior or node in external_pis:
            continue
        if node not in aig.fanin_literals:
            if 0 < node <= aig.pi_count:
                external_pis.add(node)
                continue
            raise ValueError('Cut contains a node absent from original AIG.')
        interior.add(node)
        pending.extend(literal >> 1 for literal in aig.fanin_literals[node])
    if external_pis:
        return support_cut_truth(aig, decision, interior, external_pis)
    for node in sorted(interior):
        left, right = aig.fanin_literals[node]
        values[node] = (values[left >> 1] ^ (0xffff if left & 1 else 0)) & (values[right >> 1] ^ (0xffff if right & 1 else 0))
    if values[decision['root']] != (int(decision['truth']) & 0xffff):
        raise ContextProofRequired('Independent cut truth differs; original cut-leaf relationships require CEC.')
    return {'cut_semantics': 'STRUCTURAL_BOUNDARY', 'boolean_validator': 'EXHAUSTIVE_16_BIT_TRUTH'}
