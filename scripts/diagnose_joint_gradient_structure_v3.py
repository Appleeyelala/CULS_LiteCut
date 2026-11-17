#!/usr/bin/env python3
"""Isolate phase composition from folding/hashing, using existing replay only."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from global_timing.boolean_candidates import parse_library, verify_library_truth
from global_timing.structure import parse_aig


def structural_arrivals(aig):
    values = {i: 0. for i in range(aig.pi_count + 1)}
    for node, pair in sorted(aig.fanin_literals.items()):
        values[node] = max(values[lit >> 1] + 1. + .2 * (lit & 1) for lit in pair)
    return [values[lit >> 1] + .2 * (lit & 1) for lit in aig.po_literals]


def literal_composition(aig, decisions, choices, library, simplify=False):
    """Diagnostic explicit graph. Keep output phase as a literal until consumed.

    Original and candidate edges compose with XOR, exactly as AIG literals do.
    simplify=False changes ONLY phase representation versus scalar-path splicing.
    No claim about native replay order or full equivalence until audited.
    """
    active = {d['root']: (d, c) for d, c in zip(decisions, choices) if c}
    if len(active) != sum(bool(c) for c in choices):
        raise ValueError('Multiple selected decisions at one root')
    translated = {i: i << 1 for i in range(aig.pi_count + 1)}
    fanins, hashed, events = {}, {}, []

    def make_and(left, right):
        if simplify:
            if left == 0 or right == 0 or left == (right ^ 1): return 0
            if left == 1 or left == right: return right
            if right == 1: return left
        pair = tuple(sorted((left, right)))
        if simplify and pair in hashed: return hashed[pair]
        node = aig.pi_count + len(fanins) + 1
        fanins[node] = pair
        hashed[pair] = node << 1
        return node << 1

    for node, pair in sorted(aig.fanin_literals.items()):
        if node not in active:
            translated[node] = make_and(*(translated[lit >> 1] ^ (lit & 1) for lit in pair))
            continue
        d, choice = active[node]
        subgraph = d['candidate_subgraph_indices'][choice - 1]
        proof = verify_library_truth(d, subgraph, library)
        truth = int(d['truth']) & 0xffff
        phase = library['p_phases'][truth]
        perm = library['perms4'][library['p_perms'][truth]]
        leaves = list(d['cut_leaves'])
        if len(leaves) == 3: leaves.append(0)
        values = [translated[leaves[perm[k]]] ^ ((phase >> k) & 1) for k in range(4)]
        data = library['classes'][d['truth_class']]
        root = data['roots'][subgraph]
        used, pending = set(), [root]
        while pending:
            v = pending.pop()
            if v < 4 or v in used: continue
            used.add(v)
            a, b, _, _ = data['fanins'][v - 4]
            pending.extend((a, b))
        values.extend([None] * len(data['fanins']))
        for v in sorted(used):
            a, b, ca, cb = data['fanins'][v - 4]
            values[v] = make_and(values[a] ^ ca, values[b] ^ cb)
        translated[node] = values[root] ^ int(proof['output_phase_complement'])
        events.append({'root': node, 'output_complement': proof['output_phase_complement'],
                       'resolved_literal_complement': translated[node] & 1})
    return SimpleNamespace(pi_count=aig.pi_count, fanin_literals=fanins,
                           po_literals=[translated[l >> 1] ^ (l & 1) for l in aig.po_literals]), events


def read(p): return json.loads(p.read_text())
def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--panel-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not ROOT.is_relative_to('/data/wushangge'):
        p.error('Run from committed /data snapshot under Slurm')
    lib = parse_library(ROOT / 'projects/culs/src/algorithms/rewrite_library.inc')
    rows, evidence = [], []
    for folder in sorted(args.panel_root.glob('case_*')):
        frozen = read(folder / 'frozen_predictions.json')
        source = Path(frozen['source']) / 'input.aig'
        assert sha(source) == frozen['input_sha256']
        aig = parse_aig(source)
        decisions = read(folder / 'catalogue.json')['decisions']
        for record in read(folder / 'measurements.json')['results']:
            mask = record['mask']
            installed = source if not mask else Path(record['execution']['aig'])
            actual = structural_arrivals(parse_aig(installed))
            row = {'circuit_id': frozen['circuit'], 'mask': mask,
                   'installed_aig': str(installed), 'installed_sha256': sha(installed)}
            choices = [(mask >> i) & 1 for i in range(len(decisions))]
            for simplify, name in [(False, 'literal_only'), (True, 'canonical')]:
                graph, events = literal_composition(aig, decisions, choices, lib, simplify)
                pred = structural_arrivals(graph)
                row[name + '_max_po_error_units'] = max(abs(a-b) for a,b in zip(actual,pred))
            rows.append(row)
            if mask:
                evidence.append({'circuit_id': frozen['circuit'], 'mask': mask, 'phase_events': events})
        print(json.dumps({'circuit': frozen['circuit'], 'checked': 16}), flush=True)
    summary = {'schema': 'literal_composition_diagnosis_v3', 'new_mapping_calls': 0,
               'rows': len(rows), 'nonempty': sum(bool(r['mask']) for r in rows),
               'units': 'AND=1, complemented edge=0.2, PI=0; not ps',
               'comparators': {name: {'mismatches': sum(r[name+'_max_po_error_units'] > 1e-8 for r in rows),
                                      'maximum_error': max(r[name+'_max_po_error_units'] for r in rows)}
                              for name in ['literal_only', 'canonical']}}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    (args.output/'phase_events.json').write_text(json.dumps(evidence, indent=2)+'\n')
    with (args.output/'cases.csv').open('x', newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    print(json.dumps(summary,indent=2))


if __name__ == '__main__': main()
