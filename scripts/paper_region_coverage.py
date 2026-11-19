"""Measure editable MFFC coverage in original AIG fanin cones of near-critical mapped outputs."""
import csv
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from global_timing.boolean_candidates import parse_library
from global_timing.candidates import Catalogue
from global_timing.common import digest, save_json
from global_timing.evaluation import parse_timing
from global_timing.feasibility import CandidateFeasibility
from global_timing.structure import parse_aig


def measure(row, library):
    work = Path(row['work'])
    inputs = json.loads((work / 'inputs.json').read_text())
    source = Path(inputs['original'])
    prepared = work / 'prepared'
    manifest = json.loads((prepared / 'prepared.json').read_text())
    assert digest(source) == manifest['original_sha256'] == inputs['original_sha256']
    assert digest(prepared / 'catalogue.json') == manifest['files']['catalogue.json']
    aig = parse_aig(source)
    timing = parse_timing((work / 'original/mapping.log').read_text())
    arrivals = timing['po_arrivals_ps']
    assert len(arrivals) == len(aig.po_literals)
    cat = Catalogue(**json.loads((prepared / 'catalogue.json').read_text()))
    allowed = CandidateFeasibility(aig, cat, library).allowed()
    roots = {d['root']: d['decision_owner_region'] for d, mask in zip(cat.decisions, allowed)
             if any(mask[1:])}
    near = [i for i, value in enumerate(arrivals) if value >= 0.95 * timing['delay_ps']]
    # Information capped at two distinct owner IDs suffices for the per-PO indicator.
    first_two = {}
    for node, parents in sorted(aig.fanin_literals.items()):
        members = {roots[node]} if node in roots else set()
        for literal in parents:
            members.update(first_two.get(literal >> 1, ()))
        if members:
            first_two[node] = tuple(sorted(members)[:2])
    cone = set()
    pending = [aig.po_literals[i] >> 1 for i in near]
    while pending:
        node = pending.pop()
        if node not in cone:
            cone.add(node)
            pending.extend(lit >> 1 for lit in aig.fanin_literals.get(node, ()))
    editable_owners = set(roots.values())
    covered = {owner for node, owner in roots.items() if node in cone}
    multi = sum(len(first_two.get(aig.po_literals[i] >> 1, ())) == 2 for i in near)
    return dict(circuit=row['circuit'], original_sha256=inputs['original_sha256'], ands=aig.and_count,
                po_count=len(arrivals), near_po_count=len(near), near_po_with_two_regions=multi,
                fraction_near_po_with_two=multi/len(near), editable_owners=len(editable_owners),
                near_editable_owners=len(covered), fraction_editable_owners_near=len(covered)/len(editable_owners)
                if editable_owners else 0, original_ps=timing['delay_ps'])


if __name__ == '__main__':
    if 'SLURM_JOB_ID' not in os.environ:
        raise SystemExit('Slurm required')
    out = Path(sys.argv[1])
    out.mkdir()
    library = parse_library(ROOT / 'projects/culs/src/algorithms/rewrite_library.inc')
    with (ROOT / 'analysis/paper14_fullflow/results.csv').open() as stream:
        for row in csv.DictReader(stream):
            result = measure(row, library)
            save_json(out / (row['circuit'] + '.json'), result)
            print(row['circuit'], result['near_po_with_two_regions'], flush=True)
