#!/usr/bin/env python3
"""Compare hard port composition with installed AIG structural timing, no ABC.

All values are AND/inversion structural units, never mapped picoseconds.
Uses blend=0, hard max, zero PI arrival to isolate structural composition from
frozen LiteCut scaling, output anchoring, smoothing and mapped timing.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from global_timing.structure import parse_aig
from global_timing.boolean_candidates import parse_library
from global_timing.compositional_proxy import candidate_paths,condense_original


def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def arrivals(aig, overrides=None):
    values={0:0.}
    overrides=overrides or {}
    for node,pair in sorted(aig.fanin_literals.items()):
        if node in overrides:
            values[node]=max(values.get(port,0.)+cost for port,cost in overrides[node].items())
        else:
            values[node]=max(values.get(lit>>1,0.)+1.+.2*(lit&1) for lit in pair)
    return [values.get(lit>>1,0.)+.2*(lit&1) for lit in aig.po_literals]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--panel-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not ROOT.is_relative_to('/data/wushangge'):
        p.error('Read installed graphs only under Slurm in /data.')
    library=parse_library(ROOT/'projects/culs/src/algorithms/rewrite_library.inc')
    rows=[];cases=[]
    for folder in sorted(args.panel_root.glob('case_*')):
        f=read(folder/'frozen_predictions.json');source=Path(f['source'])/'input.aig'
        assert sha(source)==f['input_sha256']
        aig=parse_aig(source);ds=read(folder/'catalogue.json')['decisions']
        options=[candidate_paths(d,d['candidate_subgraph_indices'][0],library) for d in ds]
        retained={lit>>1 for lit in aig.po_literals}
        for d in ds:retained.add(d['root']);retained.update(d['cut_leaves'])
        original=condense_original(aig,retained)
        base=arrivals(aig)
        actual={r['mask']:r for r in read(folder/'measurements.json')['results']}
        case_rows=[]
        for mask in range(16):
            overrides={d['root']:paths for i,(d,paths) in enumerate(zip(ds,options)) if mask>>i&1}
            explicit=arrivals(aig,overrides)
            values={0:0.}
            for node,paths in sorted(original.items()):
                values[node]=max(values.get(port,0.)+cost for port,cost in overrides.get(node,paths).items())
            composed=[values.get(lit>>1,0.)+.2*(lit&1) for lit in aig.po_literals]
            compression_error=max(abs(x-y) for x,y in zip(explicit,composed))
            assert compression_error < 1e-9, (f['circuit'],mask,compression_error)
            record=actual[mask]
            replay=source if mask==0 else Path(record['execution']['aig']) if record.get('execution',{}).get('status')=='PASS' else None
            real=arrivals(parse_aig(replay)) if replay and replay.is_file() else None
            assert real is None or len(real)==len(composed)
            row=dict(circuit_id=f['circuit'],mask=mask,compression_max_po_error_units=compression_error,
                original_structural_critical_units=max(base),composed_structural_critical_units=max(composed),
                installed_structural_critical_units=max(real) if real else None,
                composed_vs_installed_max_po_error_units=max(abs(x-y) for x,y in zip(composed,real)) if real else None,
                composed_vs_installed_critical_error_units=max(composed)-max(real) if real else None,
                installed_aig=str(replay) if replay else None,installed_sha256=sha(replay) if real else None)
            rows.append(row);case_rows.append(row)
        valid=[r for r in case_rows if r['mask'] and r['installed_sha256']]
        cases.append(dict(circuit_id=f['circuit'],nonempty_checked=len(valid),
            mismatched_po_vectors=sum(r['composed_vs_installed_max_po_error_units']>1e-8 for r in valid),
            median_max_po_error_units=statistics.median(r['composed_vs_installed_max_po_error_units'] for r in valid),
            maximum_max_po_error_units=max(r['composed_vs_installed_max_po_error_units'] for r in valid)))
    result=dict(schema='hard_structural_composition_audit_v1',cases=cases,new_mapping_calls=0,
        compression_max_po_error_units=max(r['compression_max_po_error_units'] for r in rows),
        nonempty_checked=sum(c['nonempty_checked'] for c in cases),
        mismatched_po_vectors=sum(c['mismatched_po_vectors'] for c in cases),
        contract='Hard max, AND=1, inversion=0.2, zero PI arrival, LiteCut blend=0, no PO anchor. This checks structure, not ABC timing accuracy.',
        interpretation='Compression equality checks propagation against explicit unsimplified root replacement. A difference versus installed AIG may arise from structural simplification/hash reuse or replay identity changes; this test alone does not assign a specific cause.')
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    with (args.output/'structure.csv').open('x',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
