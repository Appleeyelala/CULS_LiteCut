#!/usr/bin/env python3
"""Read-only structural/ordering audit of paid original versus replay no-op."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from joint_gradient_panel import read,save,sha
from global_timing.structure import parse_aig


def fingerprints(aig):
    # PI identity/PO order are fixed interface positions. AND inputs commute;
    # inversion remains a literal attribute. No Boolean simplification occurs.
    hashes={i:hashlib.sha256(f'PI:{i}'.encode()).digest() for i in range(aig.pi_count+1)}
    hashes[0]=hashlib.sha256(b'CONST0').digest()
    counts=Counter()
    labeled=hashlib.sha256(f'{aig.pi_count},{aig.po_count},{aig.and_count};'.encode())
    for node,pair in sorted(aig.fanin_literals.items()):
        labeled.update(f'{node},{pair[0]},{pair[1]};'.encode())
        parts=sorted(hashes[l>>1]+bytes([l&1]) for l in pair)
        value=hashlib.sha256(b'AND'+b''.join(parts)).digest();hashes[node]=value;counts[value]+=1
    outputs=[hashlib.sha256(hashes[l>>1]+bytes([l&1])).hexdigest() for l in aig.po_literals]
    labeled.update(','.join(map(str,aig.po_literals)).encode())
    forest=hashlib.sha256(b''.join(h+k.to_bytes(8,'little') for h,k in sorted(counts.items()))).hexdigest()
    return {'ands':aig.and_count,'pis':aig.pi_count,'pos':aig.po_count,'output_hashes':outputs,
            'internal_hash_multiset':forest,'duplicate_structural_nodes':sum(n-1 for n in counts.values()),
            'labeled_graph_digest':labeled.hexdigest()}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not ROOT.is_relative_to('/data/wushangge'):p.error('Committed /data source under Slurm required')
    rows=[]
    for folder in sorted(a.data.glob('case_*')):
        f=read(folder/'frozen.json');baseline=read(folder/'baseline.json');m=read(folder/'measurements.json')
        noop=next(r for r in m['results'] if r['index']==0)
        if noop['status']!='PASS':rows.append({'circuit_id':f['circuit_id'],'status':noop['status']});continue
        original=Path(f['source'])/'input.aig';replayed=Path(noop['execution']['aig'])
        before=fingerprints(parse_aig(original));after=fingerprints(parse_aig(replayed))
        changed=[i for i,(x,y) in enumerate(zip(before['output_hashes'],after['output_hashes'])) if x!=y]
        rows.append({'circuit_id':f['circuit_id'],'role':f['role'],'status':'PASS',
            'original_delay_ps':baseline['delay_ps'],'serialized_noop_delay_ps':noop['metrics']['delay_ps'],
            'delay_ratio':noop['metrics']['delay_ps']/baseline['delay_ps'],
            'original_area':baseline['area'],'serialized_noop_area':noop['metrics']['area'],
            'cec':noop['metrics']['cec'],'structural_timing_error':noop['structural_max_error'],
            'original_aig_sha256':sha(original),'replayed_aig_sha256':sha(replayed),'byte_identical':sha(original)==sha(replayed),
            'original_and_count':before['ands'],'replayed_and_count':after['ands'],
            'output_count_equal':before['pos']==after['pos'],'changed_po_structural_hash_indices':changed,
            'internal_hash_multiset_equal':before['internal_hash_multiset']==after['internal_hash_multiset'],
            'exact_numbered_graph_equal':before['labeled_graph_digest']==after['labeled_graph_digest'],
            'original_duplicate_structural_nodes':before['duplicate_structural_nodes'],
            'replayed_duplicate_structural_nodes':after['duplicate_structural_nodes'],
            'original_path':str(original),'replayed_path':str(replayed)})
    save(a.output,{'schema':'replay_noop_structural_fingerprint_v3','rows':rows,'new_mapping_calls':0,
        'semantics':'Commutative structural cone hashes with PI identity and PO order, not Boolean functional hashes.',
        'caution':'Matching cone hashes alone do not prove identical sharing when duplicate structural nodes exist; no mapper causality claim.'})


if __name__=='__main__':main()
