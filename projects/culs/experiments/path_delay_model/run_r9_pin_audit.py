#!/usr/bin/env python3
"""Audit only the CEC-proved R9 downstream NAND4's existing mapper records.

Execute under Slurm in a /data run directory. No mapper mutation, new
candidate, forced permutation, or new timing/prediction algorithm.
"""
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sys

root=Path(sys.argv[1])
prior=Path('/data/wushangge/runs/CULS-mapper-internal-20261003-34U6OZ')
case=Path('/data/wushangge/runs/CULS-R9-mechanism-20261012-9CY36o')
original=Path('/data/wushangge/runs/CULS-source-pair-20260930-qU1wxi/bin/abc_pair_timing')
def f32(x): return struct.unpack('f',struct.pack('f',x))[0]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def execute(script,log):
    env={k:v for k,v in os.environ.items() if not k.startswith('AIGREWRITE_MAP_DEBUG_')}
    p=subprocess.run([str(original),'-c',script],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env=env)
    log.write_text(p.stdout); assert p.returncode==0
    return p.stdout
def body(p): return '\n'.join(p.read_text().splitlines()[1:])
def pair(text):
    row=next(s for s in text.splitlines() if s.startswith('PAIR_RESULT '))
    return float(re.search(r'pair_delay=([^ ]+)',row)[1])
def metrics(text):
    # Liberty stime gate count/area are used only for noninterference checks.
    return re.findall(r'Gates\s*=\s*([\d.]+).*?Area\s*=\s*([\d.]+)',text)
assert sha(prior/'input/c1355_orig.aig')=='2602b10f131a6ba5996521036ba130aab128a4aa15e3ab49ce44f2f02047f0b0'
assert sha(prior/'input/S002.aig')=='59dbe54c03d28c336775785e42d3e4781713e66e96769ac688fd38977672b5e1'
manifest=[json.loads(s) for s in (prior/'input/S002.jsonl').read_text().splitlines()]
accepted=[x for x in manifest if x['final_status']=='ACCEPTED']
assert len(accepted)==1 and accepted[0]['root']==197 and accepted[0]['selected_subgraph']==2
assert accepted[0]['cut_slot']==2 and accepted[0]['cut_leaves']==[17,33,151,188]
assert 'Networks are equivalent' in execute(f'cec {prior}/input/c1355_orig.aig {prior}/input/S002.aig',root/'cec.log')
evidence=json.loads((case/'mechanism_evidence.json').read_text())
gate=evidence['comparisons'][6]
assert gate['function_match']=='CEC_PASS' and gate['baseline_literal']==771 and gate['candidate_literal']==731
rows=[]; summary=[]
for variant,expected,target,aig,critical in [('baseline',194.169998,385,'c1355_orig.aig',176),('candidate',193.519989,365,'S002.aig',339)]:
    text=(root/f'{variant}.log').read_text()
    normal=execute(f'read {prior}/input/asap7.lib; read {prior}/input/{aig}; map; write_blif {root}/{variant}_original.blif; topo; stime; print_stats; print_pair_delay -tv po11 pi03',root/f'{variant}_original.log')
    assert abs(pair(text)-expected)<=0.00002 and pair(text)==pair(normal)
    assert body(root/f'{variant}.blif')==body(root/f'{variant}_original.blif')==body(prior/f'output/{variant}_original.blif')
    assert metrics(text)==metrics(normal) and metrics(text)
    rec=[dict(json.loads(s[7:]),event_index=i) for i,s in enumerate(text.splitlines()) if s.startswith('MAPDBG ')]
    best=next(x for x in rec if x['type']=='best' and x['stage']=='final' and x['phase']==0)
    ci=best['cut_index']; leaves=best['leaves']; assert leaves[1]==critical
    truth=next(x for x in rec if x['type']=='truth' and x['cut_index']==ci)
    inv=[x for x in rec if x['type']=='match_inventory' and x['cut_index']==ci and x['phase']==0 and x['super']['root_cell']=='NAND4xp25_ASAP7_75t_L']
    assert len(inv)==24 and all(x['within_match_limit'] for x in inv)
    attempts=[x for x in rec if x['type']=='match_attempt' and x['cut_index']==ci and x['phase']==0 and x['super']['root_cell']=='NAND4xp25_ASAP7_75t_L']
    # The existing debug records contain bounded timing attempts. A rejected
    # attempt's rise/fall may be partial and Worst is the large sentinel.
    # Audit-only unbounded recomputation below uses that attempt's recorded
    # leaf arrivals and library arcs, NOT a forced mapping experiment.
    for mode in sorted({x['mode'] for x in attempts}):
        for sid in ((2043,2044,2046,2047) if mode==0 else (2046,2047)):
            first=next(x for x in attempts if x['mode']==mode and x['super']['id']==sid)
            sup=first['super']; perm=[x['leaf_pin'] for x in sup['implementation']['fanins']]
            critical_pin='ABCD'[perm.index(1)]
            contribution=[]
            for i,(arrival,arc) in enumerate(zip(first['leaf_arrivals'],sup['arcs'])):
                contribution.append(max(f32(f32(arrival[t])+f32(arc[a])) for t,a in (('rise','rr'),('fall','fr'),('rise','rf'),('fall','ff'))))
            statuses=[x['status'] for x in attempts if x['mode']==mode and x['super']['id']==sid]
            row=dict(variant=variant,function_point=target,mapper_node_id=best['mapper_id'],mapping_mode=mode,
                cut_index=ci,leaves=json.dumps(leaves),leaf_mapper_ids=json.dumps(best['leaf_mapper_ids']),
                output_phase=0,input_phase_mask=first['input_phase_mask'],truth_lo=truth['truth_lo'],truth_hi=truth['truth_hi'],
                supergate_id=sid,cell=sup['root_cell'],physical_pin_to_leaf=json.dumps(perm),critical_leaf_index=1,critical_pin=critical_pin,
                leaf_rise_arrivals=json.dumps([x['rise'] for x in first['leaf_arrivals']]),leaf_fall_arrivals=json.dumps([x['fall'] for x in first['leaf_arrivals']]),
                leaf_arc_rr=json.dumps([x['rr'] for x in sup['arcs']]),leaf_arc_fr=json.dumps([x['fr'] for x in sup['arcs']]),
                leaf_arc_rf=json.dumps([x['rf'] for x in sup['arcs']]),leaf_arc_ff=json.dumps([x['ff'] for x in sup['arcs']]),
                recorded_output_rise=first['arrival']['rise'],recorded_output_fall=first['arrival']['fall'],recorded_output_worst=first['arrival']['worst'],
                complete_timing_record=first['status']!='delay_limit',audit_unbounded_output=max(contribution),
                audit_leaf_contributions=json.dumps(contribution),matching_cost=first['area_cost'],library_cell_area=sup['area'],
                limit=first['limit'],statuses=json.dumps(statuses),selected_in_cut='selected_in_cut' in statuses,
                final_best_supergate=best['super']['id'],final_selected=(best['super']['id']==sid),event_index=first['event_index'],
                estimated_load='UNAVAILABLE',pair_delay=pair(text))
            if first['status']!='delay_limit': assert abs(max(contribution)-first['arrival']['worst'])<0.00002
            rows.append(row)
    selected0=[x for x in attempts if x['mode']==0 and x['status']=='selected_in_cut']
    summary.append(dict(variant=variant,truth=truth,inventory_count=len(inv),final=best,
        delay_dp=next(x for x in rec if x['type']=='best' and x['stage']=='delay_dp' and x['phase']==0),
        first_selections=selected0,metrics=metrics(text),pair_delay=pair(text),noninterference=True,
        node=next(x for x in rec if x['type']=='node' and x['aiger_var']==target)))
assert summary[0]['truth']['truth_lo']==summary[1]['truth']['truth_lo']
assert summary[0]['truth']['truth_hi']==summary[1]['truth']['truth_hi']
with (root/'R9_NAND4_PIN_DECISION_COMPARISON.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(root/'pin_audit_evidence.json').write_text(json.dumps(dict(summary=summary,rows=rows,
    candidate_manifest=accepted[0],observer_binary_sha=sha(prior/'bin/abc_pair_timing'),
    original_binary_sha=sha(original),library_sha=sha(prior/'input/asap7.lib')),indent=2)+'\n')
for r in rows:
    if r['mapping_mode']==0: print(r['variant'],r['supergate_id'],r['critical_pin'],r['leaf_rise_arrivals'],r['audit_unbounded_output'],r['matching_cost'],r['statuses'])
print('PASS: hashes, CEC, canonical truth,24 permutations, BLIF identity, gates/area, pair timing, arithmetic.')
