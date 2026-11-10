#!/usr/bin/env python3
"""One fixed R9 case: validate traces and CEC-prove function correspondence.

Run from a staged /data workspace under Slurm. No rewrite or mapper decisions
are modified. Comparison uses function proofs, never cross-network gate IDs.
"""
import csv
import hashlib
import json
from pathlib import Path
import random
import re
import struct
import subprocess
import sys
from gate3b_analyze import parse_binary_aig

ROOT = Path(sys.argv[1])
PRIOR = Path('/data/wushangge/runs/CULS-mapper-internal-20261003-34U6OZ')
ABC = PRIOR / 'bin/abc_pair_timing'

def records(text, prefix):
    return [dict(re.findall(r'([a-z_]+)=([^\s]+)', line))
            for line in text.splitlines() if line.startswith(prefix+' ')]

def f32(x):
    return struct.unpack('f', struct.pack('f', x))[0]

def probe(g, literal, path):
    data = bytearray(f'aig {g.maximum_variable} {g.pi_count} 0 1 {g.and_count}\n{literal}\n'.encode())
    def varint(v):
        while v >= 128:
            data.append((v & 127) | 128); v >>= 7
        data.append(v)
    for n, ins in sorted(g.fanin_literals.items()):
        lo, hi = sorted(ins); varint(2*n-hi); varint(hi-lo)
    path.write_bytes(data)

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def aig_lit(raw, correspondence):
    # vOrigNodeIds packs ABC pre-map object ID and mapper fPhase, NOT
    # ordinary AIG complement polarity: phase1 implements positive function.
    raw = int(raw)
    if raw < 0: return None
    obj = raw >> 1
    if obj not in correspondence: return None
    var, pol = correspondence[obj][0]
    return 2*var + (pol ^ (1-(raw & 1)))

def csv_write(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

inputs = [PRIOR/'input/c1355_orig.aig', PRIOR/'input/S002.aig']
assert [sha(p) for p in inputs] == [
    '2602b10f131a6ba5996521036ba130aab128a4aa15e3ab49ce44f2f02047f0b0',
    '59dbe54c03d28c336775785e42d3e4781713e66e96769ac688fd38977672b5e1']
accepted = [json.loads(s) for s in (PRIOR/'input/S002.jsonl').read_text().splitlines()
            if json.loads(s)['final_status']=='ACCEPTED']
assert len(accepted)==1 and accepted[0]['root']==197 and accepted[0]['selected_subgraph']==2
assert accepted[0]['cut_leaves']==[17,33,151,188]
assert 'Networks are equivalent' in (ROOT/'cec.log').read_text()
graphs = [parse_binary_aig(p) for p in inputs]
traces, arcs, provenance, validations, correspondences = [], [], [], [], []
for variant, expected in [('baseline',194.169998), ('candidate',193.519989)]:
    log = (ROOT/f'{variant}_instrumented.log').read_text()
    corr = {}
    for r in records(log, 'AIG_CASE'):
        corr.setdefault(int(r['abc_obj_id']), []).append((int(r['aiger_var']),int(r['polarity'])))
    correspondences.append(corr)
    parts = log.split('PAIR_RESULT ')[1:]
    assert len(parts)==2 and 'PAIR_TRACE_ERROR' not in log
    result = records('PAIR_RESULT '+parts[0], 'PAIR_RESULT')[0]
    assert abs(float(result['pair_delay'])-expected) <= 0.00002
    ordinary = records((ROOT/f'{variant}.log').read_text(), 'PAIR_RESULT')
    assert all(r['pair_delay']==result['pair_delay'] for r in ordinary)
    def body(p): return '\n'.join(p.read_text().splitlines()[1:])
    assert body(ROOT/f'{variant}.blif') == body(ROOT/f'{variant}_instrumented.blif')
    assert body(ROOT/f'{variant}.blif') == body(PRIOR/f'output/{variant}_original.blif')
    pre = records(parts[0], 'PAIR_TRACE'); post = records(parts[1], 'PAIR_TRACE')
    assert pre[0]['name']=='pi03' and pre[-1]['name']=='po11'
    assert len(pre)==len(post)==13
    prov = {r['node_id']:r for r in records(parts[0], 'PAIR_CASE')}
    aa = records(parts[0], 'PAIR_ARC')
    accum = 0.0; previous = pre[0]
    for i, (p,q) in enumerate(zip(pre,post)):
        p['trace_position']=i; p['source_pi']='pi03'; p['target_po']='po11'
        if p['kind']=='gate':
            for key in ('cell','pin','arc_delay','arrival','input_transition','output_transition'):
                assert p[key]==q[key]
            assert p['input_id']==previous['node_id']
            assert abs(float(p['input_arrival'])-float(previous['arrival'])) < 0.00002
            # %.9g round-trips float32, but first parse back to float32;
            # adding the decimal double directly can change a half-ULP tie.
            accum = f32(accum+f32(float(p['arc_delay'])))
            assert abs(accum-float(p['arrival'])) < 0.00002
            p['topo_node_id']=q['node_id']
            p['mapper_orig_lit']=prov[p['node_id']]['orig_lit']
            p['serialized_aig_literal']=aig_lit(p['mapper_orig_lit'],corr)
            selected = next(a for a in aa if a['node_id']==p['node_id'] and a['pin']==p['pin'])
            p['pin_name']=selected['pin_name']; p['mio_phase']=selected['phase']
        previous = p
    assert abs(accum-float(result['pair_delay'])) < 0.00002
    csv_write(ROOT/f'R9_{variant.upper()}_PAIR_TRACE.csv', pre)
    traces.append(pre); arcs.append(aa); provenance.append(prov)
    validations.append(dict(variant=variant,pair_delay=float(result['pair_delay']),
        reproduction_error= float(result['pair_delay'])-expected,epsilon_ps=0.00002,
        reconstructed_float32=accum, mapped_blif_identical=True, stime_changes_pair=False))

rng=random.Random(20261012); bits=4096; mask=(1<<bits)-1
pis=[0]+[rng.getrandbits(bits) for _ in range(41)]
def simulate(g):
    values=pis.copy()
    for n,(u,v) in sorted(g.fanin_literals.items()):
        assert n==len(values)
        values.append((values[u>>1] ^ (mask if u&1 else 0)) &
                      (values[v>>1] ^ (mask if v&1 else 0)))
    return values
values = [simulate(g) for g in graphs]
def val(side, lit): return values[side][lit>>1] ^ (mask if lit&1 else 0)
proofs = {}; count=0
def prove(a,b):
    global count
    if a is None or b is None: return 'UNMATCHED'
    if val(0,a)!=val(1,b): return 'UNMATCHED'
    key=(a,b)
    if key not in proofs:
        count+=1; pa=ROOT/f'proof_b_{count}.aig'; pb=ROOT/f'proof_c_{count}.aig'
        probe(graphs[0],a,pa); probe(graphs[1],b,pb)
        p=subprocess.run([str(ABC),'-c',f'cec {pa} {pb}'], text=True, stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        (ROOT/f'proof_{count}.log').write_text(p.stdout)
        assert p.returncode==0 and 'Networks are equivalent' in p.stdout, (a,b,p.stdout)
        proofs[key]=dict(baseline_literal=a,candidate_literal=b,cec='PASS',log=f'proof_{count}.log')
    return 'CEC_PASS'

comparisons=[]; fanin_comparisons=[]
for i,(a,b) in enumerate(zip(traces[0][1:-1],traces[1][1:-1]),1):
    match=prove(a['serialized_aig_literal'],b['serialized_aig_literal'])
    ai=[r for r in arcs[0] if r['node_id']==a['node_id']]
    bi=[r for r in arcs[1] if r['node_id']==b['node_id']]
    for x in ai:
        al = 2*int(x['input_id']) if int(x['input_id'])<=41 else aig_lit(x['input_orig_lit'],correspondences[0])
        choices=[]
        for y in bi:
            bl=2*int(y['input_id']) if int(y['input_id'])<=41 else aig_lit(y['input_orig_lit'],correspondences[1])
            if prove(al,bl)=='CEC_PASS': choices.append(y)
        fanin_comparisons.append(dict(trace_position=i,baseline_pin=x['pin_name'],baseline_input_literal=al,
            candidate_matching_pins=[y['pin_name'] for y in choices],
            candidate_matching_literals=[aig_lit(y['input_orig_lit'],correspondences[1]) if int(y['input_id'])>41 else 2*int(y['input_id']) for y in choices],
            status='CEC_PASS' if choices else 'UNMATCHED'))
    comparisons.append(dict(position=i,cell=a['cell'],baseline_node=a['node_id'],candidate_node=b['node_id'],
        baseline_literal=a['serialized_aig_literal'],candidate_literal=b['serialized_aig_literal'],function_match=match,
        baseline_pin=a['pin_name'],candidate_pin=b['pin_name'],baseline_arc=float(a['arc_delay']),candidate_arc=float(b['arc_delay']),
        baseline_arrival=float(a['arrival']),candidate_arrival=float(b['arrival'])))

cover_context=[]
for side,variant in enumerate(('baseline','candidate')):
    net=records((ROOT/f'{variant}_instrumented.log').read_text(),'CASE_NET')
    indexed={r['node_id']:r for r in net}
    cone=set()
    def visit(n):
        if n in cone: return
        cone.add(n)
        if n in indexed:
            for f in indexed[n]['fanins'].split(','): visit(f)
    visit(traces[side][-1]['driver_id'])
    for r in net:
        lit=aig_lit(r['orig_lit'],correspondences[side])
        points=(197,211,212,213) if side==0 else (287,313,314,322)
        if lit is not None and lit>>1 in points:
            cover_context.append(dict(variant=variant,serialized_aig_literal=lit,
                in_po11_cone=r['node_id'] in cone,on_critical_trace=any(t['node_id']==r['node_id'] for t in traces[side]),**r))
    csv_write(ROOT/f'R9_{variant.upper()}_MAPPED_STATE.csv',net)
(ROOT/'mechanism_evidence.json').write_text(json.dumps(dict(validations=validations,comparisons=comparisons,
    fanin_comparisons=fanin_comparisons,proofs=list(proofs.values()),candidate=accepted[0],
    cover_context=cover_context,binary_sha=sha(ROOT/'abc/abc'),library_sha=sha(PRIOR/'input/asap7.lib')),indent=2)+'\n')
csv_write(ROOT/'R9_TRACE_FANIN_ALTERNATIVES.csv', [dict(variant=v,**r) for v,aa in zip(('baseline','candidate'),arcs) for r in aa])
print(json.dumps(dict(validations=validations,comparisons=comparisons),indent=2))
