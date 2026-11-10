#!/usr/bin/env python3
"""Structural frontier audit only; never reads candidate full-map timing."""
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
from gate3b_analyze import parse_binary_aig

out=Path(sys.argv[1])
prior=Path('/data/wushangge/runs/CULS-mapper-internal-20261003-34U6OZ')
baseline=prior/'input/c1355_orig.aig'; candidate=prior/'input/S002.aig'
assert hashlib.sha256(baseline.read_bytes()).hexdigest()=='2602b10f131a6ba5996521036ba130aab128a4aa15e3ab49ce44f2f02047f0b0'
assert hashlib.sha256(candidate.read_bytes()).hexdigest()=='59dbe54c03d28c336775785e42d3e4781713e66e96769ac688fd38977672b5e1'
manifest=[json.loads(s) for s in (prior/'input/S002.jsonl').read_text().splitlines()]
accepted=[r for r in manifest if r['final_status']=='ACCEPTED']
assert len(accepted)==1 and accepted[0]['root']==197 and accepted[0]['cut_slot']==2 and accepted[0]['selected_subgraph']==2
a,b=parse_binary_aig(baseline),parse_binary_aig(candidate)
rng=random.Random(20261014); mask=(1<<4096)-1
pis=[0]+[rng.getrandbits(4096) for _ in range(41)]
def simulate(g):
    values=pis.copy()
    for n,(u,v) in sorted(g.fanin_literals.items()):
        assert n==len(values)
        values.append((values[u>>1]^(mask if u&1 else 0))&(values[v>>1]^(mask if v&1 else 0)))
    return values
va,vb=simulate(a),simulate(b)
reachable=set()
def visit(n):
    if n in reachable:return
    reachable.add(n)
    for lit in b.fanin_literals.get(n,[]):visit(lit>>1)
for lit in b.po_literals:visit(lit>>1)
def probe(g,lit,p):
    data=bytearray(f'aig {g.maximum_variable} {g.pi_count} 0 1 {g.and_count}\n{lit}\n'.encode())
    def vint(x):
        while x>=128:data.append((x&127)|128);x>>=7
        data.append(x)
    for n,ins in sorted(g.fanin_literals.items()):
        lo,hi=sorted(ins);vint(2*n-hi);vint(hi-lo)
    p.write_bytes(data)
mapping={}
for node in [17,33,151,188,197,209,213]:
    matches=[(n,int(vb[n]!=va[node])) for n in sorted(reachable) if vb[n]==va[node] or (vb[n]^mask)==va[node]]
    assert len(matches)==1,(node,matches)
    n,pol=matches[0];mapping[node]=dict(candidate_node=n,polarity=pol)
    ap=out/f'frontier_baseline_{node}.aig';bp=out/f'frontier_candidate_{node}.aig'
    probe(a,2*node,ap);probe(b,2*n+pol,bp)
    p=subprocess.run([str(prior/'bin/abc_pair_timing'),'-c',f'cec {ap} {bp}'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    (out/f'frontier_cec_{node}.log').write_text(p.stdout)
    assert 'Networks are equivalent' in p.stdout
    mapping[node]['cec']='PASS'
def cone(g,root,frontier):
    nodes=set();crossed=set()
    def dfs(n):
        if n in frontier:crossed.add(n);return
        if n in nodes:return
        assert n in g.fanin_literals,('unexpected source',n)
        nodes.add(n)
        for lit in g.fanin_literals[n]:dfs(lit>>1)
    dfs(root)
    return dict(nodes=sorted(nodes),frontier=sorted(crossed),fanins={str(n):g.fanin_literals[n] for n in sorted(nodes)})
frontier=[17,33,151,188,209]
result=dict(mapping=mapping,manifest=accepted[0],baseline=cone(a,213,set(frontier)),
    candidate=cone(b,322,{mapping[n]['candidate_node'] for n in frontier}),
    prediction_inputs=['baseline AIG','accepted candidate AIG','replay manifest','function-only CEC'],
    forbidden_inputs_read=False)
(out/'frontier_structure.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
