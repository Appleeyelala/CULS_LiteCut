#!/usr/bin/env python3
"""One fixed R9 rewrite: prove function identity, observe mapper, check noninterference."""
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).parent / 'tools'))
from gate3b_analyze import parse_binary_aig


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_out(path, rows, fields=None):
    fields = fields or list(rows[0])
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for row in rows:
            w.writerow({k: json.dumps(v, separators=(',', ':')) if isinstance(v, (list,dict)) else v
                        for k,v in row.items()})


def run(binary, script, log, env=None):
    e = os.environ.copy()
    for k in list(e):
        if k.startswith('AIGREWRITE_MAP_DEBUG_'): del e[k]
    if env: e.update(env)
    p = subprocess.run([str(binary), '-c', script], text=True, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, env=e)
    log.write_text(p.stdout)
    if p.returncode or 'Assertion' in p.stdout:
        raise RuntimeError(f'ABC error: {log}')
    return p.stdout


def probe_aig(aig, literal, path):
    # Separate equivalence-only network. Never sent to the mapping audit.
    data = bytearray(f'aig {aig.maximum_variable} {aig.pi_count} 0 1 {aig.and_count}\n{literal}\n'.encode())
    def varint(v):
        while v >= 128:
            data.append((v & 127) | 128); v >>= 7
        data.append(v)
    for n, fanins in sorted(aig.fanin_literals.items()):
        lo,hi = sorted(fanins)
        varint(2*n-hi); varint(hi-lo)
    path.write_bytes(data)


def function_correspondence(root, baseline, candidate, abc, out):
    a, b = parse_binary_aig(baseline), parse_binary_aig(candidate)
    assert a.pi_count == b.pi_count == 41 and a.po_count == b.po_count == 32
    rng=random.Random(20261003); bits=4096; mask=(1<<bits)-1
    pis=[0]+[rng.getrandbits(bits) for _ in range(a.pi_count)]
    def sim(g):
        values=pis.copy()
        for n,(u,v) in sorted(g.fanin_literals.items()):
            assert n==len(values)
            get=lambda lit: values[lit>>1] ^ (mask if lit&1 else 0)
            values.append(get(u)&get(v))
        return values
    va,vb=sim(a),sim(b)
    reachable=set()
    def reach(n):
        if n in reachable:return
        reachable.add(n)
        if n in b.fanin_literals:
            for lit in b.fanin_literals[n]:reach(lit>>1)
    for lit in b.po_literals:reach(lit>>1)
    rows=[]; correspondence={}
    for n in (17,33,151,188,197,213,211,212):
        matches=[(i,int(vb[i]!=va[n])) for i in sorted(reachable)
                 if vb[i]==va[n] or (vb[i]^mask)==va[n]]
        if len(matches)!=1:
            if n in (211,212):
                continue  # optional downstream comparison only
            write_json(out/'function_correspondence_failure.json',dict(original_node=n,matches=matches))
            raise RuntimeError(f'Function point {n}: ambiguous/unavailable reachable simulation candidates {matches}')
        node,pol=matches[0]
        ap=out/f'proof_original_{n}.aig';bp=out/f'proof_candidate_{n}.aig'
        probe_aig(a,2*n,ap);probe_aig(b,2*node+pol,bp)
        log=run(abc,f'read "{ap}"; read "{bp}"; cec -n',out/f'function_cec_{n}.log')
        assert 'Networks are equivalent' in log, (n,log)
        rows.append(dict(original_aiger_var=n,candidate_aiger_var=node,candidate_literal_polarity=pol,
                         simulation_bits=bits,cec='PASS'))
        correspondence[n]=(node,pol)
    write_json(out/'function_point_proofs.json',rows)
    return correspondence


def parse_records(output):
    return [json.loads(line[7:]) for line in output.splitlines() if line.startswith('MAPDBG ')]


def mapping_metrics(text):
    text=re.sub(r'\x1b\[[0-9;]*m','',text)
    q=re.search(r'Gates\s*=\s*(\d+).*?Area\s*=\s*([\d.]+).*?Delay\s*=\s*([\d.]+)',text)
    pair=re.findall(r'PAIR_RESULT .*?pair_delay=([\d.eE+-]+)',text)
    assert q and pair, text[-1000:]
    return dict(gates=int(q[1]),area=float(q[2]),liberty_stime_delay=float(q[3]),pair_mio_delay=float(pair[-1]))


def canonical_blif(path):
    return '\n'.join(s for s in path.read_text().splitlines() if s.strip() and not s.startswith('#'))


def observe(args, variant, aig, corr):
    root=args; out=root/'output'; lib=root/'input/asap7.lib'
    original=Path('/data/wushangge/runs/CULS-source-pair-20260930-qU1wxi/bin/abc_pair_timing')
    debug=root/'bin/abc_pair_timing'
    target=197 if variant=='baseline' else corr[197][0]
    ids=[17,33,151,188,197,213] if variant=='baseline' else [corr[n][0] for n in (17,33,151,188,197,213)]
    def script(path):
        return f'read "{lib}"; read "{aig}"; map; write_blif "{path}"; topo; stime; print_stats; print_pair_delay po11 pi03'
    original_path=out/f'{variant}_original.blif'
    off_path=out/f'{variant}_debug_off.blif'
    debug_path=out/f'{variant}_debug_on.blif'
    orig_text=run(original,script(original_path),out/f'{variant}_original.log')
    off_text=run(debug,script(off_path),out/f'{variant}_debug_off.log')
    env=dict(AIGREWRITE_MAP_DEBUG_NODE=str(target),AIGREWRITE_MAP_DEBUG_IDS=','.join(map(str,ids)))
    on_text=run(debug,script(debug_path),out/f'{variant}_debug_on.log',env)
    assert 'MAPDBG ' not in off_text
    assert canonical_blif(original_path)==canonical_blif(off_path)==canonical_blif(debug_path), 'Instrumentation changed mapped network'
    assert mapping_metrics(orig_text)==mapping_metrics(off_text)==mapping_metrics(on_text)
    for label,path in [('original',original_path),('debug_on',debug_path)]:
        text=run(original,f'read "{lib}"; read "{aig}"; read "{path}"; cec -n',out/f'{variant}_mapped_cec_{label}.log')
        assert 'Networks are equivalent' in text
    records=parse_records(on_text)
    (out/f'{variant}_mapper_records.jsonl').write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in records))
    write_json(out/f'{variant}_instrumentation_invariance.json',dict(
        mapped_network_identical=True,cec='PASS',metrics=mapping_metrics(on_text),
        original_binary_sha=sha(original),instrumented_binary_sha=sha(debug),debug_env=env))
    return records


def export(variant,records,out,corr):
    cuts=[r for r in records if r['type']=='cut']
    truths={tuple(r['leaves']):r for r in records if r['type']=='truth'}
    target=[17,33,151,188] if variant=='baseline' else [corr[n][0] for n in (17,33,151,188)]
    reverse={v[0]:k for k,v in corr.items()}
    for r in cuts:
        r['original_verified_leaves']=[x if variant=='baseline' else reverse.get(x,f'unlabelled_serial_{x}') for x in r['leaves']]
        r['target_cut']=set(r['leaves'])==set(target)
        tr=truths.get(tuple(r['leaves']),{})
        r.update(truth_lo=tr.get('truth_lo'),truth_hi=tr.get('truth_hi'),retained=True)
    suffix='baseline' if variant=='baseline' else 'candidate'
    stem='node197' if variant=='baseline' else 'node197_or_equiv'
    csv_out(out/f'{stem}_cuts_{suffix}.csv',cuts,list(cuts[0]))
    stages=[r for r in records if r['type']=='cut_stage']
    csv_out(out/f'{stem}_cut_stages_{suffix}.csv',stages,sorted(set().union(*(r.keys() for r in stages))))
    matches=[]
    for r in records:
        if r['type'] not in ('match_inventory','match_attempt'):continue
        s=r['super']; arr=r.get('arrival')
        complete=r.get('status') in ('eligible','selected_in_cut')
        matches.append(dict(record_type=r['type'],cut_index=r['cut_index'],phase=r['phase'],mode=r.get('mode'),
                            library_rank=r.get('library_rank'),input_phase_mask=r['input_phase_mask'],
                            supergate_id=s['id'],root_cell=s['root_cell'],library_implementation=s['implementation'],
                            rise_arrival=arr['rise'] if arr and complete else None,fall_arrival=arr['fall'] if arr and complete else None,
                            worst_arrival=arr['worst'] if arr and complete else None,arrival_complete=complete,
                            partial_or_saved_arrival=arr if not complete else None,
                            supergate_area=s['area'],area_cost=r.get('area_cost') if complete or r.get('mode',0)>0 else None,arc_delays=s['arcs'],
                            leaf_arrivals=r.get('leaf_arrivals'),status=r.get('status','library_list_only'),
                            within_match_limit=r.get('within_match_limit'),estimated_load='NOT_AVAILABLE'))
    csv_out(out/f'{stem}_matches_{suffix}.csv',matches,list(matches[0]))
    # Explicit empty table lookups are observations, not invented timed matches.
    for cut in cuts:
        if cut['n_leaves']==1:continue
        for phase in (0,1):
            inventory=[r for r in records if r['type']=='match_inventory'
                       and r['cut_index']==cut['cut_index'] and r['phase']==phase]
            if not inventory:
                matches.append(dict(record_type='empty_library_lookup',cut_index=cut['cut_index'],phase=phase,
                                    status='NO_DIRECT_LIBRARY_MATCH',arrival_complete=False))
    csv_out(out/f'{stem}_matches_{suffix}.csv',matches,list(matches[0]))
    best=[r for r in records if r['type']=='best']
    write_json(out/f'{stem}_best_match_{suffix}.json',dict(phases=best,
        timing_semantics='mapper all-PI arrival; phase0 negative / phase1 positive; not source-restricted pair delay'))
    cover=[r for r in records if r['type']=='cover']
    final=[r for r in best if r['stage']=='final']
    b=next(r for r in final if r['phase']==1)
    aig=parse_binary_aig(out.parent/'input'/('c1355_orig.aig' if variant=='baseline' else 'S002.aig'))
    serial_target=197 if variant=='baseline' else corr[197][0]
    table='| Output serialized AIG var | Verified original function | Phase | Supergate | Cell | Cut leaves | Arrival | Internal nodes | Witness to target |\n|---|---|---|---|---|---|---:|---|---|\n'
    for entry in cover:
        leafset=set(entry['leaves']);inside=set()
        def walk(n):
            if n in leafset or n in inside:return
            inside.add(n)
            for lit in aig.fanin_literals.get(n,()):walk(lit>>1)
        walk(entry['output_aiger_var'])
        def path(n,seen):
            if n in leafset or n in seen:return None
            if n==serial_target:return [n]
            seen.add(n)
            for lit in aig.fanin_literals.get(n,()):
                found=path(lit>>1,seen)
                if found:return [n]+found
            return None
        witness=path(entry['output_aiger_var'],set())
        assert witness and serial_target in inside
        verified=entry['output_aiger_var'] if variant=='baseline' else reverse.get(entry['output_aiger_var'],'NOT_PROVEN')
        entry['verified_original_output']=verified
        entry['aiger_internal_witness']=witness
        table+=f'| {entry["output_aiger_var"]} | {verified} | {entry["phase"]} | {entry["super"]["id"]} | {entry["super"]["root_cell"]} | {entry["leaves"]} | {entry["arrival"]["worst"]} | {sorted(inside)} | {witness} |\n'
    details=json.dumps(cover,indent=2)
    (out/f'{stem}_final_cover_{suffix}.md').write_text(
        f'# {variant}: final cover proof\n\nTarget mapper node {b["mapper_id"]}; active phase refs {b["refs"]}. '
        'The listed covers are selected (nRefAct>0); mapper-subject DFS inside their cut boundaries reaches the target as an internal node. '
        'The accompanying witness is independently reconstructed with the existing AIG parser, using the exact serialized input for this run. '
        'No mapper marks or references are modified.\n\n'+table+'\n```json\n'+details+'\n```\n')
    target_cuts=[r for r in cuts if r['target_cut']]
    inventory=[r for r in records if r['type']=='match_inventory' and r['cut_index'] in [c['cut_index'] for c in target_cuts]]
    summary=dict(variant=variant,function_point='original_AIG_function_197',mapper_node_id=b['mapper_id'],
                 num_cuts=len(cuts),num_nontrivial_cuts=len(cuts)-1,target_cut_present=bool(target_cuts),
                 target_cut_num_matches=len({r['super']['id'] for r in inventory}),
                 best_cut=b.get('original_verified_leaves',b.get('leaves')),best_supergate=b['super']['id'] if b['super'] else None,
                 best_root_cell=b['super']['root_cell'] if b['super'] else None,best_area_cost=b['area_cost'],
                 best_rise_arrival=b['node_arrival']['rise'],best_fall_arrival=b['node_arrival']['fall'],best_worst_arrival=b['node_arrival']['worst'],
                 used_in_final_cover=b['refs'][2]>0,
                 absorbing_cover_output=[r['output_aiger_var'] for r in cover if r['contains_target_internal']],
                 absorbing_supergate=[r['super']['id'] for r in cover if r['contains_target_internal']])
    return summary


def main():
    root=Path(sys.argv[1]);out=root/'output';out.mkdir(exist_ok=True)
    baseline=root/'input/c1355_orig.aig';candidate=root/'input/S002.aig'
    abc=Path('/data/wushangge/runs/CULS-source-pair-20260930-qU1wxi/bin/abc_pair_timing')
    assert sha(baseline)=='2602b10f131a6ba5996521036ba130aab128a4aa15e3ab49ce44f2f02047f0b0'
    assert sha(candidate)=='59dbe54c03d28c336775785e42d3e4781713e66e96769ac688fd38977672b5e1'
    trace=[json.loads(s) for s in (root/'input/S002.jsonl').read_text().splitlines()]
    accepted=[r for r in trace if r['final_status']=='ACCEPTED']
    assert len(accepted)==1 and accepted[0]['root']==197 and accepted[0]['cut_slot']==2 and accepted[0]['selected_subgraph']==2
    assert accepted[0]['cut_leaves']==[17,33,151,188]
    text=run(abc,f'read "{baseline}"; read "{candidate}"; cec -n',out/'rewrite_cec.log')
    assert 'Networks are equivalent' in text
    corr=function_correspondence(root,baseline,candidate,abc,out)
    correspondence=[];summary=[]
    for variant,aig in [('baseline',baseline),('candidate',candidate)]:
        records=observe(root,variant,aig,corr)
        summary.append(export(variant,records,out,corr))
        reverse={v[0]:k for k,v in corr.items()}
        for r in records:
            if r['type']=='node':
                original=r['aiger_var'] if variant=='baseline' else reverse[r['aiger_var']]
                correspondence.append(dict(variant=variant,original_function_point=original,serialized_aiger_var=r['aiger_var'],
                    abc_aig_id=r['abc_obj_id'],abc_aig_name=f'pi{original-1:02d}' if r['node_type']=='PI' else 'UNNAMED_INTERNAL_AIGER_NODE',
                    mapper_node_id=r['mapper_id'],node_type=r['node_type'],fanin_mapper_ids=r['fanin_mapper_ids'],
                    fanin_complemented=r['fanin_complemented'],correspondence_verified=True))
    csv_out(out/'mapper_node_correspondence.csv',correspondence)
    csv_out(out/'R9_mapper_internal_summary.csv',summary)
    write_json(out/'experiment_summary.json',dict(summaries=summary,function_correspondence=corr,
        status='PASS',source_commit='c738ed6e86e6ca10bcef0d4c649cc07d925b4be8'))


if __name__=='__main__':main()
