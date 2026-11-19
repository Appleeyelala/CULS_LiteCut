"""Verify frozen-vector identity and summarize proposal, delivery and serialization effects."""
import csv
import json
from pathlib import Path
import sys

from paper14_frozen_choices import COUNTS, read, require, vector_hash
from global_timing.common import digest

def csv_file(path, rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w') as stream:
        writer=csv.DictWriter(stream,fields,lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)

def sign(value):
    return (value>0)-(value<0)

def report(audit, out):
    out.mkdir(exist_ok=True,parents=True)
    manifest=read(audit/'freeze.json')
    records=[]
    decisions=[]
    for name,case in manifest.items():
        old=Path(case['old']); log=old/'proposal.json'
        require(digest(log)==case['proposal_sha256'],name+' changed search')
        search=read(log); trace=search['trace']
        rows=[]
        for item in [dict(id='original',steps=[],edits=[],chosen=False)]+case['choices']:
            key=item['id']; p=audit/name/key; attempt=read(p/'attempt.json')
            r=dict(circuit=name,configuration=key,steps='|'.join(map(str,item['steps'])),edits=len(item['edits']),
                   chosen=item['chosen'],proxy_ps=item.get('proxy_ps'),vector_sha256=item.get('sha256'),
                   status=attempt['status'],measurement=str(p))
            if key!='original':
                for step in item['steps']:
                    require(vector_hash(trace[step]['vector'])==item['sha256'],name+' vector')
                replay=read(p/'replay/replay.json')
                r.update(requested=len(replay['requested']),applied=replay.get('applied_rewrites'))
            if attempt['status']=='PASS':
                m=read(p/'evaluation/result.json')
                require(m['cec']=='PASS' and m['input_sha256']==digest(Path(attempt['aig'])),name+' evaluation')
                if key!='original':require(r['requested']==r['applied']==r['edits'],name+' incomplete replay')
                r.update(cec=m['cec'],delay_ps=m['delay_ps'],area=m['area'],critical_po='|'.join(map(str,m['critical_po_indices'])),
                         aig_sha256=m['input_sha256'],po_record=str(p/'evaluation/result.json'))
            else:
                r['error']=attempt['error']
            rows.append(r)
        nonempty=[r for r in rows if r['edits'] and r['status']=='PASS']
        original,noop=rows[:2]
        require(noop['configuration']=='noop' and len(rows)==COUNTS[name]+2,name+' row count')
        o,z=original['delay_ps'],noop['delay_ps']
        for r in rows:
            if r['status']!='PASS':continue
            r.update(vs_original_ps=r['delay_ps']-o,vs_noop_ps=r['delay_ps']-z)
            if r['edits']:
                r['baseline_direction_differs']=sign(r['vs_original_ps'])!=sign(r['vs_noop_ps'])
                r['nonempty_rank']=1+sum(x['delay_ps']<r['delay_ps'] for x in nonempty)
        selected=next(r for r in rows if r['chosen'])
        best=min(nonempty,key=lambda r:r['delay_ps'])
        oracle=min([original]+nonempty,key=lambda r:r['delay_ps'])
        require(selected['status']=='PASS',name+' previous proposal failed')
        prev=read(old/'final/result.json')
        require(prev['delay_ps']==selected['delay_ps'] and prev['area']==selected['area']
                and prev['input_sha256']==selected['aig_sha256'],name+' historical mismatch')
        guarded=min(o,selected['delay_ps']); denom=oracle['delay_ps']
        switched=any(r['baseline_direction_differs'] for r in nonempty)
        if best['delay_ps']>=o:
            action='control_serialization_then_edit_count' if switched else 'test_nested_edit_counts'
        elif guarded>denom:
            action='trace_proxy_at_true_critical_outputs'
        else:
            action='retain_positive_control'
        decisions.append(dict(circuit=name,next_action=action,
                              selected_minus_best_nonempty_ps=selected['delay_ps']-best['delay_ps'],original_ps=o,noop_ps=z,noop_minus_original_ps=z-o,
                              selected=selected['configuration'],selected_ps=selected['delay_ps'],selected_rank=selected['nonempty_rank'],
                              valid_nonempty=len(nonempty),best_nonempty=best['configuration'],best_ps=best['delay_ps'],
                              best_ties='|'.join(r['configuration'] for r in nonempty if r['delay_ps']==best['delay_ps']),
                              best_deliverable=oracle['configuration'],best_deliverable_ps=denom,
                              proposal_regret_ps=selected['delay_ps']-denom,proposal_regret_pct=100*(selected['delay_ps']/denom-1),
                              delivered_ps=guarded,delivery_regret_ps=guarded-denom,delivery_regret_pct=100*(guarded/denom-1),
                              faster_than_original=sum(r['delay_ps']<o for r in nonempty),faster_than_noop=sum(r['delay_ps']<z for r in nonempty),
                              direction_changes='|'.join(r['configuration'] for r in nonempty if r['baseline_direction_differs'])))
        records.extend(rows)
    csv_file(out/'measurements.csv',records)
    csv_file(out/'decisions.csv',decisions)
    return decisions

if __name__=='__main__':
    print(json.dumps(report(*map(Path,sys.argv[1:])),indent=2))
