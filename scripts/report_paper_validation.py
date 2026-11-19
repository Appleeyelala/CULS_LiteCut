"""Summarize the sealed panel once, keeping incomplete circuits and primary/control identities."""
import argparse
import math
import json
from pathlib import Path

from paper_joint_evidence import digest, read
from report_paper_evidence import METHODS, table, verified


def collect(panel, shards):
    cases={}
    for shard in shards:
        freeze=read(shard/'freeze.json')
        assert freeze['arm']=='raw_fixed' and freeze['manifest_sha256']==digest(panel)
        failures={f['index']:f['error'] for f in read(shard/'failures.json')}
        for index in freeze['indices']:
            if index in cases:
                raise ValueError('Overlapping validation shards')
            cases[index]=(shard/f'case_{index:02d}',failures.get(index))
    designs=read(panel)['rows']
    assert sorted(cases)==list(range(len(designs))) and len(designs)==24
    info=[]
    methods=[]
    for i,design in enumerate(designs):
        w,error=cases[i]
        base=read(w/'baseline.json') if (w/'baseline.json').is_file() else {}
        raw=read(w/'raw/result.json') if (w/'raw/result.json').is_file() else {}
        attempt=read(w/'attempt.json') if (w/'attempt.json').is_file() else {}
        common=dict(index=i,family=design['family'],ands=design['ands'],aig_sha256=design['aig_sha256'])
        if raw:
            source=read(w/'raw/inputs.json')
            assert source['original_sha256']==design['aig_sha256'] and source['alpha']==.5 and source['method']=='gradient'
            assert source['steps']==8 and source['max_edits']==16 and source['seed']==20261117
            assert raw['cec']=='PASS' and digest(Path(raw['returned_aig']))==raw['selected_aig_sha256']
        if base:
            assert base['cec']=='PASS'
        original=raw.get('original_delay_ps')
        b=base.get('delay_ps')
        methods.append(dict(**common,method='primary_raw_gradient',original_ps=original,b_ps=b,
            proposal_ps=raw.get('proposal_delay_ps'),delivered_ps=raw.get('delay_ps'),status=raw.get('status','INCOMPLETE'),
            delivered_cec=raw.get('cec'),area_pct=(100*(raw['area_ratio']-1) if raw else None),
            edits=(sum(bool(c) for c in raw['vector']) if raw else None),identity=raw.get('selected_aig_sha256')))
        controls=w/'controls'/design['family']
        available=(controls/'selected.json').is_file()
        direct=None
        noop=None
        selections={}
        if available:
            rows=read(controls/'selected.json')
            direct,noop=rows[:2]
            verified(direct)
            verified(noop)
            assert direct['status']=='PASS' and direct['delay_ps']==original
            selections={r['label']:r for r in rows[2:]}
            assert set(selections)==set(METHODS)
        for m in METHODS:
            row=selections.get(m)
            if row is not None:
                verified(row)
                delivery=row if row['status']=='PASS' and row['delay_ps']<original else direct
                data=dict(proposal_ps=row.get('delay_ps'),delivered_ps=delivery['delay_ps'],status=row['status'],
                          delivered_cec=delivery['cec'],area_pct=100*(delivery['area']/direct['area']-1),
                          edits=len(row['edits'] or []),identity=delivery['input_sha256'])
            else:
                data=dict(proposal_ps=None,delivered_ps=None,status='INCOMPLETE',delivered_cec=None,
                          area_pct=None,edits=None,identity=None)
            methods.append(dict(**common,method='control_'+m,original_ps=original,b_ps=b,**data))
        info.append(dict(**common,source=design['source'],stage_error=error,original_ps=original,
            noop_ps=noop.get('delay_ps') if noop else None,b_ps=b,primary_delivered_ps=raw.get('delay_ps'),
            raw_wall_s=attempt.get('seconds'),primary_status=raw.get('status'),controls_complete=available,
            **{m+'_delivered_ps':(min(original,r['delay_ps']) if r['status']=='PASS' else original)
               for m,r in selections.items()},work=str(w)))
    return info,methods


def main(out,panel,shards):
    out.mkdir()
    designs,methods=collect(panel,shards)
    table(out/'validation.csv',designs)
    table(out/'validation_methods.csv',methods)
    summary=[]
    for name in dict.fromkeys(r['method'] for r in methods):
        rows=[r for r in methods if r['method']==name]
        ok=[r for r in rows if r['delivered_ps'] is not None]
        paired=[r for r in ok if r['b_ps'] is not None]
        def change(items,base):
            if items:
                return 100*math.expm1(sum(math.log(r['delivered_ps']/r[base]) for r in items)/len(items))
        summary.append(dict(method=name,panel_n=len(rows),measured=len(ok),missing=len(rows)-len(ok),
                      fallback_after_failure=sum(r['status'] in ('FAIL','ORIGINAL_AFTER_TOOL_FAILURE') for r in rows),
                      faster_than_original=sum(r['delivered_ps']<r['original_ps'] for r in ok),
                      paired_b=len(paired),faster_than_b=sum(r['delivered_ps']<r['b_ps'] for r in paired),
                      overtook_b=sum(r['delivered_ps']<r['b_ps']<=r['original_ps'] for r in paired),
                      gm_vs_original_pct=change(ok,'original_ps'),gm_vs_b_pct=change(paired,'b_ps')))
    table(out/'validation_aggregate.csv',summary)
    sources=[]
    for shard in shards:
        manifest=next(parent/'run.json' for parent in shard.parents if (parent/'run.json').exists())
        m=read(manifest)
        sources.append(dict(manifest=str(manifest),job=m['slurm_job_id'],state=m['state'],commit=m['source_commit'],
                            command=m['command'],resources=m['resources'],freeze=read(shard/'freeze.json')))
    (out/'validation_sources.json').write_text(json.dumps(dict(panel=str(panel),panel_sha256=digest(panel),
                                                sources=sources),indent=2)+'\n')



if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    p.add_argument('manifest',type=Path)
    p.add_argument('shards',type=Path,nargs='+')
    a=p.parse_args()
    main(a.output,a.manifest,a.shards)
