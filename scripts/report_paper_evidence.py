"""Regenerate compact publication tables, independently cross-checking frozen ABC outputs."""
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median

ROOT=Path(__file__).resolve().parents[1]
METHODS=('gradient','gradient_no_litecut','local_greedy','global_prefix','global_block','greedy','random','global_marginal')


def read(path):
    return json.loads(path.read_text())


def table(path,rows):
    if not rows:
        raise ValueError(f'Empty table {path.name}')
    columns=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def verified(row):
    work=Path(row['work'])
    if row['status']!='PASS':
        return
    ev=read(work/'evaluation/result.json')
    assert ev['cec']==row['cec']=='PASS'
    for key in ('delay_ps','area','input_sha256','critical_po_indices'):
        assert ev[key]==row[key]
    if row['edits'] is not None:
        replay=read(work/'replay/replay.json')
        assert replay['status']=='PASS' and replay['applied_rewrites']==len(row['edits'])
        assert [[i,c] for i,c in enumerate(replay['choices']) if c]==row['edits']


def label(circuit,row,original,noop,base):
    verified(row)
    ok=row['status']=='PASS'
    delay=row.get('delay_ps')
    chosen=ok and delay<original['delay_ps']
    delivered=row if chosen else original
    area_pct=(delivered['area']/original['area']-1)*100
    return dict(circuit=circuit,method=row['label'],edits=len(row['edits'] or []),
                evaluation_status=row['status'],cec=row.get('cec'),raw_ps=delay,raw_area=row.get('area'),
                original_ps=original['delay_ps'],noop_ps=noop['delay_ps'],b_ps=base,
                delivered_ps=delivered['delay_ps'],area_pct=area_pct,
                delivered_ps_if_area_cap_5pct=original['delay_ps'] if area_pct>5 else delivered['delay_ps'],
                delivered_vs_b_pct=(delivered['delay_ps']/base-1)*100,
                raw_aig_sha256=row.get('input_sha256'),work=row['work'])


def outputs(args):
    out=args.output
    out.mkdir(exist_ok=False,parents=True)
    with (ROOT/'analysis/paper14_fullflow/results.csv').open() as f:
        reference={r['circuit']:r for r in csv.DictReader(f)}
    singles=[]
    for p in sorted(args.singles.glob('*/results.json')):
        rows=read(p)
        assert len(rows)==19
        name=p.parent.name
        for row in rows:
            item=label(name,row,*rows[:2],float(reference[name]['baseline_b_ps']))
            item['edit_pairs']=json.dumps(row['edits'])
            singles.append(item)
    table(out/'isolated.csv',singles)

    methods=[]
    nested=[]
    pilots=[]
    ordinal=[]
    for pilot,dirs in ((False,args.selectors),(True,[args.pilot])):
        for directory in dirs:
            for p in sorted(directory.glob('*/selected.json')):
                name=p.parent.name
                rows=read(p)
                original,noop=rows[:2]
                base=float(reference[name]['baseline_b_ps'])
                freeze=p.parent/'frozen.json'
                if not freeze.exists():
                    freeze=args.frozen_system
                selections=read(freeze)['methods']
                for row in rows[2:]:
                    item=label(name,row,original,noop,base)
                    m=row['label']
                    if m.startswith('nested_'):
                        item['priority_order_size']=len(row['edits'])
                        item['pairs']=json.dumps(row['edits'])
                        nested.append(item)
                        continue
                    trace=selections[m]['trace']
                    sel=selections[m].get('selection',selections[m].get('selected'))
                    assert (row['edits'] or [])==sel
                    item.update(hard_calls=len(trace),search_seconds=selections[m]['seconds'])
                    oracle=p.parent/(m+'_oracle.json')
                    if oracle.exists():
                        candidates=read(oracle)
                        assert len(candidates)==len(trace)
                        for i,(candidate,t) in enumerate(zip(candidates,trace)):
                            verified(candidate)
                            edits=t.get('edits',t.get('pairs'))
                            assert (candidate['edits'] or [])==edits
                            ordinal.append(dict(circuit=name,method=m,stage='size_pilot' if pilot else 'controls',
                                  step=i,edits=len(edits),proxy_ps=t['proxy_ps'],real_ps=candidate.get('delay_ps'),
                                  status=candidate['status'],work=candidate['work']))
                        best=min([original['delay_ps']]+[c['delay_ps'] for c in candidates if c['status']=='PASS'])
                        item['oracle_deliverable_ps']=best
                        item['delivery_regret_ps']=item['delivered_ps']-best
                    (pilots if pilot else methods).append(item)
    for p in sorted(args.marginals.glob('*/marginal.json')):
        name=p.parent.name
        rows=read(p)
        freeze=read(p.parent/'marginal_freeze.json')
        item=label(name,rows[2],*rows[:2],float(reference[name]['baseline_b_ps']))
        assert (rows[2]['edits'] or [])==freeze['selection']
        methods.append(dict(**item,hard_calls=len(freeze['trace']),search_seconds=freeze['seconds']))
    assert len(set((r['circuit'],r['method']) for r in methods))==len(methods)
    for name in {r['circuit'] for r in methods}:
        assert {r['method'] for r in methods if r['circuit']==name}==set(METHODS)
    designs=[]
    aggregate=[]
    for name in sorted({r['circuit'] for r in methods}):
        arms={r['method']:r for r in methods if r['circuit']==name}
        gradient=arms['gradient']
        best_control=min(arms[m]['delivered_ps'] for m in METHODS if not m.startswith('gradient'))
        designs.append(dict(circuit=name,original_ps=gradient['original_ps'],noop_ps=gradient['noop_ps'],
                       b_ps=gradient['b_ps'],native_litecut_ps=reference[name]['litecut_ps'],
                       gradient_raw_ps=gradient['raw_ps'],gradient_area_pct=gradient['area_pct'],
                       gradient_area_cap_5pct_ps=gradient['delivered_ps_if_area_cap_5pct'],
                       **{m+'_delivered_ps':r['delivered_ps'] for m,r in arms.items()},
                       best_nongradient_delivered_ps=best_control,
                       gradient_vs_best_nongradient_pct=(gradient['delivered_ps']/best_control-1)*100))
    for m in METHODS:
        rows=[r for r in methods if r['method']==m]
        def geo_change(base):
            return 100*math.expm1(sum(math.log(r['delivered_ps']/r[base]) for r in rows)/len(rows))
        aggregate.append(dict(method=m,designs=len(rows),accepted=sum(r['delivered_ps']<r['original_ps'] for r in rows),
                         faster_than_b=sum(r['delivered_ps']<r['b_ps'] for r in rows),
                         accepted_and_faster_than_b=sum(r['delivered_ps']<min(r['b_ps'],r['original_ps']) for r in rows),
                         overtook_b=sum(r['delivered_ps']<r['b_ps']<=r['original_ps'] for r in rows),
                         failed_proposals=sum(r['evaluation_status']!='PASS' for r in rows),
                         gm_vs_original_pct=geo_change('original_ps'),gm_vs_b_pct=geo_change('b_ps'),
                         gm_area_change_pct=100*math.expm1(sum(math.log1p(r['area_pct']/100) for r in rows)/len(rows)),
                         above_five_pct_area=sum(r['area_pct']>5 for r in rows),
                         median_search_s=median(r['search_seconds'] for r in rows)))
    table(out/'development.csv',designs)
    table(out/'aggregate.csv',aggregate)
    owners=[]
    for p in sorted(args.owners.glob('*/owners.json')):
        data=read(p)
        rows=data['measurements']
        name=p.parent.name
        for row in rows[2:]:
            item=label(name,row,*rows[:2],float(reference[name]['baseline_b_ps']))
            item['all_edited_owners']=len(data['edits_by_owner'])
            owners.append(item)
    table(out/'owners.csv',owners)
    endpoints=[]
    for p in sorted(args.endpoints.glob('*/endpoint_summary.json')):
        data=read(p)
        endpoints.extend(dict(circuit=p.parent.name,true_noop_critical_po=data['true_noop_critical_po'],
                              proxy_noop_critical_po=data['proxy_noop_critical_po'],**r) for r in data['rows'])
    table(out/'endpoints.csv',endpoints)
    table(out/'selectors.csv',methods)
    table(out/'nested.csv',nested)
    table(out/'size_pilot.csv',pilots)
    table(out/'offline_oracles.csv',ordinal)
    table(out/'coverage.csv',[read(p) for p in sorted(args.coverage.glob('*.json'))])
    panel=read(args.panel)['rows']
    assert len(panel)==len({r['family'] for r in panel})==len({r['ordered_cones_sha256'] for r in panel})==24
    table(out/'validation_manifest.csv',panel)

    times=[]
    for directory in args.runtime:
        for p in sorted(directory.glob('trial_*/*.attempt.json')):
            attempt=read(p)
            assert attempt['exit_code']==0
            w=p.parent/p.name.removesuffix('.attempt.json')
            r=read(w/'result.json')
            meta=read(w/'inputs.json')
            prior=Path(attempt['reference'])
            old=read(prior/'result.json')
            case=prior.name
            assert r['cec']=='PASS' and meta['original_sha256']==read(prior/'inputs.json')['original_sha256']
            times.append(dict(circuit=case,device=meta['device'],alpha=meta['alpha'],trial=p.parent.name,host=attempt['host'],
                raw_wall_s=attempt['seconds'],archived_raw_wall_s=reference[case]['wall_s'],search_s=r['stages']['search'],
                cuda_peak_gib=r['cuda_peak_bytes']/2**30,cpu_peak_gib=r['process_maxrss_kib']/2**20,
                child_peak_gib=r['child_maxrss_kib']/2**20,
                vector_matches_archived=r['vector']==old['vector'],returned_bytes_match=r['selected_aig_sha256']==old['selected_aig_sha256'],
                original_ps=r['original_delay_ps'],delivered_ps=r['delay_ps'],status=r['status'],
                **{('stage_'+k):v for k,v in r['stages'].items()},**{('template_'+k):v for k,v in r['template_stages'].items()},work=str(w)))
    table(out/'runtime.csv',times)
    parser_rows=[]
    for p in sorted(args.parsers.glob('*.json')):
        rows=read(p)
        for filename in ('initial.v','templates.v'):
            a,b=[r for r in rows if r['filename']==filename]
            assert a['sha256']==b['sha256'] and a['gates']==b['gates']
            parser_rows.append(dict(circuit=p.stem,file=filename,gates=a['gates'],ordered_sha256=a['sha256'],
                            old_s=a['seconds'],indexed_s=b['seconds'],speedup=a['seconds']/b['seconds']))
    table(out/'parser.csv',parser_rows)
    fixed=[]
    for p in sorted(args.fixed.glob('*/devices.json')):
        v=read(p)
        for i,gr in enumerate(v['gradient_comparison']):
            fixed.append(dict(circuit=p.parent.name,logits=['zero','random'][i],
                cpu_hard_ps=v['cpu']['hard'][i],cuda_hard_ps=v['cuda']['hard'][i],
                **{d+'_'+k:x for d in ('cpu','cuda') for k,x in v[d]['soft'][i].items()},
                **gr,cuda_peak_bytes=v['cuda']['peak_bytes'],work=str(p.parent)))
    table(out/'devices_fixed.csv',fixed)
    cone=read(args.cone)
    replay=next(r for r in cone if r['label']=='noop')
    table(out/'system_cone.csv',[dict(**{k:v for k,v in r.items() if k!='cell_counts'},
         unchanged_signed_cone=r['signed_aig_cone_sha256']==replay['signed_aig_cone_sha256'],
         unchanged_cell_cone=r['mapped_cone_sha256']==replay['mapped_cone_sha256']) for r in cone])
    from plot_paper_ablation import main as plot
    plot(out,out/'joint_ablation.pdf')
    manifests={}
    inputs={}
    for field,value in vars(args).items():
        if field=='output':
            continue
        paths=value if isinstance(value,list) else [value]
        inputs[field]=[str(p) for p in paths]
        for p in paths:
            manifest=next(parent/'run.json' for parent in (p,*p.parents) if (parent/'run.json').is_file())
            raw=read(manifest)
            manifests[raw['slurm_job_id']]=dict(manifest=str(manifest),state=raw['state'],
                                  commit=raw['source_commit'],resources=raw['resources'],command=raw['command'])
    (out/'sources.json').write_text(json.dumps(dict(inputs=inputs,runs=manifests),indent=2)+'\n')



if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('singles','pilot','frozen-system','coverage','panel','parsers','fixed','owners','endpoints','cone','marginals','output'):
        ap.add_argument('--'+name,type=Path,required=True)
    for name in ('selectors','runtime'):
        ap.add_argument('--'+name,type=Path,nargs='+',required=True)
    outputs(ap.parse_args())
