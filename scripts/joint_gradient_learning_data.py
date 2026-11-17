#!/usr/bin/env python3
"""Freeze, materialize, and measure a bounded v3 training/validation panel.

Sampling sees source, graph size, family/hash and legal catalogue only. It never
uses mapped candidate QoR to choose circuits, edits, or replacements for failures.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from joint_gradient_panel import read,save,sha,restrict


def freeze(args):
    manifest=read(args.cohort)
    dev_ids={i for values in manifest['development12'].values() for i in values}
    dev_groups={r['group_id'] for r in manifest['circuits'] if r['circuit_id'] in dev_ids}
    rows=[dict(r,role='development_only' if r['group_id'] in dev_groups else r['split']) for r in manifest['circuits']]
    selected=[]
    # A bounded first learning pilot, explicitly excluding large graphs. This
    # bound is chosen before any new candidate timing is opened.
    for role in ['train','validation']:
        strata={}
        for r in rows:
            if r['role']==role and r['and_count']<=64000:
                strata.setdefault((r['input'].split('/')[0],r['size_bin']),[]).append(r)
        rng=random.Random(20261117+(role=='validation'))
        keys=sorted(strata);rng.shuffle(keys)
        for values in strata.values():rng.shuffle(values)
        chosen=[];used=set()
        while len(chosen)<12 and any(strata.values()):
            for key in keys:
                while strata[key]:
                    r=strata[key].pop()
                    if r['group_id'] not in used:
                        used.add(r['group_id']);chosen.append(r);break
                if len(chosen)==12:break
        if len(chosen)!=12:raise ValueError('Insufficient independent families for frozen pilot')
        selected.extend(chosen)
    save(args.output,{'schema':'joint_v3_learning_panel_v1','seed':20261117,'cohort_sha256':sha(args.cohort),
        'reference_run':str(args.reference_run),'development_ids':sorted(dev_ids),'development_groups':sorted(dev_groups),
        'circuits':rows,'selected':selected,'maximum_new_mappings':216,'nonempty_per_circuit':8,
        'sampling':'source/size strata, one family per role, <=64000 ANDs; no candidate timing consulted',
        'limitations':['Existing 905 aggregate outcomes previously inspected; not a pristine blind benchmark.',
                       'Pilot excludes >64000-AND graphs; no large-circuit generalization claim.']})


def sampled_catalogue(source, library):
    from global_timing.candidates import Catalogue,conflict_reason
    from global_timing.saved_arrays import iter_saved_array
    from global_timing.feasibility import CandidateFeasibility
    from global_timing.structure import parse_aig,make_partition
    # Reservoir sampling spans the full verified decision domain. Candidate
    # alternatives are then sampled from their complete subgraph list.
    rng=random.Random(20261117);reservoir=[];seen=0
    path=source/'work/cut_validation/verified_catalogue.json'
    for i,d in enumerate(iter_saved_array(path,'decisions')):
        if not d['candidate_subgraph_indices']:continue
        seen+=1
        if len(reservoir)<128:reservoir.append((i,d))
        else:
            j=rng.randrange(seen)
            if j<128:reservoir[j]=(i,d)
    decisions=[];indices=[];choice_sources=[]
    for i,d in sorted(reservoir):
        d=copy.deepcopy(d);cs=list(range(1,d['candidate_count']+1));rng.shuffle(cs);cs=sorted(cs[:2])
        d['candidate_subgraph_indices']=[d['candidate_subgraph_indices'][c-1] for c in cs];d['candidate_count']=len(cs)
        decisions.append(d);indices.append(i);choice_sources.append(cs)
    aig=parse_aig(source/'input.aig');partition=make_partition(aig)
    conflicts=[(i,j) for i in range(len(decisions)) for j in range(i) if conflict_reason(decisions[i],decisions[j])]
    cat=Catalogue(decisions,conflicts,{r:[] for r in range(len(partition.regions))})
    for i,d in enumerate(decisions):cat.region_decisions[d['decision_owner_region']].append(i)
    rule=CandidateFeasibility(aig,cat,library)
    allowed=[(i,c) for i,row in enumerate(rule.allowed()) for c,ok in enumerate(row) if c and ok]
    rng.shuffle(allowed)
    vectors=[];kinds=[]
    def add(edits,kind):
        v=[0]*len(decisions)
        for i,c in edits:v[i]=c
        try:rule.validate(v)
        except ValueError:return False
        if not any(v) or v in vectors:return False
        vectors.append(v);kinds.append(kind);return True
    for e in allowed:
        if add([e],'single') and len(vectors)==2:break
    # Include an alternate candidate at the same root when available.
    for i,c in allowed:
        if any(v[i] and v[i]!=c for v in vectors):
            if add([(i,c)],'alternative_at_same_decision'):break
    edges=set(partition.edges)
    for relationship in ['serial_regions','parallel_regions']:
        found=False
        for a in allowed:
            for b in allowed:
                if a[0]>=b[0]:continue
                ra=decisions[a[0]]['decision_owner_region'];rb=decisions[b[0]]['decision_owner_region']
                serial=(ra,rb) in edges or (rb,ra) in edges
                # Parallel here means no reachability in either direction,
                # checked explicitly, not merely lack of an immediate edge.
                if relationship=='serial_regions' and not serial:continue
                if relationship=='parallel_regions':
                    if ra==rb:continue
                    adjacency={}
                    for u,v in edges:adjacency.setdefault(u,[]).append(v)
                    def reaches(u,v):
                        pending=[u];visited=set()
                        while pending:
                            n=pending.pop()
                            if n==v:return True
                            if n not in visited:visited.add(n);pending.extend(adjacency.get(n,[]))
                        return False
                    if reaches(ra,rb) or reaches(rb,ra):continue
                if add([a,b],relationship):found=True;break
            if found:break
    for attempt in range(128):
        if len(vectors)>=8:break
        order=allowed.copy();rng.shuffle(order);chosen=[]
        for i,c in order:
            if any(j==i for j,_ in chosen):continue
            v=[0]*len(decisions)
            for j,k in chosen+[(i,c)]:v[j]=k
            try:rule.validate(v)
            except ValueError:continue
            chosen.append((i,c))
            if len(chosen)>=min(2+attempt%7,8):break
        add(chosen,'legal_joint')
    used=sorted({i for v in vectors for i,c in enumerate(v) if c})
    wanted={indices[i]:{0, *choice_sources[i]} for i in used}
    lookup={indices[i]:i for i in used};score_path=read(source/'work/litecut.json')['scores_path'];scores=[]
    for r in iter_saved_array(score_path,'scores'):
        i,c=r['decision_index'],r['choice_index']
        if i in wanted and c in wanted[i]:
            index=lookup[i];scores.append(dict(r,decision_index=index,choice_index=0 if c==0 else choice_sources[index].index(c)+1))
        if len(scores)==sum(len(v) for v in wanted.values()):break
    if len(scores)!=sum(len(v) for v in wanted.values()):raise ValueError('Missing saved LiteCut features')
    small,scores=restrict(cat,scores,used,len(partition.regions))
    combinations=[{'index':0,'kind':'serialized_noop','choices':[0]*len(used)}]
    combinations += [dict(index=k+1,kind=kind,choices=[v[i] for i in used]) for k,(v,kind) in enumerate(zip(vectors,kinds))]
    return small,scores,combinations,{'full_decisions_seen':seen,'reservoir_decisions':len(reservoir),
           'legal_reservoir_choices':len(allowed),'source_indices':[indices[i] for i in used],
           'source_choices':[choice_sources[i] for i in used],'scores_path':score_path}


def prepare(args):
    manifest=read(args.manifest)
    record=next(r for r in manifest['selected'] if r['circuit_id']==args.circuit)
    source=Path(manifest['reference_run'])/'output'/f'case_{args.circuit:04d}_Ours'
    if sha(source/'input.aig')!=record['input_sha256']:raise ValueError('Input hash changed')
    lib=ROOT/'projects/culs/src/algorithms/rewrite_library.inc'
    started=time.perf_counter();cat,scores,combos,audit=sampled_catalogue(source,lib)
    args.output.mkdir(parents=True,exist_ok=False)
    save(args.output/'catalogue.json',dict(decisions=cat.decisions,conflicts=cat.conflicts,region_decisions=cat.region_decisions))
    save(args.output/'scores.json',{'scores':scores})
    save(args.output/'baseline.json',read(source/'work/original/result.json'))
    save(args.output/'frozen.json',{'schema':'joint_v3_frozen_combinations_v1',**record,'source':str(source),
         'combinations':combos,'sampling_audit':audit,'manifest_sha256':sha(args.manifest),
         'source_commit':os.environ.get('V3_SOURCE_COMMIT'),'preparation_s':time.perf_counter()-started,
         'scope':'Full-domain reservoir, no candidate measured timing or historical selected edits used for sampling'})
    # Reuse the already measured historical combination as an additional
    # example only AFTER freezing the new domain. Its labels cannot influence
    # the sampler above. Preserve incomplete/missing records as exclusions.
    from joint_gradient_panel import load_subset
    from global_timing.structure import parse_aig
    from global_timing.boolean_candidates import parse_library
    from global_timing.phase_ports import hard_phase_arrivals
    from diagnose_joint_gradient_structure_v3 import structural_arrivals
    summary=read(source/'work/summary.json');final=summary.get('final_candidate') or {}
    metrics=final.get('metrics') or {}
    if metrics.get('cec')=='PASS' and metrics.get('timing_model')=='ABC_SCL_LIBERTY' and final.get('applied_rewrites',0)>0:
        old,saved_scores,source_indices,score_path=load_subset(source,minimum_decisions=1)
        aig=parse_aig(source/'input.aig');library=parse_library(lib)
        predicted=hard_phase_arrivals(aig,old.decisions,[1]*len(old.decisions),library)
        actual=structural_arrivals(parse_aig(Path(final['aig'])))
        error=max(abs(a-b) for a,b in zip(predicted,actual))
        save(args.output/'historical.json',{'catalogue':dict(decisions=old.decisions,conflicts=old.conflicts,region_decisions=old.region_decisions),
            'scores':saved_scores,'choices':[1]*len(old.decisions),'metrics':metrics,
            'structural_max_error':error,'area_feasible':metrics['area']<=summary['original']['area']*1.05,
            'source_indices':source_indices,'source_summary':str(source/'work/summary.json'),'source_summary_sha256':sha(source/'work/summary.json'),
            'installed_aig':final['aig'],'installed_sha256':sha(Path(final['aig']))})
    else:
        save(args.output/'historical_excluded.json',{'reason':'No complete CEC-PASS measured nonempty combination',
             'source':str(source/'work/summary.json'),'status':final.get('status')})
    print(json.dumps({'circuit':args.circuit,'prepared':len(combos),'seconds':time.perf_counter()-started}),flush=True)


def measure(args):
    from global_timing.candidates import Catalogue
    from global_timing.execution import replay_many
    from global_timing.evaluation import Evaluator
    from global_timing.structure import parse_aig
    from global_timing.boolean_candidates import parse_library
    from global_timing.phase_ports import hard_phase_arrivals
    from diagnose_joint_gradient_structure_v3 import structural_arrivals
    out=args.output;frozen=read(out/'frozen.json');digest=sha(out/'frozen.json');source=Path(frozen['source'])
    context=read(source/'context.json');raw=read(out/'catalogue.json')
    cat=Catalogue(raw['decisions'],raw['conflicts'],{int(k):v for k,v in raw['region_decisions'].items()})
    vectors=[r['choices'] for r in frozen['combinations']]
    destinations=[out/'measurements'/f'combo_{r["index"]:02d}'/'rewrite' for r in frozen['combinations']]
    for p in destinations:p.parent.mkdir(parents=True,exist_ok=False)
    os.environ['CULS_ALLOW_NONPOSITIVE_CANDIDATES']='1'
    executions=replay_many(args.gpuls,source/'input.aig',source/'work/partition/owners.csv',cat,vectors,
                           out/'batch_replay',destinations,timeout=900,prepared=True,allow_nonpositive_gain=True)
    lib=parse_library(ROOT/'projects/culs/src/algorithms/rewrite_library.inc');aig=parse_aig(source/'input.aig')
    evaluator=Evaluator(Path(context['tools']['abc']['path']),Path(context['tools']['library']['path']),source/'input.aig',900)
    results=[]
    for combination,execution,destination in zip(frozen['combinations'],executions,destinations):
        row={**combination,'execution':execution,'status':execution['status']}
        if execution['status']=='PASS':
            pred=hard_phase_arrivals(aig,cat.decisions,combination['choices'],lib)
            actual=structural_arrivals(parse_aig(Path(execution['aig'])))
            if len(pred)!=len(actual):raise ValueError('Replay PO count changed')
            row['structural_max_error']=max(abs(a-b) for a,b in zip(pred,actual))
            # An installation representation error cannot be learned away.
            if row['structural_max_error']>1e-8:
                row.update(status='STRUCTURAL_MISMATCH',mapping_skipped=True)
            else:
                try:
                    row['metrics']=evaluator.evaluate(Path(execution['aig']),destination.parent/'evaluation')
                    row['area_feasible']=row['metrics']['area']<=read(out/'baseline.json')['area']*1.05
                except Exception as exc:row.update(status='FAIL',error=str(exc))
        save(destination.parent/'observation.json',row);results.append(row)
        print(json.dumps({'circuit':frozen['circuit_id'],'index':row['index'],'status':row['status']}),flush=True)
    if sha(out/'frozen.json')!=digest:raise ValueError('Frozen vectors changed')
    save(out/'measurements.json',{'results':results,'frozen_sha256':digest,'gpuls_sha256':sha(args.gpuls),
        'tools':context['tools'],'new_mapping_calls':len(list(out.glob('measurements/*/evaluation/mapping.command.json')))})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['freeze','prepare','measure'])
    p.add_argument('--cohort',type=Path);p.add_argument('--reference-run',type=Path);p.add_argument('--manifest',type=Path)
    p.add_argument('--circuit',type=int);p.add_argument('--output',type=Path,required=True);p.add_argument('--gpuls',type=Path)
    a=p.parse_args()
    if a.phase!='freeze' and (not os.environ.get('SLURM_JOB_ID') or not ROOT.is_relative_to('/data/wushangge')):
        p.error('Use committed /data source under Slurm')
    globals()[a.phase](a)


if __name__=='__main__':main()
