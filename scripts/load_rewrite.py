"""Run a raw or prepared AIG through local-cover preparation, GPU search and final CEC."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import time
from types import SimpleNamespace
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from global_timing.boolean_candidates import parse_library
from global_timing.candidates import Catalogue, enumerate_candidates
from global_timing.common import digest, run_tool, save_json, tool_path
from global_timing.cut_validation import validate_catalogue
from global_timing.evaluation import cec_passed, parse_timing
from global_timing.feasibility import CandidateFeasibility
from global_timing.liberty_cover import Liberty
from global_timing.litecut import score_catalogue
from global_timing.load_bank import LoadBank
from global_timing.load_search import optimize
from global_timing.structure import make_partition, parse_aig
from abc_sparse_rewrite import evaluate_proposal


def read(p):
    return json.loads(p.read_text())


def design(args, original, out):
    start=time.monotonic()
    out.mkdir(parents=True,exist_ok=False)
    stages={}
    def stage(name,fn,*a,**kw):
        begin=time.monotonic()
        try:
            return fn(*a,**kw)
        finally:
            stages[name]=time.monotonic()-begin
    required=['abc','frozen_abc','gpuls','liberty']
    if args.alpha:
        required.append('checkpoint')
    paths={k:Path(args.tools[k]).resolve(strict=True) for k in required}
    tools=SimpleNamespace(**paths)
    save_json(out/'inputs.json',dict(original=str(original),original_sha256=digest(original),
        tools={k:dict(path=str(p),sha256=digest(p)) for k,p in paths.items()},mode=args.mode,
        method=args.method,alpha=args.alpha,source_snapshot=str(ROOT),slurm_id=os.getenv('SLURM_JOB_ID'),
        seed=args.seed,steps=args.steps,seconds=args.seconds,max_edits=args.max_edits,
        device=args.device,adaptive_sizes=args.adaptive_sizes,anchor_outputs=args.anchor_outputs,prepared=str(args.prepared) if args.prepared else None))
    if args.device=='cuda': torch.cuda.reset_peak_memory_stats()
    aig=stage('read_aig',parse_aig,original)
    if args.prepared:
        f=args.prepared
        manifest=read(f/'prepared.json')
        if manifest['original_sha256']!=digest(original): raise ValueError('Prepared catalogue input mismatch')
        for n,h in manifest['files'].items():
            if digest(f/n)!=h: raise ValueError('Changed prepared input: '+n)
        owners=f/'owners.csv'
        cat=Catalogue(**read(f/'catalogue.json')); scores=read(f/'scores.json')['scores']
        if args.alpha and manifest.get('checkpoint_sha256')!=digest(tools.checkpoint):
            scores=[]
    else:
        part=stage('partition',make_partition,aig)
        part.write(out/'partition')
        owners=out/'partition/owners.csv'
        raw=stage('candidate_enumeration',enumerate_candidates,tools.gpuls,original,part,out/'candidates',900,True)
        cat,_=stage('candidate_validation',validate_catalogue,tools.frozen_abc,part,raw,out/'cut_validation',900)
        scores,_=stage('student',score_catalogue,original,part,cat,tools.checkpoint,out/'litecut_cache') if args.alpha and cat.decisions else ([],None)
        f=out/'prepared'; f.mkdir()
        shutil.copyfile(owners,f/'owners.csv')
        save_json(f/'catalogue.json',asdict(cat));save_json(f/'scores.json',{'scores':scores})
        save_json(f/'prepared.json',dict(original_sha256=digest(original),
            checkpoint_sha256=digest(tools.checkpoint) if args.alpha else None,files={n:digest(f/n) for n in ('owners.csv','catalogue.json','scores.json')}))
    if args.alpha and cat.decisions and not scores:
        part=stage('partition',make_partition,aig)
        scores,_=stage('student',score_catalogue,original,part,cat,tools.checkpoint,out/'litecut_cache')
    rules=stage('rewrite_library',parse_library,ROOT/'projects/culs/src/algorithms/rewrite_library.inc')
    rule=stage('feasibility',CandidateFeasibility,aig,cat,rules)
    work=out/'original';work.mkdir()
    tables=work/'tables'
    if not cec_passed(stage('original_cec',run_tool,[tools.frozen_abc,'-c',f'cec -n {tool_path(original)} {tool_path(original)}'],work,'cec')):
        raise ValueError('Original CEC did not pass')
    cmd=f'read_lib {tool_path(tools.liberty)}; read_aiger {tool_path(original)}; map; topo; stime -p; write_verilog {tool_path(work/"mapped.v")}'
    log=stage('original_map',run_tool,[tools.abc,'-c',cmd],work,'mapping',900,dict(os.environ,CULS_GLOBAL_TIMING_REPORT='1',CULS_LIB_TABLES=str(tables)))
    baseline=parse_timing(log)
    zero=[0]*len(cat.decisions)
    vector=zero
    trace=dict(method=args.method,trace=[],unique_vectors=1,hard_calls=0)
    area_limit=1.05 if args.mode=='area' else None
    nstates=0; once=0
    if any(any(row[1:]) for row in rule.allowed()):
        tech=stage('load_timing_tables',Liberty,tables,args.device)
        bank=stage('template_preparation',LoadBank,aig,cat,rule,scores,rules,tech,tools.abc,tools.liberty,out/'templates',args.alpha)
        nstates=bank.states;once=1
        observed = [[po['rise_ps'],po['fall_ps']] for po in baseline['po_rise_fall']] if args.anchor_outputs else None
        vector,trace=stage('search',optimize,bank,rule,args.method,args.steps,args.seconds,args.max_edits,area_limit,args.seed,
                           adaptive_sizes=args.adaptive_sizes,observed_po=observed)
    save_json(out/'proposal.json',dict(vector=vector,**trace))
    call=SimpleNamespace(output=out,original=original,owners=owners,area_limit=area_limit,**paths)
    row=stage('final_replay_and_validation',evaluate_proposal,call,cat,rule,vector,baseline)
    dest=out/'selected.aig';shutil.copyfile(row['selected_aig'],dest)
    if digest(dest)!=row['selected_aig_sha256']: raise ValueError('Returned AIG changed')
    row.update(returned_aig=str(dest),proposal_delay_ps=row['candidate_delay_ps'] if any(vector) else row['original_delay_ps'],
        proposal_area_ratio=row['candidate_area_ratio'] if any(vector) else 1.,
        mode=args.mode,method=args.method,template_preparations=once,template_match_passes=2*once,decisions=len(cat.decisions),local_states=nstates,
        search_seconds=trace.get('seconds',0.),unique_vectors=trace['unique_vectors'],stages=stages,
        template_stages=bank.preparation_times if once else {},
        cuda_peak_bytes=torch.cuda.max_memory_allocated() if args.device=='cuda' else 0,
        process_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        child_maxrss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        elapsed_s=time.monotonic()-start,timing_scope='prepared catalogue' if args.prepared else 'raw AIG')
    save_json(out/'result.json',row)
    return row


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original',nargs='+',type=Path,required=True)
    p.add_argument('--tools',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prepared',type=Path)
    p.add_argument('--mode',choices=['delay','area'],default='delay')
    p.add_argument('--method',choices=['gradient','greedy','random'],default='gradient')
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--alpha',choices=[0.,.5],type=float,default=0.)
    p.add_argument('--adaptive-sizes',action='store_true',help='Explore edit capacities within the same gradient search budget')
    p.add_argument('--anchor-outputs',action='store_true',help='Align no-edit PO rise/fall to direct ABC mapping before scoring changes')
    p.add_argument('--steps',type=int,default=8);p.add_argument('--seconds',type=float)
    p.add_argument('--seed',type=int,default=20261117);p.add_argument('--max-edits',type=int,default=16)
    a=p.parse_args()
    if (not os.getenv('SLURM_JOB_ID') or not a.output.resolve().is_relative_to('/data/wushangge')
            or a.steps<1 or a.max_edits<0 or (a.seconds is not None and a.seconds<=0)
            or (a.adaptive_sizes and a.method!='gradient') or (a.prepared and len(a.original)!=1)):
        p.error('Use a dedicated /data Slurm output, positive budgets and one input for prepared mode')
    torch.set_num_threads(1)
    a.tools=read(a.tools);a.output=a.output.resolve()
    if len(a.original)==1:
        design(a,a.original[0].resolve(strict=True),a.output)
    else:
        a.output.mkdir(exist_ok=False,parents=True)
        records=[]
        for i,src in enumerate(a.original):
            work=a.output/f'case_{i:04d}'
            try:
                row=design(a,src.resolve(strict=True),work)
                records.append(dict(input=str(src),result=str(work/'result.json'),status=row['status']))
            except Exception as exc:
                records.append(dict(input=str(src),error=repr(exc),status='FAILED'))
        save_json(a.output/'batch.json',records)
        if any(r['status']=='FAILED' for r in records): raise SystemExit(1)

if __name__=='__main__':
    main()
