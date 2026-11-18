"""Opt-in bounded ABC/GPU rewrite from a prepared catalogue and frozen LiteCut scores."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from global_timing.abc_choices import choice_program, symbolic_states
from global_timing.abc_sparse_timing import SparseTiming
from global_timing.boolean_candidates import parse_library
from global_timing.candidates import Catalogue, replay
from global_timing.common import digest, run_tool, save_json, tool_path
from global_timing.evaluation import Evaluator, cec_passed, parse_timing
from global_timing.feasibility import CandidateFeasibility
from global_timing.structure import parse_aig


def joint_search(model, rule, steps, edit_budget):
    zero=[0]*len(rule.catalogue.decisions)
    z=torch.zeros(model.state_count,device=model.device,dtype=model.dtype,requires_grad=True)
    opt=torch.optim.Adam([z],lr=.12)
    best,best_delay=zero,float(model.forward(z.detach(),model.encode(zero)))
    if best_delay >= 1.e11: raise ValueError("No complete hard timing path for the original")
    history=[]
    for _ in range(steps):
        vals=z.detach().cpu().tolist()
        offers=[];offset=0
        for group in model.groups:
            offers.extend((vals[offset+k]-vals[offset],i,c) for k,(i,c) in enumerate(group) if k)
            offset+=len(group)
        vector=zero.copy();occupied=set()
        for margin,i,c in sorted(offers,key=lambda item:(-item[0],item[1],item[2])):
            root=rule.catalogue.decisions[i]["root"]
            if margin<=0 or len(occupied)>=edit_budget:break
            if root in occupied:continue
            vector[i]=c
            try:
                rule.validate(vector)
            except ValueError:
                vector[i]=0
                continue
            occupied.add(root)
        loss=model.straight_through(z,model.encode(vector))
        delay=float(loss.detach())
        if delay < best_delay:
            best,best_delay=vector.copy(),delay
        opt.zero_grad()
        if not loss.requires_grad:
            history.append(dict(choices=vector,hard=delay,gradient_norm=0.))
            break
        loss.backward()
        if not torch.isfinite(z.grad).all():raise ValueError("Nonfinite joint gradient")
        history.append(dict(choices=vector,hard=delay,gradient_norm=float(z.grad.norm())))
        opt.step()
    return best,history


def run(args):
    start=time.perf_counter()
    out=args.output
    out.mkdir(exist_ok=False)
    cat=Catalogue(**json.loads(args.catalogue.read_text()))
    scores=json.loads(args.scores.read_text())["scores"]
    lib=parse_library(ROOT/"projects/culs/src/algorithms/rewrite_library.inc")
    aig=parse_aig(args.original)
    rule=CandidateFeasibility(aig,cat,lib)
    zero=[0]*len(cat.decisions);singletons=[zero]
    for i,row in enumerate(rule.allowed()):
        for c in range(1,len(row)):
            if row[c]:
                vector=zero.copy();vector[i]=c
                singletons.append(vector)
    groups=symbolic_states(cat.decisions,singletons)
    save_json(out/"inputs.json",{k:{"path":str(v),"sha256":digest(v)} for k,v in vars(args).items()
                                 if isinstance(v,Path) and v.is_file()})
    work=out/"original";work.mkdir()
    cec=run_tool([args.frozen_abc,"-c",f"cec -n {tool_path(args.original)} {tool_path(args.original)}"],work,"cec")
    if not cec_passed(cec):raise ValueError("Original CEC failed")
    env=dict(os.environ,CULS_BASE_ANCHORS=str(work/"anchors"),CULS_GLOBAL_TIMING_REPORT="1")
    base_cmd=f"read_lib {tool_path(args.liberty)}; read_aiger {tool_path(args.original)}; map"
    raw=run_tool([args.abc,"-c",base_cmd+"; topo; stime -p"],work,"mapping",900,env)
    original=parse_timing(raw)
    vector=zero;trace=[]
    if groups:
        prep=out/"prepare";prep.mkdir()
        lines,_=choice_program(aig,cat.decisions,singletons,scores,lib,args.alpha,symbolic=True)
        spec=prep/"choices.txt";spec.write_text("\n".join(lines)+"\n")
        bank=prep/"bank.txt"
        run_tool([args.abc,"-c",base_cmd],prep,"local_matches",900,dict(os.environ,CULS_CHOICES=str(spec),
                 CULS_PREPARE_BANK=str(bank),CULS_BANK_ANCHORS=str(work/"anchors")))
        model=SparseTiming(bank,work/"anchors",groups,args.device)
        vector,trace=joint_search(model,rule,args.steps,args.max_edits)
    save_json(out/"frozen_proposal.json",dict(vector=vector,trace=trace,alpha=args.alpha,device=args.device))
    final=original;count=1;status="ORIGINAL";candidate=None
    if any(vector):
        rule.validate(vector)
        installed,_=replay(args.gpuls,args.original,args.owners,cat,vector,out/"replay",900,
                           allow_nonpositive_gain=True,allow_nonpositive_candidates=True)
        count+=1
        candidate=Evaluator(args.frozen_abc,args.liberty,args.original,900).evaluate(installed,out/"final")
        if candidate["area"]<=1.05*original["area"] and candidate["delay_ps"]<original["delay_ps"]:
            final=candidate;status="ACCEPTED"
        else:status="ORIGINAL_AFTER_FINAL_CHECK"
    row=dict(status=status,cec="PASS",mapped_evaluations=count,precover_calls=int(bool(groups)),
             prepared_catalogue_seconds=time.perf_counter()-start, original_delay_ps=original["delay_ps"],
             delay_ps=final["delay_ps"],area_ratio=final["area"]/original["area"],vector=vector,
             candidate_delay_ps=candidate["delay_ps"] if candidate else None,
             scope="Starts from supplied catalogue, owners and Student scores; upstream costs excluded.")
    save_json(out/"result.json",row)
    return row


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for n in ("original","catalogue","scores","owners","abc","frozen-abc","liberty","gpuls","output"):
        p.add_argument("--"+n,type=Path,required=True)
    p.add_argument("--steps",type=int,default=8);p.add_argument("--max-edits",type=int,default=16)
    p.add_argument("--alpha",type=float,choices=[0.,.25,.5,1.],default=0.)
    p.add_argument("--device",choices=["cpu","cuda"],default="cuda")
    a=p.parse_args()
    if (not os.getenv("SLURM_JOB_ID") or not a.output.is_absolute() or
            not a.output.resolve().is_relative_to("/data/wushangge") or a.steps<1 or a.max_edits<0):
        p.error("Use a dedicated /data Slurm output and valid search budgets")
    for k,v in vars(a).items():
        if isinstance(v,Path):setattr(a,k,v.resolve())
    torch.set_num_threads(1)
    print(json.dumps(run(a)))

