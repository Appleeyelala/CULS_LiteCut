"""Bounded CPU/CUDA shadow, gradient and equal-proposal diagnostic on frozen banks."""
import argparse
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from abc_choice_reference import read
from abc_mode0_panel import write_csv
from global_timing.abc_choices import choice_program, symbolic_states
from global_timing.abc_sparse_timing import SparseTiming
from global_timing.boolean_candidates import parse_library
from global_timing.candidates import Catalogue
from global_timing.common import digest, run_tool, save_json
from global_timing.structure import parse_aig


def optimize(model, vectors, indices, budget=8):
    states=[model.encode(v) for v in vectors]
    indices=list(indices)
    n=model.state_count
    seed=20261117
    rng=random.Random(seed)
    z=torch.zeros(n,device=model.device,dtype=model.dtype,requires_grad=True)
    cost=lambda k:float(model.forward(z.detach(),states[k]).detach())
    norm=lambda k:(sum(c!=0 for c in vectors[k]),k)
    def best(trace):
        return min(trace,key=lambda r:(r[1],norm(r[0])))[0]
    random_ids=rng.sample(indices,min(budget,len(indices)))
    random_trace=[(k,cost(k)) for k in random_ids]
    first=min(indices,key=norm)
    greedy=[(first,cost(first))]
    while len(greedy)<budget:
        base=best(greedy)
        rest=set(indices)-{k for k,_ in greedy}
        if not rest: break
        k=min(rest,key=lambda k:(sum(x!=y for x,y in zip(vectors[k],vectors[base])),norm(k)))
        greedy.append((k,cost(k)))
    offsets=[];start=0
    for group in model.groups:
        offsets.append(start);start+=len(group)
    selector=torch.tensor([[offset+s for offset,s in zip(offsets,states[k])] for k in indices],device=model.device)
    opt=torch.optim.Adam([z],lr=.12)
    gradient=[];gradnorm=[]
    for step in range(budget):
        logits=z[selector].sum(1)
        chosen=min((k for k,v in zip(indices,logits.detach().cpu().tolist()) if v==float(logits.max().detach())),key=norm)
        opt.zero_grad()
        objective=model.straight_through(z,states[chosen])
        gradient.append((chosen,float(objective.detach())))
        objective.backward()
        gradnorm.append(float(z.grad.norm()))
        opt.step()
    return {"random":random_trace,"greedy":greedy,"gradient":gradient,"gradient_norms":gradnorm,
            "selected":{k:best(t) for k,t in [("random",random_trace),("greedy",greedy),("gradient",gradient)]}}


def case(folder, binary, lib, work, factorial):
    work.mkdir()
    f=read(folder/("frozen_predictions.json" if factorial else "frozen.json"))
    original=Path(f["source"])/"input.aig"
    if digest(original)!=f["input_sha256"]:raise ValueError("Input drift")
    liberty=Path(read(Path(f["source"])/"context.json")["tools"]["library"]["path"])
    cat=Catalogue(**read(folder/"catalogue.json"));scores=read(folder/"scores.json")["scores"]
    vectors=[v["choices"] for v in f["combinations"]]
    for v in vectors:cat.validate(v)
    groups=symbolic_states(cat.decisions,vectors)
    lines,opts=choice_program(parse_aig(original),cat.decisions,vectors,scores,lib,0.,symbolic=True)
    spec=work/"symbolic.txt";spec.write_text("\n".join(lines)+"\n")
    anchors=work/"anchor.txt";bank=work/"bank.txt"
    cmd=f"read_lib {liberty}; read_aiger {original}; map"
    run_tool([binary,"-c",cmd+"; topo; stime -p"],work,"original",900,
             dict(os.environ,CULS_BASE_ANCHORS=str(anchors),CULS_GLOBAL_TIMING_REPORT="1"))
    run_tool([binary,"-c",cmd],work,"prepare",900,dict(os.environ,CULS_CHOICES=str(spec),
             CULS_PREPARE_BANK=str(bank),CULS_BANK_ANCHORS=str(anchors)))
    models={}
    for device in ("cpu","cuda"):
        models[device]=SparseTiming(bank,anchors,groups,device)
    indices=list(range(0 if factorial else 1,len(vectors)))
    ready=optimize(models["cuda"],vectors,indices)
    save_json(work/"proposals.json",ready)
    # The exhaustive check follows proposal freezing.
    results={}
    for device,m in models.items():
        z=torch.linspace(-.15,.15,m.state_count,device=device,dtype=m.dtype).requires_grad_()
        start=time.perf_counter()
        hard=[float(m.forward(z.detach(),m.encode(v))) for v in vectors]
        soft=m.forward(z);g=torch.autograd.grad(soft,z)[0]
        magnitude,idx=g.detach().abs().max(0)
        delta=torch.zeros_like(z);delta[idx]=1.e-4
        approx=float((m.forward(z.detach()+delta)-m.forward(z.detach()-delta))/(2.e-4))
        if not torch.isfinite(g).all() or not math.isclose(float(g[idx]),approx,rel_tol=2.e-4,abs_tol=1.e-5):
            raise ValueError(f"Relaxed finite difference mismatch on {device}: {float(g[idx])} {approx}")
        results[device]=dict(hard=hard,soft=float(soft),gradient=g.tolist(),max_grad=float(magnitude),
                             finite_difference=approx,diagnostic_s=time.perf_counter()-start)
    x,y=results["cpu"],results["cuda"]
    peak=max(abs(a-b) for a,b in zip(x["hard"],y["hard"]))
    gradient_error=max(abs(a-b) for a,b in zip(x["gradient"],y["gradient"]))
    if peak>1.e-6 or gradient_error>1.e-6 or abs(x["soft"]-y["soft"])>1.e-6:
        raise ValueError(f"CPU/CUDA mismatch: {peak}, {gradient_error}")
    return dict(circuit=int(folder.name[5:]), source=str(folder), vectors=vectors,
                allowed=indices, matcher_sha256=digest(binary),library_sha256=digest(liberty),
                input_sha256=digest(original),groups=list(groups.values()),match_count=models["cpu"].match_count,
                bank_bytes=bank.stat().st_size,stages=len(models["cpu"].stages),states=models["cpu"].state_count,
                bank_sha256=digest(bank),anchors_sha256=digest(anchors),hard_error=peak,grad_error=gradient_error,
                cpu=x,cuda=y,methods=ready, cuda_peak_bytes=torch.cuda.max_memory_allocated())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ("abc","panel","factorial","output"):p.add_argument("--"+n,type=Path,required=True)
    a=p.parse_args()
    if not torch.cuda.is_available() or not os.getenv("SLURM_JOB_ID") or not a.output.resolve().is_relative_to("/data/wushangge"):
        p.error("Use a /data GPU Slurm snapshot")
    torch.set_num_threads(1)
    lib=parse_library(ROOT/"projects/culs/src/algorithms/rewrite_library.inc")
    results=[]
    for i in (44,75,179,665):
        torch.cuda.reset_peak_memory_stats()
        f=(a.factorial if i==665 else a.panel)/f"case_{i:04d}"
        work=a.output/f.name
        r=case(f,a.abc,lib,work,i==665)
        save_json(work/"numeric.json",r);results.append(r)
        print(i,r["match_count"],r["hard_error"],r["grad_error"],flush=True)
    save_json(a.output/"frozen.json",{str(r["circuit"]):digest(a.output/f"case_{r['circuit']:04d}"/"proposals.json") for r in results})
    records=[]
    for r in results:
        factorial=r["circuit"]==665;folder=Path(r["source"])
        metrics={v["mask" if factorial else "index"]:v["metrics"] for v in read(folder/"measurements.json")["results"]
                 if v["status"]=="PASS"}
        limit=read(folder/"baseline.json")["area"]*1.05
        domain=r["allowed"];hard=r["cpu"]["hard"];best_hard=min(hard[k] for k in domain)
        best_real=min(metrics[k]["delay_ps"] for k in domain if metrics[k]["area"]<=limit)
        for method,k in r["methods"]["selected"].items():
            m=metrics[k]
            records.append(dict(circuit=r["circuit"],method=method,chosen=k,budget=len(r["methods"][method]),
                 unique=len({idx for idx,_ in r["methods"][method]}),solver_gap=hard[k]-best_hard,
                 real_ps=m["delay_ps"],feasible_oracle_ps=best_real,real_gap_ps=m["delay_ps"]-best_real,
                 area_ok=m["area"]<=limit,cec=m["cec"]))
    write_csv(a.output/"solver.csv",records)


if __name__=="__main__":
    main()
