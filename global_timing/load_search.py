"""Joint hard proposals with a differentiable conditional pin-load objective."""
import math
import random
import time
import torch


def optimize(bank, rule, method, budget, seconds, edit_limit, area_ratio, seed):
    start=time.monotonic()
    zero=[0]*len(rule.catalogue.decisions)
    z=torch.zeros(bank.states,device=bank.device,requires_grad=True)
    optimizer=torch.optim.Adam([z],lr=.15) if method=='gradient' and bank.states else None
    def evaluate(vector):
        rule.validate(vector)
        with torch.no_grad():
            delay,area=bank.forward(bank.hard(vector))
        values=float(delay),float(area)
        if not all(map(math.isfinite,values)): raise ValueError('Nonfinite hard score')
        return values
    d0,a0=evaluate(zero)
    base=max(d0,1.e-6)
    limit=a0*area_ratio if area_ratio is not None else math.inf
    best=zero
    rank=(d0,0,tuple(zero))
    rng=random.Random(seed)
    trace=[]
    visited=set()
    options=[pair for g in bank.groups for pair in g[1:]]
    rng.shuffle(options)
    for step in range(budget):
        if step and seconds is not None and time.monotonic()-start>=seconds: break
        candidate=zero.copy()
        if step and method=='gradient':
            weights=z.detach().tolist()
            edits=[];offset=0
            for g in bank.groups:
                k=max(range(len(g)),key=lambda j:weights[offset+j])
                if k:
                    i,c=g[k]
                    edits.append((weights[offset+k]-weights[offset],i,c))
                offset+=len(g)
            candidate=rule.project(zero,edits,edit_limit)
        elif step and method=='random':
            picks=[rng.choice(g[1:]) for g in bank.groups]
            candidate=rule.project(zero,[(rng.random(),i,c) for i,c in picks],
                                   rng.randint(1,edit_limit) if edit_limit else 0)
        elif step and method=='greedy':
            found=False
            for i,c in options:
                trial=best.copy();trial[i]=c if best[i]!=c else 0
                if tuple(trial) in visited or sum(bool(x) for x in trial)>edit_limit: continue
                try: rule.validate(trial)
                except ValueError: continue
                candidate=trial;found=True;break
            if not found: break
        if not step: delay,area=d0,a0
        else: delay,area=evaluate(candidate)
        key=(delay,sum(bool(x) for x in candidate),tuple(candidate))
        tolerance=max(2.e-5,2.e-6*abs(rank[0]))
        improves=delay < rank[0]-tolerance
        tied=abs(delay-rank[0])<=tolerance
        if area<=limit and (improves or (tied and key[1:]<rank[1:])):
            rank=key;best=candidate.copy()
        visited.add(tuple(candidate))
        row=dict(step=step,vector=candidate,delay_ps=delay,area=area)
        if optimizer is not None and step+1<budget:
            optimizer.zero_grad()
            weights=bank.soft(z)
            # Hard forward values determine selection; gradients follow the soft
            # pin loads, active references and arrival/slew on this same network.
            sd,sa=bank.forward(weights)
            loss=sd/base
            if area_ratio is not None:
                loss=loss+20.*torch.relu(sa/max(a0,1.e-6)-area_ratio)
            loss.backward()
            if z.grad is None or not torch.isfinite(z.grad).all():
                raise ValueError('Nonfinite relaxed gradient')
            row['gradient_norm']=float(z.grad.norm())
            optimizer.step()
        row['elapsed_s']=time.monotonic()-start
        trace.append(row)
        if not options: break
    return best,dict(trace=trace,unique_vectors=len(visited),hard_calls=len(trace),method=method,
                     rank_delay=rank[0],original_model_delay=d0,original_model_area=a0,
                     seconds=time.monotonic()-start)
