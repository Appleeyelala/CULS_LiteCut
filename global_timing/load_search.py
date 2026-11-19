"""Joint hard proposals with a differentiable conditional pin-load objective."""
import math
import random
import time
import torch


def size_for_step(limit, tried, incumbent):
    """Explore both extremes and the log midpoint, then refine around the incumbent."""
    if not limit:
        return 0
    midpoint = max(1, round(math.sqrt(limit)))
    for size in (limit, 1, midpoint):
        if size not in tried:
            return size
    ladder = {2**k for k in range(limit.bit_length())} | {limit}
    target = max(1, min(incumbent or midpoint, limit))
    remaining = ladder - tried
    if not remaining:
        return target
    return min(remaining, key=lambda n: (abs(math.log2(n / target)), n))


def optimize(bank, rule, method, budget, seconds, edit_limit, area_ratio, seed, record_priority=False,
             adaptive_sizes=False, observed_po=None):
    start=time.monotonic()
    zero=[0]*len(rule.catalogue.decisions)
    z=torch.zeros(bank.states,device=bank.device,requires_grad=True)
    optimizer=torch.optim.Adam([z],lr=.15) if method=='gradient' and bank.states else None
    offset = None
    def prediction(weights):
        if offset is None:
            return bank.forward(weights)
        po, area, _, _ = bank.forward(weights, detail=True)
        return (po + offset).amax(), area

    def evaluate(vector):
        rule.validate(vector)
        with torch.no_grad():
            delay,area=prediction(bank.hard(vector))
        values=float(delay),float(area)
        if not all(map(math.isfinite,values)): raise ValueError('Nonfinite hard score')
        return values
    if observed_po is None:
        d0,a0=evaluate(zero)
    else:
        rule.validate(zero)
        with torch.no_grad():
            po,area,_,_=bank.forward(bank.hard(zero),detail=True)
            observed = po.new_tensor(observed_po)
            if observed.shape != po.shape or not torch.isfinite(observed).all():
                raise ValueError("Invalid ABC baseline PO rise/fall observations")
            offset = observed - po
            d0, a0 = float(observed.amax()), float(area)
    tried_sizes = set()
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
        priority=[]
        capacity=edit_limit
        if step and method=='gradient':
            weights=z.detach().tolist()
            edits=[];offset=0
            for g in bank.groups:
                k=max(range(len(g)),key=lambda j:weights[offset+j])
                if k:
                    i,c=g[k]
                    edits.append((weights[offset+k]-weights[offset],i,c))
                offset+=len(g)
            if adaptive_sizes:
                capacity = size_for_step(edit_limit, tried_sizes, rank[1])
                tried_sizes.add(capacity)
            candidate=rule.project(zero,edits,capacity)
            if record_priority:
                priority=[(i,c) for _,i,c in sorted(edits,key=lambda e:(-e[0],e[1],e[2]))
                          if candidate[i]==c]
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
        if adaptive_sizes:
            row["edit_capacity"] = capacity if step else 0
        if record_priority:
            row["priority"]=priority
        if optimizer is not None and step+1<budget:
            optimizer.zero_grad()
            weights=bank.soft(z)
            # Hard forward values determine selection; gradients follow the soft
            # pin loads, active references and arrival/slew on this same network.
            sd,sa=prediction(weights)
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
