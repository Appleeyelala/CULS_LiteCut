"""Forward block greedy using refreshed global directional derivatives."""
import math
import time

import torch


def marginal_greedy(bank,rule,budget,edit_limit):
    start=time.monotonic()
    incumbent=[0]*len(rule.catalogue.decisions)
    base_weights=bank.hard(incumbent)
    uniform=bank.soft(torch.zeros_like(base_weights))
    with torch.no_grad():
        best=float(bank.forward(base_weights)[0])
    if not math.isfinite(best):
        raise ValueError('Nonfinite baseline score')
    trace=[dict(vector=incumbent.copy(),delay_ps=best)]
    attempted=set()
    updates=0
    for step in range(1,budget):
        if updates>=edit_limit:
            break
        # A fixed midpoint keeps candidate branches observable when the incumbent is one-hot.
        probe=((bank.hard(incumbent)+uniform)*.5).detach().requires_grad_()
        delay,_=bank.forward(probe)
        grad,=torch.autograd.grad(delay,probe)
        if not torch.isfinite(grad).all():
            raise ValueError('Nonfinite global marginal')
        values=grad.detach().cpu().tolist()
        offset=0
        options=[]
        for group in bank.groups:
            if not any(incumbent[i]==c for i,c in group[1:]):
                allowed=[(values[offset]-values[offset+j],i,c) for j,(i,c) in enumerate(group[1:],1)
                         if (i,c) not in attempted]
                # At most one option per root. Projection still enforces reuse and conflicts across roots.
                if allowed:
                    score,i,c=max(allowed,key=lambda x:(x[0],-x[1],-x[2]))
                    if score>0:
                        options.append((score,i,c))
            offset+=len(group)
        capacity=math.ceil((edit_limit-updates)/(budget-step))
        trial=rule.project(incumbent,options,capacity)
        changes=[(i,c) for i,c in enumerate(trial) if c and c!=incumbent[i]]
        if not changes:
            break
        attempted.update(changes)
        with torch.no_grad():
            score=float(bank.forward(bank.hard(trial))[0])
        if not math.isfinite(score):
            raise ValueError('Nonfinite hard score')
        trace.append(dict(vector=trial,delay_ps=score))
        if score < best-max(2e-5,abs(best)*2e-6):
            incumbent=trial
            best=score
            updates+=len(changes)
    return incumbent,trace,time.monotonic()-start
