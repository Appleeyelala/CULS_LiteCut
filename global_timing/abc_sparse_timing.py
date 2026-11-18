"""Bounded sparse pin-timing prototype with hard legal and continuous forwards.

Each match uses the same supergate for output rise and fall. The hard selector
breaks delay ties by export order; ABC also consults area flow and fanout limits.
The soft selector relaxes match and rewrite selection; trainable inputs are logits.
"""
from collections import defaultdict
from pathlib import Path
import torch


class SparseTiming:
    def __init__(self, bank, anchors, groups, device="cpu", dtype=torch.float64):
        self.groups = list(groups.values())
        self.device, self.dtype = torch.device(device), dtype
        def tensor(data, kind):
            return torch.tensor(data,device=self.device,dtype=kind)
        anchor = {}
        with Path(anchors).open() as f:
            for line in f:
                i,*ts=line.split()
                anchor[int(i)] = list(map(float,ts))
        stable, levels, matches, cells, outputs = {}, {}, defaultdict(list), {}, []
        constraints, read_matches, expected = {}, 0, None
        with Path(bank).open() as f:
            for line in f:
                code,*v = line.split()
                if code=="I":
                    self.inv=tensor(list(map(float,v)),dtype)
                elif code=="A":
                    n,original=map(int,v);stable[n]=anchor[original];levels[n]=0
                elif code=="N":
                    n=int(v[0]);matches[n];levels[n]=1
                elif code=="S":
                    cells[int(v[0])]=list(map(float,v[1:]))
                elif code=="M":
                    phase,mask,cell,polar,scale,*leaf_text=v
                    leaves=list(map(int,leaf_text))
                    levels[n]=max(levels[n],1+max(levels[j] for j in leaves))
                    mask=int(mask,16)
                    cond=constraints.setdefault(mask,len(constraints))
                    matches[n].append((int(phase),cond,int(cell),int(polar),float(scale),leaves))
                    read_matches+=1
                elif code=="P":
                    npo,phase=map(int,v);outputs.append((npo,phase))
                elif code=="E":
                    expected=int(v[0])
                else:
                    raise ValueError("Unknown match bank record")
        if expected!=read_matches or not outputs:
            raise ValueError("Incomplete ABC match bank")
        self.state_count=sum(map(len,self.groups))
        self.allowed=tensor([[bool(mask & (1<<j)) for j in range(self.state_count)]
                             for mask in constraints],torch.bool).reshape(len(constraints),self.state_count)
        count=max(levels)+2
        self.boundary=torch.zeros((count,2,2),device=self.device,dtype=dtype)
        if stable:
            self.boundary[tensor(list(stable),torch.long)]=tensor(list(stable.values()),dtype).reshape(-1,2,2)
        self.outputs=tensor([2*(count-1 if n<0 else n)+p for n,p in outputs],torch.long)
        self.stages=[]
        by_level=defaultdict(list)
        for n in matches:
            by_level[levels[n]].append(n)
        for _,nodes in sorted(by_level.items()):
            src,dst,delay,target,cond=[],[],[],[],[]
            for slot,n in enumerate(nodes):
                for phase,condition,cell,polar,scale,leaves in matches[n]:
                    idx=len(target); target.append(2*slot+phase);cond.append(condition)
                    for j,leaf in enumerate(leaves):
                        leaf_phase=1-(polar>>j&1)
                        for z in range(4):
                            d=cells[cell][4*j+z]
                            if d>0:
                                src.append(4*leaf+2*leaf_phase+(z&1))
                                dst.append(2*idx+z//2);delay.append(d*scale)
            if not target:
                raise ValueError("No local supergate alternatives on a prepared level")
            self.stages.append(tuple(tensor(v,k) for v,k in ((nodes,torch.long),(src,torch.long),
                (dst,torch.long),(delay,dtype),(target,torch.long),(cond,torch.long))))
        self.match_count=expected

    def encode(self, vector):
        states=[]
        for group in self.groups:
            selected=[k for k,(i,c) in enumerate(group) if k and vector[i]==c]
            if len(selected)>1:
                raise ValueError("Rewrite vector conflicts at a shared root")
            states.append(selected[0] if selected else 0)
        if sum(bool(s) for s in states) != sum(bool(c) for c in vector):
            raise ValueError("Rewrite vector contains a choice outside the prepared bank")
        return states

    def probabilities(self, logits, states=None):
        parts=[];start=0
        for j,group in enumerate(self.groups):
            if states is None:
                p=logits[start:start+len(group)].softmax(0)
            else:
                p=torch.zeros(len(group),device=self.device,dtype=self.dtype)
                p[states[j]]=1.
            parts.append(p);start+=len(group)
        if logits.numel()!=start:
            raise ValueError("Wrong logit dimension")
        return parts

    def forward(self, logits, states=None, temperature=2.0):
        if temperature<=0:
            raise ValueError("Positive relaxation temperature required")
        prob=self.probabilities(logits,states)
        weights=torch.ones(self.allowed.shape[0],device=self.device,dtype=self.dtype)
        start=0
        for p in prob:
            weights=weights*(self.allowed[:,start:start+len(p)].to(self.dtype) @ p)
            start+=len(p)
        values=self.boundary
        inf=1.e12
        for nodes,src,dst,delay,target,cond in self.stages:
            nphase=2*len(nodes)
            arcs=values.flatten()[src]+delay
            matches=arcs.new_zeros(2*len(target)).scatter_reduce(0,dst,arcs,reduce="amax").view(-1,2)
            worst=matches.amax(1)
            if states is not None:
                score=worst.masked_fill(weights[cond]==0,inf)
                minimum=score.new_full((nphase,),inf).scatter_reduce(0,target,score,reduce="amin")
                order=torch.arange(len(score),device=self.device)
                first=order.new_full((nphase,),len(score)).scatter_reduce(
                    0,target,torch.where(score==minimum[target],order,len(score)),reduce="amin")
                chosen=matches[first.clamp_max(len(score)-1)].masked_fill((minimum>=inf/2)[:,None],inf)
            else:
                # Share probability across both output transitions of each physical match.
                score=-worst/temperature + weights[cond].clamp_min(1.e-30).log()
                score=score.masked_fill(weights[cond]==0,-1.e30)
                maximum=score.new_full((nphase,),-1.e30).scatter_reduce(0,target,score,reduce="amax")
                mass=(score-maximum[target].detach()).exp()*(weights[cond]>0)
                denom=score.new_zeros(nphase).index_add(0,target,mass)
                chosen=score.new_zeros(nphase,2).index_add(0,target,mass[:,None]*matches)
                chosen=(chosen/denom.clamp_min(1.e-30)[:,None]).masked_fill((denom==0)[:,None],inf)
            direct=chosen.view(-1,2,2)
            inverted=direct.flip((1,2))+self.inv
            # Native delay-pass phase dropping permits the opposite implementation plus inverter.
            timing=torch.where((direct.amax(2)>inverted.amax(2)+.001)[:,:,None],inverted,direct)
            values=values.index_copy(0,nodes,timing)
        return values.reshape(-1,2)[self.outputs].amax()

    def straight_through(self, logits, states):
        hard=self.forward(logits,states)
        soft=self.forward(logits)
        return hard.detach()+(soft-soft.detach())
