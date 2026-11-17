"""Library-matched cut timing with conditional Rewrite ports (no learned model).

Four channels per AIG node: logical positive/complement x electrical rise/fall.
Every mapper option chooses ONE cell/pin assignment for both transitions. Cuts
stop at Rewrite decision roots, not at arbitrary MFFC boundaries. Candidate
internal covers are frozen at zero boundary arrival; upstream arrival is then
propagated through their distinct max-plus input arcs.
"""
from collections import Counter, defaultdict
from types import SimpleNamespace
import math
import numpy as np
import torch

from .boolean_candidates import verify_library_truth


def enumerate_cuts(aig, barriers=(), maximum=8):
    barriers = set(barriers)
    cuts = {i:[(i,)] for i in range(aig.pi_count+1)}
    cuts[0]=[()]
    truths = {}
    def truth(node, leaves):
        k=len(leaves); mask=(1 << (1 << k))-1
        value={n:sum(((x>>i)&1)<<x for x in range(1<<k)) for i,n in enumerate(leaves)}
        value.setdefault(0,0)
        stack=[(node,False)]
        while stack:
            n,done=stack.pop()
            if n in value: continue
            a,b=aig.fanin_literals[n]
            if not done:
                stack.extend([(n,True),(a>>1,False),(b>>1,False)])
            else: value[n]=(value[a>>1] ^ (mask if a&1 else 0)) & (value[b>>1] ^ (mask if b&1 else 0))
        return value[node]
    for node,(a,b) in sorted(aig.fanin_literals.items()):
        ca=[(a>>1,)] if a>>1 in barriers else cuts[a>>1]
        cb=[(b>>1,)] if b>>1 in barriers else cuts[b>>1]
        merged={tuple(sorted(set(x)|set(y))) for x in ca for y in cb if len(set(x)|set(y))<=4}
        # Keep the immediate 2-input implementation plus larger cuts. No timing labels.
        direct=tuple(sorted({a>>1,b>>1}-{0}))
        chosen=[direct]+[c for c in sorted(merged,key=lambda c:(-len(c),c)) if c!=direct][:maximum-1]
        truths[node]=[(c,truth(node,c)) for c in chosen]
        cuts[node]=[(node,)]+chosen
    return truths


def node_options(aig, library, barriers=(), maximum_cuts=8):
    cuts=enumerate_cuts(aig,barriers,maximum_cuts)
    refs=Counter(l>>1 for pair in aig.fanin_literals.values() for l in pair)
    refs.update(l>>1 for l in aig.po_literals)
    result={}
    for node, entries in cuts.items():
        channels=[]
        for phase in (0,1):
            options=[]
            for leaves, truth in entries:
                n=len(leaves); target=truth ^ (((1 << (1 << n))-1) if phase else 0)
                for ci,perm,phases in library.matches.get((n,target),()):
                    arcs=library.arcs(ci,refs[node]); edges=[]
                    for pin,leaf_index in enumerate(perm):
                        for inp in (0,1):
                            for out in (0,1):
                                d=arcs[pin,inp,out]
                                if np.isfinite(d):edges.append((4*leaves[leaf_index]+2*((phases>>pin)&1)+inp,out,float(d)))
                    options.append(edges)
            channels.append(options)
        # Universal NAND/AND cover is required, never substitute structural depth.
        if not all(channels): raise ValueError(f'No physical library match at AIG node {node}')
        result[node]=channels
    return result


def local_transfers(aig, library):
    """One consistent zero-arrival cover, with a max-plus matrix to every PI.

    PI complements are boundary channels, not charged an extra inverter here;
    the caller already has the arrivals of both logical phases.
    """
    dimension=4*aig.pi_count
    paths={4*i+c:np.eye(1,dimension,4*(i-1)+c)[0]*0.-np.inf for i in range(1,aig.pi_count+1) for c in range(4)}
    for i in range(1,aig.pi_count+1):
        for c in range(4): paths[4*i+c][4*(i-1)+c]=0.
    for c in range(4): paths[c]=np.full(dimension,-np.inf)
    for node, channels in node_options(aig,library).items():
        for phase,options in enumerate(channels):
            best=None; best_delay=math.inf
            for edges in options:
                matrix=np.full((2,dimension),-np.inf)
                for source,out,d in edges: matrix[out]=np.maximum(matrix[out],paths[source]+d)
                delay=float(matrix.max())
                if delay<best_delay:best,best_delay=matrix,delay
            if best is None: raise ValueError('Constant-only local cone is unsupported')
            for out in (0,1):paths[4*node+2*phase+out]=best[out]
    root=aig.po_literals[0]
    return np.stack([paths[4*(root>>1)+2*((root&1)^phase)+out] for phase in (0,1) for out in (0,1)])


def local_original(aig, decision):
    leaves=list(decision['cut_leaves']);mapping={n:2*(i+1) for i,n in enumerate(leaves)};mapping.setdefault(0,0)
    needed=set();external=set();stack=[decision['root']]
    while stack:
        n=stack.pop()
        if n in mapping or n in needed or n in external:continue
        if 0<n<=aig.pi_count:
            external.add(n);continue
        if n not in aig.fanin_literals:raise ValueError('Unknown original node')
        needed.add(n);stack.extend(l>>1 for l in aig.fanin_literals[n])
    # A verified functional-support cut can omit structurally reachable PIs.
    # Keep these paths in the original cost reference; never substitute a
    # different Boolean implementation and call it the original structure.
    leaves.extend(sorted(external));mapping.update({n:2*(i+1) for i,n in enumerate(leaves)})
    fanins={}
    for n in sorted(needed):
        target=len(leaves)+len(fanins)+1
        fanins[target]=tuple(mapping[l>>1]^(l&1) for l in aig.fanin_literals[n]);mapping[n]=2*target
    return SimpleNamespace(pi_count=len(leaves),fanin_literals=fanins,po_literals=[mapping[decision['root']]],
                           extra_boundary_pis=sorted(external))


def local_peak(aig,library):
    """Scalar zero-arrival reference without a dense all-PI transfer matrix."""
    arrival=np.zeros(4*(aig.pi_count+len(aig.fanin_literals)+1))
    for node,channels in node_options(aig,library).items():
        for phase,options in enumerate(channels):
            best=None;best_delay=math.inf
            for edges in options:
                pair=np.full(2,-np.inf)
                for src,out,d in edges:pair[out]=max(pair[out],arrival[src]+d)
                delay=float(pair.max())
                if delay<best_delay:best,best_delay=pair,delay
            arrival[4*node+2*phase:4*node+2*phase+2]=best
    root=aig.po_literals[0]
    # Same two logical phase reference as local_transfers().max().
    return float(arrival[4*(root>>1):4*(root>>1)+4].max())


def local_candidate(decision, choice, rewrite_library):
    lib=rewrite_library;truth=int(decision['truth'])&65535
    subgraph=decision['candidate_subgraph_indices'][choice-1]
    proof=verify_library_truth(decision,subgraph,lib)
    phase=lib['p_phases'][truth];perm=lib['perms4'][lib['p_perms'][truth]]
    count=len(decision['cut_leaves']);leaves=[2*(i+1) for i in range(count)]+([0] if count==3 else [])
    values=[leaves[perm[k]]^((phase>>k)&1) for k in range(4)]
    data=lib['classes'][decision['truth_class']];root=data['roots'][subgraph]
    used=set();pending=[root]
    while pending:
        v=pending.pop()
        if v<4 or v in used:continue
        used.add(v);a,b,_,_=data['fanins'][v-4];pending.extend((a,b))
    values.extend([None]*len(data['fanins']));fanins={}
    for v in sorted(used):
        a,b,ca,cb=data['fanins'][v-4];l,r=values[a]^ca,values[b]^cb
        # Native elementary folding; output polarity remains a literal.
        if l==0 or r==0 or l==(r^1):lit=0
        elif l==1 or l==r:lit=r
        elif r==1:lit=l
        else:
            node=count+len(fanins)+1;fanins[node]=(l,r);lit=2*node
        values[v]=lit
    return SimpleNamespace(pi_count=count,fanin_literals=fanins,
                           po_literals=[values[root]^int(proof['output_phase_complement'])])


class CutTimingProxy(torch.nn.Module):
    def __init__(self,aig,catalogue,scores,library,rewrite_library):
        super().__init__()
        self.catalogue=catalogue;self.counts=[d['candidate_count']+1 for d in catalogue.decisions]
        self.width=max(self.counts,default=1);self.n_decisions=len(self.counts)
        by_root=defaultdict(list)
        for i,d in enumerate(catalogue.decisions):by_root[d['root']].append(i)
        lookup={(s['decision_index'],s['choice_index']):s for s in scores}
        transfers={};audit=[]
        for i,d in enumerate(catalogue.decisions):
            original=local_original(aig,d)
            original_peak=local_peak(original,library)
            for c in range(1,self.counts[i]):
                ports=local_transfers(local_candidate(d,c,rewrite_library),library)
                peak=float(np.max(ports))
                student0=float(lookup[i,0]['predicted_local_abc_delay']);student=float(lookup[i,c]['predicted_local_abc_delay'])
                if min(student0,student)<=0 or not np.isfinite([student0,student]).all():raise ValueError('Invalid frozen Student score')
                ratio=student/student0
                scale=float(np.clip(math.sqrt(ratio/(peak/max(original_peak,1e-9))),.5,2.)) if peak>0 else 1.
                transfers[i,c]=ports*scale
                audit.append({'decision_index':i,'choice_index':c,'library_peak_ps':peak,
                              'original_library_peak_ps':original_peak,'student_ratio':ratio,'scale':scale,
                              'original_extra_boundary_pis':original.extra_boundary_pis})
        implementations=node_options(aig,library,by_root)
        levels={i:0 for i in range(aig.pi_count+1)};layers=defaultdict(list)
        for node,pair in sorted(aig.fanin_literals.items()):
            dependencies=[l>>1 for l in pair]
            for i in by_root[node]:dependencies.extend(catalogue.decisions[i]['cut_leaves'])
            levels[node]=1+max(levels[n] for n in dependencies)
            layers[levels[node]].append(node)
        self.layers=[];self.audit=audit;root_groups=[];option_sources=[];edge_sources=[];edge_cost=[];edge_dest=[]
        # Each group is (node,phase,Rewrite option). Mapper matches compete within it.
        for level,nodes in sorted(layers.items()):
            groups=[];options=[];edges=[];sources=[];edge_outputs=[];option_groups=[];group_outputs=[];group_choices=[]
            for node in nodes:
                for phase in (0,1):
                    alternatives=[(-1,implementations[node][phase])]
                    for i in by_root[node]:
                        d=catalogue.decisions[i]
                        for c in range(1,self.counts[i]):
                            port=transfers[i,c]; terms=[]
                            for out in (0,1):
                                for j,leaf in enumerate(d['cut_leaves']):
                                    for channel in range(4):
                                        value=port[2*phase+out,4*j+channel]
                                        if np.isfinite(value):terms.append((4*leaf+channel,out,float(value)))
                            if not terms:raise ValueError('No supported candidate timing arcs')
                            alternatives.append((i*self.width+c,[terms]))
                    for choice,variants in alternatives:
                        group=len(groups);groups.append(node);group_choices.append(choice)
                        group_outputs.append(4*node+2*phase)
                        for terms in variants:
                            option=len(options);options.append(option);option_groups.append(group)
                            for src,out,cost in terms:
                                sources.append(src);edges.append(cost);edge_outputs.append(2*option+out)
            self.layers.append(tuple(torch.tensor(v,dtype=torch.double if k==1 else torch.long)
                                     for k,v in enumerate([sources,edges,edge_outputs,option_groups,group_outputs,group_choices,groups])))
        initial=torch.zeros(4*(aig.pi_count+len(aig.fanin_literals)+1),dtype=torch.double)
        inverter=library.arcs(library.inverter,1)[0]
        for i in range(1,aig.pi_count+1):
            for out in (0,1):initial[4*i+2+out]=float(np.max(inverter[:,out]))
        self.register_buffer('initial',initial)
        self.register_buffer('outputs',torch.tensor([4*(l>>1)+2*(l&1)+out for l in aig.po_literals for out in (0,1)]))
        self.root_decisions={n:[i for i in ids] for n,ids in by_root.items()}
        self._device_layers={}

    def one_hot(self,choices):
        self.catalogue.validate(choices)
        return torch.nn.functional.one_hot(torch.tensor(choices,device=self.initial.device),self.width).to(self.initial.dtype)

    def forward(self,x,arc_scale=None):
        if x.ndim==2:x=x[None]
        key=str(x.device)
        if key not in self._device_layers:self._device_layers[key]=[tuple(t.to(x.device) for t in l) for l in self.layers]
        arrivals=self.initial[None].expand(len(x),-1)
        for src,cost,edge_out,option_group,group_out,choice,group_node in self._device_layers[key]:
            values=arrivals[:,src]+cost[None]*(1. if arc_scale is None else arc_scale)
            option_count=len(option_group);groups=len(group_out)
            option_value=x.new_full((len(x),2*option_count),-torch.inf).scatter_reduce(
                1,edge_out[None].expand(len(x),-1),values,reduce='amax').reshape(len(x),option_count,2)
            worst=option_value.amax(-1)
            best=x.new_full((len(x),groups),torch.inf).scatter_reduce(1,option_group[None].expand(len(x),-1),worst,reduce='amin')
            indices=torch.arange(option_count,device=x.device)[None].expand(len(x),-1)
            chosen=torch.full((len(x),groups),option_count,device=x.device,dtype=torch.long).scatter_reduce(
                1,option_group[None].expand(len(x),-1),torch.where(worst==best[:,option_group],indices,option_count),reduce='amin')
            paired=option_value.gather(1,chosen[:,:,None].expand(-1,-1,2))
            flat=x.flatten(1)
            weights=flat[:,choice.clamp_min(0)] if flat.shape[1] else x.new_ones((len(x),groups))
            original=x.new_ones((len(x),groups))
            # Original weight is one minus every nonzero choice at this root.
            for n,ds in self.root_decisions.items():
                original[:,group_node==n]=1.-x[:,ds,1:].sum((1,2))[:,None]
            weights=torch.where(choice[None]<0,original,weights)
            destinations=torch.stack((group_out,group_out+1),1).flatten()
            update=x.new_zeros(arrivals.shape).scatter_add(1,destinations[None].expand(len(x),-1),(paired*weights[:,:,None]).flatten(1))
            mask=torch.zeros(arrivals.shape[1],device=x.device,dtype=torch.bool);mask[destinations]=True
            arrivals=torch.where(mask[None],update,arrivals)
        return arrivals[:,self.outputs].reshape(len(x),-1,2)
