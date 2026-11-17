"""Experimental cut-port composition, with frozen LiteCut and baseline anchor.

Transfers are weighted AIG path abstractions, NOT mapped cell arcs. Original
logic is condensed at decision roots, cut leaves and POs; a candidate replaces
only its root transfer. Upstream chosen arrivals therefore enter downstream
candidate ports. This isolates composition from the historical region scaling.
Structural hashing, mapped load/slew and cell covering remain unmodelled.
"""
from collections import Counter
import math

import torch

from .boolean_candidates import parse_library, verify_library_truth


def merge_paths(parts):
    result={}
    for paths,cost in parts:
        for port,length in paths.items():result[port]=max(result.get(port,-math.inf),length+cost)
    return result


def condense_original(aig, retained, inversion_cost=.2):
    """Exact max-plus elimination of untouched AIG nodes; PIs start at zero.

    Parallel original paths from the same retained port keep their maximum.
    The result is exact for this structural timing model, not ABC timing.
    """
    # The parser property allocates a full set on each access. Materialize once.
    and_nodes=aig.and_nodes
    retained=set(retained)&and_nodes
    uses=Counter(lit>>1 for pair in aig.fanin_literals.values() for lit in pair)
    distances={};transfers={}
    for node in sorted(and_nodes):
        pair=aig.fanin_literals[node]
        paths=merge_paths([(distances[lit>>1] if lit>>1 in and_nodes else {0:0.},
                            1.+inversion_cost*(lit&1)) for lit in pair])
        if node in retained:transfers[node]=paths;distances[node]={node:0.}
        else:distances[node]=paths
        for lit in pair:
            parent=lit>>1;uses[parent]-=1
            if uses[parent]==0:distances.pop(parent,None)
    return transfers


def original_cut_paths(aig,root,leaves,inversion_cost=.2):
    and_nodes=aig.and_nodes
    boundary=set(leaves)|{0};pending=[root];interior=set()
    while pending:
        node=pending.pop()
        if node in boundary or node in interior:continue
        if node not in and_nodes:
            # Verified support cuts can have context-dependent external PIs.
            boundary.add(node);continue
        interior.add(node);pending.extend(lit>>1 for lit in aig.fanin_literals[node])
    paths={node:{node:0.} for node in boundary}
    for node in sorted(interior):
        paths[node]=merge_paths([(paths[lit>>1],1.+inversion_cost*(lit&1)) for lit in aig.fanin_literals[node]])
    return paths[root]


def candidate_paths(decision,subgraph,library,inversion_cost=.2):
    proof=verify_library_truth(decision,subgraph,library)
    truth=int(decision['truth'])&0xffff;phase=library['p_phases'][truth]
    permutation=library['perms4'][library['p_perms'][truth]]
    leaves=list(decision['cut_leaves'])
    if len(leaves)==3:leaves.append(0)
    data=library['classes'][decision['truth_class']]
    # Complement of a library input belongs on its first outgoing edge.
    paths=[{leaves[permutation[k]]:0.} for k in range(4)]
    for a,b,ca,cb in data['fanins']:
        paths.append(merge_paths([(paths[a],1.+inversion_cost*(ca ^ (((phase>>a)&1) if a<4 else 0))),
                                  (paths[b],1.+inversion_cost*(cb ^ (((phase>>b)&1) if b<4 else 0)))]))
    root=data['roots'][subgraph]
    extra=inversion_cost*(bool(proof['output_phase_complement']) ^ (bool((phase>>root)&1) if root<4 else False))
    return {p:d+extra for p,d in paths[root].items()}


class CompositionalTimingSurrogate(torch.nn.Module):
    def __init__(self,reference,scores,library,litecut_blend=.5,smoothing_error=.001,inversion_cost=.2):
        super().__init__()
        if not 0<=litecut_blend<=1 or smoothing_error<=0:raise ValueError('Invalid proxy parameters.')
        library=parse_library(library) if not isinstance(library,dict) else library
        self.catalogue=reference.catalogue;self.partition=reference.partition
        self.n_decisions=reference.n_decisions;self.width=reference.width;self.counts=reference.counts
        self.delay_reference=reference.delay_reference;self.area_reference=reference.area_reference
        self.area_baseline=reference.area_baseline
        for name in ['features','valid','sizes','baseline_po','area_weights']:
            self.register_buffer(name,getattr(reference,name).detach().clone())
        aig=self.partition.aig;retained={lit>>1 for lit in aig.po_literals};self.by_root={}
        for i,d in enumerate(self.catalogue.decisions):
            retained.add(d['root']);retained.update(d['cut_leaves']);self.by_root.setdefault(d['root'],[]).append(i)
        transfers=condense_original(aig,retained,inversion_cost)
        self.nodes=sorted(transfers);self.node_position={node:i+1 for i,node in enumerate(self.nodes)}
        self.original={};self.options={};groups={(s['decision_index'],s['choice_index']):s for s in scores}
        def packed(paths):
            ports=sorted(paths)
            return ([self.node_position.get(p,0) for p in ports],
                    self.features.new_tensor([paths[p] for p in ports]))
        arrival={0:0.}
        for node in self.nodes:
            arrival[node]=max(arrival.get(p,0.)+d for p,d in transfers[node].items())
            self.original[node]=packed(transfers[node])
        structural=max((arrival.get(lit>>1,0.)+inversion_cost*(lit&1) for lit in aig.po_literals),default=1.)
        self.unit_ps=self.delay_reference/max(structural,1.)
        max_arity=max((len(t) for t in transfers.values()),default=2)
        self.temperature=self.delay_reference*smoothing_error/max(len(self.nodes)*math.log(max(max_arity,4)),1.)
        for i,decision in enumerate(self.catalogue.decisions):
            original=original_cut_paths(aig,decision['root'],decision['cut_leaves'],inversion_cost)
            original_depth=max(original.values(),default=0.)
            local0=max(float(groups[i,0]['predicted_local_abc_delay']),1e-6)
            for c,subgraph in enumerate(decision['candidate_subgraph_indices'],1):
                paths=candidate_paths(decision,subgraph,library,inversion_cost)
                depth=max(paths.values(),default=0.)
                ratio=max(float(groups[i,c]['predicted_local_abc_delay']),1e-6)/local0
                # Blend a frozen local-delay ratio with structural depth while
                # retaining per-port asymmetry. Zero-depth wire stays a wire.
                scale=(max(original_depth,1e-6)*ratio/max(depth,1e-6))**litecut_blend if depth>0 else 1.
                self.options[i,c]=packed({p:d*scale for p,d in paths.items()})
        self.output_positions=torch.tensor([self.node_position.get(lit>>1,0) for lit in aig.po_literals],device=self.features.device)
        self.output_costs=self.features.new_tensor([inversion_cost*(lit&1)*self.unit_ps for lit in aig.po_literals])
        with torch.no_grad():self.register_buffer('baseline_dp',self.propagate(self.one_hot([0]*self.n_decisions).unsqueeze(0))[0])
        self.metadata={'schema':'compositional_cut_ports_v1','litecut_blend':litecut_blend,'inversion_cost':inversion_cost,
                       'smoothing_error':smoothing_error,'retained_nodes':len(self.nodes),'unit_ps':self.unit_ps,
                       'area_model':'unchanged historical gate-count predictor',
                       'limitations':['unmapped structural path arcs','no constant folding or dynamic hash/load/cell covering','PI arrivals zero before PO anchoring',
                                      'soft incompatible mixtures are only a gradient extension; only hard feasible vectors are executable']}

    def one_hot(self,choices):
        self.catalogue.validate(choices)
        return torch.nn.functional.one_hot(torch.tensor(choices,device=self.features.device),self.width).to(self.features.dtype)

    def propagate(self,x):
        values=[x.new_zeros(x.shape[0])]
        def transfer(entry):
            indices,costs=entry
            ports=torch.stack([values[i] for i in indices],-1)
            return self.temperature*torch.logsumexp((ports+costs*self.unit_ps)/self.temperature,-1)
        for node in self.nodes:
            original=transfer(self.original[node]);value=original
            for i in self.by_root.get(node,[]):
                for c in range(1,self.counts[i]):
                    value=value+x[:,i,c]*(transfer(self.options[i,c])-original)
            values.append(value)
        return torch.stack(values,-1)[:,self.output_positions]+self.output_costs

    def forward(self,probabilities):
        single=probabilities.ndim==2;x=probabilities.unsqueeze(0) if single else probabilities
        if x.shape[1:]!=self.features.shape[:2]:raise ValueError('Selection tensor shape mismatch.')
        po=torch.clamp(self.baseline_po+self.propagate(x)-self.baseline_dp,min=0.)
        expected=(x.unsqueeze(-1)*self.features.unsqueeze(0)).sum(2)
        summary=(expected*self.sizes[None,:,None]).sum(1)/max(self.partition.aig.and_count,1)
        area=self.area_baseline*torch.exp(2.*torch.tanh((summary*self.area_weights).sum(-1)/2.))
        return (po[0],area[0]) if single else (po,area)
