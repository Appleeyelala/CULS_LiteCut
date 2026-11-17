"""Research v3: conditional literal ports, frozen LiteCut, PO-only calibration.

No internal mapped timing, candidate measurements, or physical slew/capacitance
enters the forward pass. Shared parameters are independent of circuit size.
"""
from collections import Counter, defaultdict
import math
from types import SimpleNamespace

import torch
from torch import nn

from .boolean_candidates import parse_library
from .conditional_arrival import conditional_arrivals, conditional_reference
from .phase_ports import original_transfers, candidate_transfers, hard_phase_arrivals


def make_port_reference(partition, catalogue, scores, baseline):
    """Build the existing selection/area contract without another timing graph."""
    counts=[d['candidate_count']+1 for d in catalogue.decisions]
    width=max(counts,default=1);n=len(counts)
    features=torch.zeros((n,width,4),dtype=torch.double)
    valid=torch.zeros((n,width),dtype=torch.bool);sizes=[]
    lookup={(r['decision_index'],r['choice_index']):r for r in scores}
    for i,count in enumerate(counts):
        valid[i,:count]=True;zero=lookup[i,0];raw0=zero['continuous_features_19'];sizes.append(max(raw0[11],1.))
        for c in range(1,count):
            r=lookup[i,c];raw=r['continuous_features_19']
            features[i,c]=torch.tensor([(r['predicted_local_abc_delay']-zero['predicted_local_abc_delay'])/max(abs(zero['predicted_local_abc_delay']),1.),
                 (raw[8]-raw0[8])/max(abs(raw0[8]),1.),(raw[11]-raw0[11])/max(abs(raw0[11]),1.),1.],dtype=torch.double)
    if baseline['timing_model']!='ABC_SCL_LIBERTY':raise ValueError('Wrong timing labels')
    po=torch.tensor(baseline['po_arrivals_ps'],dtype=torch.double)
    if len(po)!=partition.aig.po_count or not torch.isfinite(po).all():raise ValueError('Baseline PO mismatch')
    return SimpleNamespace(partition=partition,catalogue=catalogue,n_decisions=n,width=width,counts=counts,
        delay_reference=max(float(baseline['delay_ps']),1e-6),area_reference=max(float(baseline['area']),1e-6),
        area_baseline=float(baseline['area']),features=features.clamp(-5,5),valid=valid,
        sizes=torch.tensor(sizes,dtype=torch.double),baseline_po=po,area_weights=torch.tensor([0.,0.,1.,0.],dtype=torch.double))


class PortTimingParameters(nn.Module):
    def __init__(self):
        super().__init__()
        self.port = nn.Sequential(nn.Linear(32, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 1))
        self.global_head = nn.Sequential(nn.Linear(20, 32), nn.SiLU(), nn.Linear(32, 32), nn.SiLU(), nn.Linear(32, 1))
        # Identity structural prior and zero residual before any label fitting.
        for head in [self.port, self.global_head]:
            nn.init.zeros_(head[-1].weight); nn.init.zeros_(head[-1].bias)


class LearnedPortSurrogate(nn.Module):
    def __init__(self, reference, scores, library, parameters=None, variant='combined',
                 smoothing_error=.001, backend='bounded_tape', inversion_cost=.2):
        super().__init__()
        if variant not in {'fixed', 'ports', 'global', 'combined'}:
            raise ValueError('Unknown v3 ablation')
        if backend not in {'bounded_tape', 'reference'} or smoothing_error <= 0:
            raise ValueError('Invalid propagation configuration')
        self.variant, self.backend = variant, backend
        self.shared = parameters if parameters is not None else PortTimingParameters().double()
        self.catalogue, self.partition = reference.catalogue, reference.partition
        self.n_decisions, self.width, self.counts = reference.n_decisions, reference.width, reference.counts
        self.delay_reference, self.area_reference = reference.delay_reference, reference.area_reference
        self.area_baseline = reference.area_baseline
        for name in ['features', 'valid', 'sizes', 'baseline_po', 'area_weights']:
            self.register_buffer(name, getattr(reference, name).detach().clone())
        lib = parse_library(library) if not isinstance(library, dict) else library
        aig = self.partition.aig
        retained = {lit >> 1 for lit in aig.po_literals}
        by_root = defaultdict(list)
        for i, d in enumerate(self.catalogue.decisions):
            retained.add(d['root']); retained.update(d['cut_leaves']); by_root[d['root']].append(i)
        originals = original_transfers(aig, retained, inversion_cost)
        nodes = sorted(originals)
        positions = {(n,p): 2+2*i+p for i,n in enumerate(nodes) for p in (0,1)}
        source_position = lambda key: positions.get(key, key[1])  # PI/constant arrivals are fixed.
        structural = hard_phase_arrivals(aig, self.catalogue.decisions, [0]*self.n_decisions, lib, inversion_cost)
        self.unit_ps = self.delay_reference / max(max(structural, default=1.), 1.)
        groups = {(r['decision_index'],r['choice_index']):r for r in scores}
        options, edge_costs, edge_features, edge_regions, option_choice, active_roots = [], [], [], [], [], []
        flat_indices, root_indices = [], []
        node_levels = {0:0, 1:0}
        layers = defaultdict(lambda: [[], [], [], [], [], []])
        maximum_arity = 2
        owner = self.partition.owner
        for n in nodes:
            alternatives = [(None, 0, originals[n])]
            for i in by_root[n]:
                for c, subgraph in enumerate(self.catalogue.decisions[i]['candidate_subgraph_indices'], 1):
                    alternatives.append((i, c, candidate_transfers(self.catalogue.decisions[i],subgraph,lib,inversion_cost)))
                    flat_indices.append(i*self.width+c); root_indices.append(len(active_roots))
            active_roots.append(n)
            level = 1 + max(node_levels[source_position(k)] for _,_,paths in alternatives for ch in paths for k in ch)
            layer = layers[level]
            for phase in (0,1):
                dest = positions[n,phase]; dest_local = len(layer[0]); layer[0].append(dest)
                node_levels[dest] = level
                for i,c,paths in alternatives:
                    option = len(options); local_option = len(layer[1]); layer[1].append(option)
                    layer[5].append(dest_local)
                    options.append(dest)
                    option_choice.append((i*self.width+c) if i is not None else -1-len(active_roots)+1)
                    score = groups.get((i,c)) if i is not None else groups.get((by_root[n][0],0)) if by_root[n] else None
                    raw = (score or {}).get('continuous_features_19', [0.]*19)
                    if len(raw) != 19: raise ValueError('Expected frozen 19-feature LiteCut record')
                    raw = [math.copysign(math.log1p(abs(float(v))),v)/10. for v in raw]
                    ratio = (float(score['predicted_local_abc_delay'])/max(abs(float(groups[i,0]['predicted_local_abc_delay'])),1.)) if i is not None else 1.
                    maximum_arity = max(maximum_arity,len(paths[phase]))
                    for key,cost in sorted(paths[phase].items()):
                        edge = len(edge_costs); layer[2].append(edge); layer[3].append(source_position(key)); layer[4].append(local_option)
                        edge_costs.append(cost*self.unit_ps)
                        edge_features.append(raw + [math.log(max(ratio,1e-8)),cost/max(max(structural,default=1.),1.),key[1],phase,float(c>0)])
                        edge_regions.append(owner[n])
        self._cpu_layers = tuple(tuple(torch.tensor(v,dtype=torch.long) for v in layers[k]) for k in sorted(layers))
        self._layer_cache = {}
        self.temperature = self.delay_reference*smoothing_error/max(len(layers)*math.log(maximum_arity),1.)
        initial = self.features.new_zeros(2+2*len(nodes)); initial[1] = inversion_cost*self.unit_ps
        for name,value in [('initial',initial),('edge_prior',self.features.new_tensor(edge_costs)),
                           ('edge_features',self.features.new_tensor(edge_features).reshape(-1,24)),
                           ('edge_regions',torch.tensor(edge_regions,dtype=torch.long)),
                           ('option_choice',torch.tensor(option_choice,dtype=torch.long)),
                           ('flat_indices',torch.tensor(flat_indices,dtype=torch.long)),
                           ('root_indices',torch.tensor(root_indices,dtype=torch.long)),
                           ('outputs',torch.tensor([source_position((l>>1,l&1)) for l in aig.po_literals],dtype=torch.long)),
                           ('decision_regions',torch.tensor([d['decision_owner_region'] for d in self.catalogue.decisions],dtype=torch.long))]:
            self.register_buffer(name,value)
        self.root_count = len(active_roots)
        regions = len(self.partition.regions)
        pairs = sorted(set(self.partition.edges)); degree = Counter(a for a,b in pairs)
        indices = torch.tensor(pairs,dtype=torch.long).T if pairs else torch.empty((2,0),dtype=torch.long)
        context = torch.sparse_coo_tensor(indices,self.features.new_tensor([1./degree[a] for a,b in pairs]),(regions,regions)).coalesce()
        self.register_buffer('context_matrix',context)
        sizes_by_region=Counter(owner.values());boundary=defaultdict(set);successors=defaultdict(list)
        for node,pair in aig.fanin_literals.items():
            for literal in pair:
                source=literal>>1
                if owner.get(source)!=owner[node]:boundary[owner[node]].add(source)
        for a,b in pairs:successors[a].append(b)
        distance={}
        for region in reversed(getattr(self.partition,'order',list(range(regions)))):
            distance[region]=max((distance[s]+1 for s in successors[region]),default=0)
        static=[[math.log1p(sizes_by_region[r])/10.,math.log1p(len(boundary[r]))/10.,
                 math.log1p(degree[r])/10.,distance[r]/max(max(distance.values(),default=1),1)] for r in range(regions)]
        self.register_buffer('region_static',self.features.new_tensor(static).reshape(regions,4))
        po_regions = [owner.get(l>>1,regions) for l in aig.po_literals]
        self.register_buffer('po_regions',torch.tensor(po_regions,dtype=torch.long))
        po_static = [[float(self.baseline_po[j])/self.delay_reference,structural[j]/max(max(structural,default=1.),1.),
                      l&1,math.log1p(aig.and_count)/20.] for j,l in enumerate(aig.po_literals)]
        self.register_buffer('po_static',self.features.new_tensor(po_static).reshape(-1,4))
        self.metadata = {'schema':'learned_literal_ports_v3','variant':variant,'backend':backend,
                         'nodes':len(nodes),'edges':len(edge_costs),'levels':len(layers),'temperature_ps':self.temperature,
                         'unit_ps':self.unit_ps,'smoothing_error':smoothing_error,
                         'logical_channels':'positive/complement, not rise/fall','internal_mapped_alignment':False,
                         'context':'two fixed reverse region-DAG mean passes; no learned message passing',
                         'limitation':'structural AIG paths with learned effective costs; no dynamic mapper cover or physical loads'}

    def one_hot(self, choices):
        self.catalogue.validate(choices)
        return torch.nn.functional.one_hot(torch.tensor(choices,device=self.features.device),self.width).to(self.features.dtype)

    def _layers(self, device):
        key = str(device)
        if key not in self._layer_cache:
            self._layer_cache[key] = tuple(tuple(t.to(device) for t in layer) for layer in self._cpu_layers)
        return self._layer_cache[key]

    def _context(self, x):
        expected = (x.unsqueeze(-1)*self.features[None]).sum(2)
        r = self.context_matrix.shape[0]
        local = x.new_zeros((x.shape[0],r,4)).index_add(1,self.decision_regions,expected)
        # Sum of edit features, bounded smoothly; no candidate labels.
        h = torch.tanh(local+self.region_static[None]).permute(1,0,2).reshape(r,-1)
        h1 = torch.sparse.mm(self.context_matrix,h)
        h2 = torch.sparse.mm(self.context_matrix,h1)
        context = torch.cat([h1.reshape(r,x.shape[0],4),h2.reshape(r,x.shape[0],4)],-1).permute(1,0,2)
        return expected, context

    def propagate(self, x):
        expected, context = self._context(x)
        flat = torch.cat([x.flatten(1),x.new_zeros((len(x),1))],1)
        roots = x.new_zeros((len(x),self.root_count)).index_add(1,self.root_indices,flat[:,self.flat_indices])
        choice = self.option_choice
        weights = torch.where(choice[None]>=0,flat[:,choice.clamp_min(0)],1.-roots[:,(-1-choice).clamp_min(0)])
        costs = self.edge_prior[None].expand(len(x),-1)
        if self.variant in {'ports','combined'}:
            feature = torch.cat([self.edge_features[None].expand(len(x),-1,-1),context[:,self.edge_regions]],-1)
            network_dtype = next(self.shared.parameters()).dtype
            scale = (torch.nn.functional.softplus(self.shared.port(feature.to(network_dtype)).squeeze(-1))/math.log(2.)).to(x.dtype)
            costs = costs*scale
        operation = conditional_arrivals if self.backend=='bounded_tape' else conditional_reference
        arrival = operation(costs,weights,self.initial,self._layers(x.device),self.temperature)[:,self.outputs]
        if self.variant in {'global','combined'}:
            mean = expected.sum(1)/max(self.n_decisions,1)
            weighted = (expected*self.sizes[None,:,None]).sum(1)/max(self.partition.aig.and_count,1)
            global_features = torch.cat([mean,weighted],-1)[:,None].expand(-1,len(self.outputs),-1)
            padded_context = torch.cat([context,x.new_zeros((len(x),1,8))],1)
            feature = torch.cat([self.po_static[None].expand(len(x),-1,-1),global_features,padded_context[:,self.po_regions]],-1)
            network_dtype = next(self.shared.parameters()).dtype
            residual = self.shared.global_head(feature.to(network_dtype)).squeeze(-1).to(x.dtype)
            arrival = arrival + self.delay_reference*residual
        return arrival

    def forward(self, probabilities):
        single = probabilities.ndim==2
        x = probabilities[None] if single else probabilities
        if x.shape[1:] != self.features.shape[:2]: raise ValueError('Selection shape mismatch')
        noop = self.one_hot([0]*self.n_decisions)[None]
        raw = self.propagate(torch.cat([noop,x],0))
        po = (self.baseline_po[None]+(raw[1:]-raw[0])).clamp_min(0.)
        expected = (x.unsqueeze(-1)*self.features[None]).sum(2)
        summary = (expected*self.sizes[None,:,None]).sum(1)/max(self.partition.aig.and_count,1)
        area = self.area_baseline*torch.exp(2.*torch.tanh((summary*self.area_weights).sum(-1)/2.))
        return (po[0],area[0]) if single else (po,area)
