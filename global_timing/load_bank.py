"""Compile root-local alternatives into one technology-mapped template forest."""
from collections import defaultdict
from pathlib import Path
import torch
from .abc_choices import choice_program
from .abc_hard_choices import StructuralIds
from .common import run_tool, tool_path
from .liberty_cover import TimingGraph, mapped_graph


def write_forest(path, count, ids, outputs):
    def varint(stream, value):
        while value >= 128:
            stream.write(bytes([(value & 127) | 128]))
            value >>= 7
        stream.write(bytes([value]))
    with Path(path).open('xb') as stream:
        stream.write(f'aig {count+len(ids.gates)} {count} 0 {len(outputs)} {len(ids.gates)}\n'.encode())
        for x in outputs:
            stream.write(f'{x}\n'.encode())
        for (a,b),q in ids.gates.items():
            varint(stream,q-b)
            varint(stream,b-a)


def prepare_forest(aig, catalogue, allowed, scores, lib, path):
    # Each configurable root becomes an interface pin. Its original logic and
    # every option are independent forest outputs, sharing structurally equal gates.
    by_root=defaultdict(list)
    for i,d in enumerate(catalogue.decisions):
        by_root[d['root']].extend((i,c) for c in range(1,len(allowed[i])) if allowed[i][c])
    roots=sorted(r for r,g in by_root.items() if g)
    groups=[[(-1,0)]+by_root[r] for r in roots]
    virtual={r:2*(aig.pi_count+1+j) for j,r in enumerate(roots)}
    count=aig.pi_count+len(roots)
    ids=StructuralIds(count)
    raw={i:2*i for i in range(aig.pi_count+1)}
    def signal(lit):
        n=lit>>1
        return (virtual[n] if n in virtual else raw[n])^(lit&1)
    for n,(left,right) in sorted(aig.fanin_literals.items()):
        raw[n]=ids.gate(signal(left),signal(right))
    outputs=[signal(lit) for lit in aig.po_literals]
    local={(s['decision_index'],s['choice_index']):s['predicted_local_abc_delay'] for s in scores}
    scales=[]
    for root,group in zip(roots,groups):
        outputs.append(raw[root])
        ratios=[1.]
        for i,c in group[1:]:
            d=catalogue.decisions[i]
            lines,_=choice_program(aig,[d],[[c]],[],lib,0.)
            nodes={}
            def literal(lit):
                return (nodes[lit>>1]^(lit&1)) if lit>>1 in nodes else signal(lit)
            for line in lines:
                f=line.split()
                if f[0]=='g':
                    q=ids.gate(literal(int(f[1])),literal(int(f[2])))
                    nodes[aig.maximum_variable+1+len(nodes)]=q
                else:
                    outputs.append(literal(int(f[4])))
            before,after=local.get((i,0),0.),local.get((i,c),0.)
            ratios.append(after/before if before>0 and after>0 else 1.)
        scales.append(ratios)
    write_forest(path,count,ids,outputs)
    return groups,scales


class LoadBank:
    def __init__(self, aig, catalogue, rule, scores, rewrite_lib, timing_lib, abc, liberty, work, alpha=0.):
        work.mkdir()
        source=work/'forest.aig'
        self.groups,ratios=prepare_forest(aig,catalogue,rule.allowed(),scores,rewrite_lib,source)
        mapped=work/'templates.v'
        # ABC expands supergates and runs its delay pass, reference counting and
        # required-time area recovery once on the entire bank of local templates.
        run_tool([abc,'-c',f'read_lib {tool_path(liberty)}; read_aiger {tool_path(source)}; map; write_verilog {tool_path(mapped)}'],
                 work,'compile_templates',900)
        pins,outs,graph=mapped_graph(mapped,timing_lib)
        if len(pins)!=aig.pi_count+len(self.groups) or len(outs)!=aig.po_count+sum(map(len,self.groups)):
            raise ValueError('Template port identity changed')
        selectors={}
        start=0
        for j,(g,r) in enumerate(zip(self.groups,ratios)):
            lo=aig.po_count+start
            selectors[pins[aig.pi_count+j]]=(start,outs[lo:lo+len(g)],r)
            start+=len(g)
        self.graph=TimingGraph(timing_lib,graph,pins[:aig.pi_count],outs[:aig.po_count],selectors)
        self.ratios=[stage[-1] for stage in self.graph.stages]
        self.set_alpha(alpha)
        self.states=start
        self.device=timing_lib.device

    def set_alpha(self,alpha):
        for j,r in enumerate(self.ratios):
            st=self.graph.stages[j]
            self.graph.stages[j]=(*st[:-1],r.pow(alpha))

    def hard(self, vector):
        result=torch.zeros(self.states,device=self.device)
        start=0
        chosen=[]
        edits=0
        for g in self.groups:
            active=[k for k,(i,c) in enumerate(g) if k and vector[i]==c]
            if len(active)>1: raise ValueError('Two edits at one root')
            edits+=bool(active)
            chosen.append(start+(active[0] if active else 0))
            start+=len(g)
        if chosen: result[chosen]=1.
        if sum(bool(c) for c in vector)!=edits:
            raise ValueError('Unrepresented edit')
        return result

    def soft(self,z):
        return torch.cat([v.softmax(0) for v in z.split([len(g) for g in self.groups])]) if self.groups else z
