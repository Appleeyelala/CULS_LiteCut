"""First-order load correction on paths of the original ABC mapped netlist."""
from bisect import bisect_right
from collections import Counter, defaultdict
from graphlib import TopologicalSorter
import random
import re
from statistics import median

import torch


def library_cells(path):
    """Read single-output ABC combinational cell functions, pin caps and delay tables."""
    text = path.read_text()
    if 'time_unit : "1ps"' not in text or 'capacitive_load_unit (1,ff)' not in text:
        raise ValueError('Expected ps and fF Liberty units')
    cells = {}
    parts = re.split(r'(?m)^  cell \((\w+)\) \{', text)
    for name, body in zip(parts[1::2], parts[2::2]):
        funcs = re.findall(r'(?m)^      function : "([^"]+)";', body)
        if len(funcs) != 1:
            continue
        # The frozen ASAP7 combinational library uses sum-of-products functions.
        expr = funcs[0].translate(str.maketrans({'!':'~', '*':'&', '+':'|'}))
        if not re.fullmatch(r'[A-Za-z0-9_()~&| ]+', expr):
            continue
        terms = re.split(r'(?m)^    pin \((\w+)\) \{', body)
        caps = {}
        for pin, block in zip(terms[1::2], terms[2::2]):
            cap = re.search(r'(?m)^      capacitance : ([\d.eE+-]+);', block)
            if cap:
                caps[pin] = float(cap[1])
        tables = []
        for table in re.findall(r'cell_(?:rise|fall) \([^)]*\) \{([^}]+)\}', body):
            grid = [float(x) for x in re.search(r'index_2 \("([^"]+)"', table)[1].split(',')]
            rows = re.findall(r'"([\d., eE+-]+)"',table.split('values (')[1])
            tables.append((grid,[float(x) for x in rows[len(rows)//2].split(',')]))
        cells[name] = (compile(expr,'<ASAP7 function>','eval'), caps, tables, funcs[0])
    return cells


def table_delay(tables, cap):
    """Median across timing arcs at middle slew; use outer slope beyond table."""
    values, slopes = [], []
    for grid, row in tables:
        i = max(0,min(len(grid)-2,bisect_right(grid,cap)-1))
        slope = (row[i+1]-row[i])/(grid[i+1]-grid[i])
        values.append(row[i]+slope*(cap-grid[i]))
        slopes.append(slope)
    return (median(values), max(0.,median(slopes))) if tables else (0.,0.)


def new_leaf_uses(decision, subgraph, lib, root_users):
    truth = decision['truth'] & 65535
    perm = lib['perms4'][lib['p_perms'][truth]]
    leaves = (decision['cut_leaves']+[0])[:4]
    cls = lib['classes'][decision['truth_class']]
    root = cls['roots'][subgraph]
    if root < 4:
        return Counter({leaves[perm[root]]:root_users})
    result, seen, pending = Counter(), set(), [root]
    while pending:
        n = pending.pop()
        if n in seen:
            continue
        seen.add(n)
        for v in cls['fanins'][n-4][:2]:
            if v < 4:
                result[leaves[perm[v]]] += 1
            else:
                pending.append(v)
    return result


def load_impact(model, mapped, cells, lib):
    aig = model.partition.aig
    rng = random.Random(20261117)
    mask = (1<<512)-1
    aig_sig = {0:0, **{i:rng.getrandbits(512) for i in range(1,aig.pi_count+1)}}

    def literal(lit):
        return aig_sig[lit>>1] ^ (mask if lit&1 else 0)

    for node, (a,b) in sorted(aig.fanin_literals.items()):
        aig_sig[node] = literal(a) & literal(b)
    drivers, pins, uses, caps = {}, {}, Counter(), Counter()
    all_nets = set()
    with mapped.open() as stream:
        for line in stream:
            if '_ASAP7_' in line and re.match(r'\s*\w+\s+g\d+\s*\(',line):
                kind = line.split()[0]
                conn = re.findall(r'\.(\w+)\(([^()]*)\)',line)
                if not conn or not line.strip().endswith('));'):
                    raise ValueError('Unsupported gate syntax')
                inputs = {k:n for k,n in conn if k in cells[kind][1]}
                outputs = [n for k,n in conn if k not in inputs]
                if len(outputs)!=1:
                    raise ValueError(('Unsupported cell output',kind,conn))
                out, = outputs
                for pin, net in inputs.items():
                    uses[net]+=1
                    caps[net]+=cells[kind][1][pin]
            elif line.lstrip().startswith('assign '):
                out,net = re.fullmatch(r"\s*assign\s+(\w+)\s*=\s*(\w+|1'b[01])\s*;\s*",line).groups()
                inputs={'A':net}
                kind=None
            else:
                continue
            if out in drivers:
                raise ValueError('Duplicate mapped driver')
            drivers[out],pins[out]=kind,inputs
            all_nets.update([out,*inputs.values()])
    # Topological propagation is local to this Verilog namespace.
    order=list(TopologicalSorter({n:set(ip.values()) for n,ip in pins.items()}).static_order())
    sig,arrival,parent,per_sink={}, {}, {}, {}
    probed=defaultdict(list)
    outputs={}
    for n in order:
        if n not in drivers:
            if n.startswith('pi') and n[2:].isdigit():
                sig[n]=aig_sig[int(n[2:])+1]
            elif n in ("1'b0","1'b1"):
                sig[n]=mask if n.endswith('1') else 0
            else:
                raise ValueError(('Unknown mapped source',n))
            arrival[n]=0.
            continue
        kind, ip = drivers[n],pins[n]
        pred=max(ip.values(),key=lambda v:arrival[v],default=None)
        parent[n]=pred
        if kind is None:
            sig[n],arrival[n]=sig[pred],arrival[pred]
        else:
            expr,_,tables,func=cells[kind]
            sig[n] = mask if func.strip()=='1' else eval(expr,{'__builtins__':{}},{k:sig[v] for k,v in ip.items()}) & mask
            delay,slope = table_delay(tables,caps[n])
            arrival[n]=(arrival[pred] if pred else 0.)+delay
            if uses[n] and tables:
                per_sink[n]=slope*caps[n]/uses[n]
            probed[sig[n]].append(n)
        if n.startswith('po') and n[2:].isdigit():
            outputs[int(n[2:])]=n
    if len(outputs)!=len(aig.po_literals) or any(sig[outputs[i]]!=literal(lit) for i,lit in enumerate(aig.po_literals)):
        raise ValueError('Mapped port signature/order mismatch')
    # Reject repeated AIG or mapped signatures and near-constant signatures.
    sig_counts=Counter(aig_sig.values())
    paths=defaultdict(list)
    for i,out in outputs.items():
        n=out
        while n is not None:
            if n in per_sink:
                paths[n].append(i)
            n=parent.get(n)
    counts=Counter(lit>>1 for pair in aig.fanin_literals.values() for lit in pair)
    counts.update(lit>>1 for lit in aig.po_literals)
    basis=model.features.new_zeros((model.n_decisions,model.width,len(outputs)))
    aligned=set()
    changed=set()
    for d,decision in enumerate(model.catalogue.decisions):
        leaves=set(decision['cut_leaves'])
        old=Counter(v>>1 for n in decision['removable_nodes'] for v in aig.fanin_literals[n] if v>>1 in leaves)
        for j,graph in enumerate(decision['candidate_subgraph_indices'],1):
            new=new_leaf_uses(decision,graph,lib,counts[decision['root']])
            for leaf in leaves:
                diff=new[leaf]-old[leaf]
                if not diff:
                    continue
                changed.add(leaf)
                bits=aig_sig[leaf]
                if sig_counts[bits]!=1 or not 16<=bits.bit_count()<=496 or len(probed[bits])!=1:
                    continue
                net=probed[bits][0]
                if net not in per_sink:
                    continue
                aligned.add(leaf)
                if net in paths:
                    basis[d,j,paths[net]]+=diff*per_sink[net]
    return basis,dict(changed_leaves=len(changed),aligned_leaves=len(aligned),
                      nonzero_choices=int(basis.abs().amax(-1).count_nonzero()),original_pos=len(outputs))


def corrected_arrivals(original, selections, basis):
    return (original + torch.einsum('...dc,dcp->...p',selections,basis)).clamp_min(0.)
