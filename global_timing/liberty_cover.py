"""Pin-level Liberty interpolation and sparse conditional timing on CPU/CUDA."""
from collections import defaultdict
from graphlib import TopologicalSorter
from pathlib import Path
import re
import torch


class Liberty:
    def __init__(self, path, device):
        self.device = torch.device(device)
        self.cells, caps, senses, surfaces = {}, [], [], []
        lines = iter(Path(path).read_text().splitlines())
        for line in lines:
            f = line.split()
            if f[0] == 'C':
                cell = dict(area=float(f[2]), output=f[3], pins={})
                self.cells[f[1]] = cell
            elif f[0] == 'P':
                cell['pins'][f[1]] = len(caps)
                caps.append([float(f[2]),float(f[3])])
                senses.append(int(f[4]))
                for _ in range(4):
                    tag,n,m,*values = next(lines).split()
                    n,m = int(n),int(m)
                    if tag!='T' or len(values)!=n+m+n*m or min(n,m)<1:
                        raise ValueError('Incomplete Liberty surface')
                    v = list(map(float,values))
                    surfaces.append((v[:n],v[n:n+m],v[n+m:]))
            elif f == ['END']:
                break
            else:
                raise ValueError('Unsupported Liberty export: '+line)
        else:
            raise ValueError('Unfinished Liberty export')
        nx=max(len(x) for x,_,_ in surfaces)
        ny=max(len(y) for _,y,_ in surfaces)
        gx,gy,tables,counts=[],[],[],[]
        for x,y,v in surfaces:
            counts.append([len(x),len(y)])
            gx.append(x+[float('inf')]*(nx-len(x)))
            gy.append(y+[float('inf')]*(ny-len(y)))
            rows=[v[i:i+len(y)]+[0.]*(ny-len(y)) for i in range(0,len(v),len(y))]
            tables.append(rows+[[0.]*ny]*(nx-len(rows)))
        self.x,self.y,self.tables,self.caps = [torch.tensor(v,device=device,dtype=torch.float32)
                                               for v in (gx,gy,tables,caps)]
        self.counts=torch.tensor(counts,device=device)
        self.senses=senses

    def index(self, values):
        return torch.tensor(values,device=self.device,dtype=torch.long)

    def lookup(self, ids, slew, load):
        def bracket(grid,n,v):
            i=torch.minimum((grid[ids]<=v[:,None]).sum(1)-1,n-2).clamp_min(0)
            j=torch.minimum(i+1,n-1)
            a,b=grid[ids,i],grid[ids,j]
            return i,j,(v-a)/torch.where(i==j,1.,b-a)
        n,m=self.counts[ids].unbind(1)
        i,j,s=bracket(self.x,n,slew)
        k,l,t=bracket(self.y,m,load)
        a,b=self.tables[ids,i,k],self.tables[ids,j,k]
        x=a+t*(self.tables[ids,i,l]-a)
        y=b+t*(self.tables[ids,j,l]-b)
        return x+s*(y-x)


def mapped_graph(path, lib, prefix=''):
    """Read ABC's named-pin combinational Verilog with positional IO."""
    gates, ins, outs = {}, [], []
    lines=iter(Path(path).read_text().splitlines())
    for line in lines:
        line=line.strip()
        if line.startswith(('input ', 'output ')):
            kind,buf=line.split(None,1)
            while ';' not in buf:
                buf+=' '+next(lines).strip()
            (ins if kind=='input' else outs).extend(prefix+n.strip() for n in buf.split(';')[0].split(','))
        elif line.startswith('assign '):
            out,src=re.fullmatch(r'assign\s+(\S+)\s*=\s*(\S+)\s*;',line).groups()
            gates[prefix+out]=(None,[prefix+src])
        elif re.match(r'\w+\s+g\d+\(',line):
            name=line.split()[0]
            cell=lib.cells[name]
            ports=dict(re.findall(r'\.(\w+)\(([^()]*)\)',line))
            gates[prefix+ports[cell['output']]]=(name,[prefix+ports[p] for p in cell['pins']])
    if not outs or any(n in ins for n in gates):
        raise ValueError('Invalid mapped interface')
    return ins,outs,gates


class TimingGraph:
    """Reverse active fanout and pin load, then forward arrival and slew.

    groups maps a mux output to (first logit index, input nets, pin scales).
    Distinct consumers of a shared gate each charge their pin capacitance.
    """
    def __init__(self, lib, graph, inputs, outputs, groups=None, chunk=32768):
        self.lib,self.chunk=lib,chunk
        groups=groups or {}
        net={**graph,**{n:('MUX',v[1]) for n,v in groups.items()}}
        # Only compile the transitive fanin of observable outputs.
        needed=set()
        stack=list(outputs)
        while stack:
            n=stack.pop()
            if n not in needed:
                needed.add(n)
                if n in net: stack.extend(net[n][1])
        deps={n:net[n][1] for n in net if n in needed}
        order=list(TopologicalSorter(deps).static_order())
        order.extend(n for n in dict.fromkeys(outputs) if n not in deps and n not in order)
        pos={n:i for i,n in enumerate(order)}
        self.positions=pos
        self.names=order
        self.po=lib.index([pos[n] for n in outputs])
        known=set(inputs)
        levels=defaultdict(list)
        depth={}
        for n in order:
            if n not in net:
                if n not in known and not n.endswith(("1'b0","1'b1")):
                    raise ValueError('Undriven mapped net '+n)
                depth[n]=0
                continue
            name,parents=net[n]
            depth[n]=1+max((depth[p] for p in parents),default=0)
            levels[depth[n]].append(n)
        self.stages=[]
        self.size=len(pos)
        area=[0.]*len(pos)
        self.state_count=max((start+len(p) for start,p,_ in groups.values()),default=0)
        for _,nodes in sorted(levels.items()):
            for off in range(0,len(nodes),chunk):
                subset=nodes[off:off+chunk]
                arcs=[]; wires=[]; trans=[]; scale=[]
                for j,n in enumerate(subset):
                    name,parents=net[n]
                    if name=='MUX' or name is None:
                        for k,p in enumerate(parents):
                            state=groups[n][0]+k if name=='MUX' else -1
                            wires.append((pos[p],j,state))
                            scale.append(groups[n][2][k] if name=='MUX' else 1.)
                    else:
                        cell=lib.cells[name]
                        area[pos[n]]=cell['area']
                        for p,pin in zip(parents,cell['pins'].values()):
                            k=len(arcs)
                            arcs.append((pos[p],j,pin))
                            for out in range(2):
                                for inp in range(2):
                                    if lib.senses[pin]==3 or ((out==inp)==(lib.senses[pin]==1)):
                                        trans.append((k,inp,out,4*pin+out))
                def columns(rows,width):
                    return tuple(lib.index(v) for v in zip(*rows)) if rows else (None,)*width
                self.stages.append((lib.index([pos[n] for n in subset]),columns(arcs,3),
                                    columns(wires,3),columns(trans,4),
                                    torch.tensor(scale,device=lib.device)))
        self.area=torch.tensor(area,device=lib.device)

    def forward(self, weights, detail=False):
        # Indexed writes use fresh tensors when autograd needs the previous value.
        lib=self.lib
        w=torch.cat((weights,weights.new_ones(1)))
        def zeros(*shape): return torch.zeros(shape,device=lib.device,dtype=torch.float32)
        active=zeros(self.size).index_fill(0,self.po,1.)
        loads=torch.zeros((self.size,2),device=lib.device,dtype=torch.float64)
        recovery=detail=="recovery"
        refs=zeros(self.size) if recovery else None
        for ids,(src,dst,pin),(parent,child,state),_,_ in reversed(self.stages):
            if src is not None:
                a=active[ids[dst]]
                loads=loads.index_add(0,src,(lib.caps[pin]*a[:,None]).double())
                if recovery: refs=refs.index_add(0,src,a)
                active=active.scatter_reduce(0,src,a,reduce='amax')
            if parent is not None:
                a=active[ids[child]]*w[state]
                loads=loads.index_add(0,parent,loads[ids[child]]*w[state,None])
                if recovery: refs=refs.index_add(0,parent,a)
                active=active.scatter_reduce(0,parent,a,reduce='amax')
        loads=loads.float()
        arc_delays=[]
        flow=zeros(self.size) if recovery else None
        arr=zeros(self.size,2); slew=zeros(self.size,2)
        # Local driver delay used by LiteCut scaling at candidate boundaries.
        delta=zeros(self.size,2)
        for ids,(src,dst,pin),(parent,child,state),(arc,ip,op,tab),scale in self.stages:
            a=zeros(len(ids)*2);s=zeros(len(ids)*2);d=zeros(len(ids)*2)
            if recovery:
                flow_local=self.area[ids]
                if src is not None:
                    flow_local=flow_local.index_add(0,dst,flow[src]/refs[src].clamp_min(1.))
                if parent is not None:
                    flow_local=flow_local.index_add(0,child,w[state]*flow[parent]/refs[parent].clamp_min(1.))
                flow=flow.index_copy(0,ids,flow_local)
            cost=None
            if arc is not None:
                u=src[arc]; v=dst[arc]; target=2*v+op
                sl=slew[u,ip]; cap=loads[ids[v],op]
                cost=lib.lookup(tab,sl,cap)
                value=arr[u,ip]+cost
                a=a.scatter_reduce(0,target,value,reduce='amax')
                s=s.scatter_reduce(0,target,lib.lookup(tab+2,sl,cap),reduce='amax')
                d=d.scatter_reduce(0,target,torch.where(value==a[target],cost,0.),reduce='amax')
            if recovery: arc_delays.append(cost)
            a,s,d=[v.view(-1,2) for v in (a,s,d)]
            if parent is not None:
                mass=w[state,None]
                a=a.index_add(0,child,mass*(arr[parent]+(scale[:,None]-1.)*delta[parent]))
                s=s.index_add(0,child,mass*slew[parent])
                d=d.index_add(0,child,mass*delta[parent])
            arr=arr.index_copy(0,ids,a);slew=slew.index_copy(0,ids,s);delta=delta.index_copy(0,ids,d)
        pred=arr[self.po]
        if recovery:
            req=arr.new_full(arr.shape,float("inf"))
            req=req.index_copy(0,self.po,pred.amax().expand(len(self.po),2))
            for (ids,(src,dst,_),(parent,child,state),(arc,ip,op,_),_),cost in zip(reversed(self.stages),reversed(arc_delays)):
                if src is not None:
                    rval=req[ids[dst[arc]],op]-cost
                    req=req.flatten().scatter_reduce(0,2*src[arc]+ip,rval,reduce="amin").view(-1,2)
                if parent is not None:
                    rval=req[ids[child]].masked_fill(w[state,None]<=0,float("inf"))
                    req=req.scatter_reduce(0,parent[:,None].expand(-1,2),rval,reduce="amin")
            return arr,req,flow
        return (pred,active@self.area,loads,slew) if detail else (pred.amax(),active@self.area)
