"""Static arithmetic from saved topology/features; no model/optimizer/tool invocation."""
import sys,json,math,collections
from pathlib import Path
RUN=Path('/data/wushangge/CULS_LiteCut/runs/20261116T070435Z-ste-adam-random100-d532863a')
sys.path.insert(0,str(RUN/'src'))
from global_timing.structure import parse_aig

def inspect(i, run=RUN):
 w=run/'output'/f'case_{i:04d}_Ours/work';p=json.loads((w/'partition/partition.json').read_text());v=json.loads((w/'cut_validation/verified_catalogue.json').read_text());s=json.loads((w/'summary.json').read_text());sc=json.loads(next((w/'litecut_cache').glob('*.json')).read_text())['scores'];by={(x['decision_index'],x['choice_index']):x for x in sc};aig=parse_aig(w.parent/'input.aig')
 owner={n:r['region_id'] for r in p['regions'] for n in r['internal_nodes']};trans=[{int(k):d for k,d in r.items()} for r in p['port_structural_depths']];order=p['topological_order'];D=s['original']['delay_ps'];T=.005*D
 depths=[0.]*len(trans)
 for r in order:depths[r]=max((depths[owner[port]] if port in owner else 0)+d for port,d in trans[r].items())
 maxdepth=max(depths[owner[lit>>1]] if lit>>1 in owner else 0 for lit in aig.po_literals);scale=D/max(maxdepth,1)
 arr=[0.]*len(trans);edges={}
 for r in order:
  vals={port:((arr[owner[port]] if port in owner else 0)+d*scale) for port,d in trans[r].items()};mx=max(vals.values());arr[r]=mx+T*math.log(sum(math.exp((x-mx)/T) for x in vals.values()))
  for port,value in vals.items():edges[r,port]=math.exp((value-arr[r])/T)
 po=s['original']['po_arrivals_ps'];weights=[math.exp((x/D-1)/.02) for x in po];z=sum(weights);adj=[0.]*len(trans)
 for j,lit in enumerate(aig.po_literals):
  if lit>>1 in owner:adj[owner[lit>>1]]+=weights[j]/z/D
 edge_sens={}
 for r in reversed(order):
  for port,d in trans[r].items():
   g=adj[r]*edges[r,port];edge_sens[r,port]=g*d*scale
   if port in owner:adj[owner[port]]+=g
 # Original structural ports reaching decision roots; same incidence definition.
 ports_at={}
 for node in sorted(aig.fanin_literals):
  rid=owner[node];ports=set()
  for lit in aig.fanin_literals[node]:
   n=lit>>1;ports.update(ports_at[n] if owner.get(n)==rid else {n})
  ports_at[node]=ports
 out=[]
 for (d,c),x in by.items():
  if not c:continue
  b=by[d,0];raw=x['continuous_features_19'];raw0=b['continuous_features_19'];feat=[max(-5,min(5,(x['predicted_local_abc_delay']-b['predicted_local_abc_delay'])/max(abs(b['predicted_local_abc_delay']),1))),max(-5,min(5,(raw[8]-raw0[8])/max(abs(raw0[8]),1))),max(-5,min(5,(raw[11]-raw0[11])/max(abs(raw0[11]),1)))]
  q=.08*feat[0]+.15*feat[1];dec=v['decisions'][d];rid=dec['decision_owner_region'];ports=ports_at[dec['root']];sensitivity=sum(edge_sens[rid,port] for port in ports)
  out.append(dict(circuit_id=i,decision_index=d,choice_index=c,root=dec['root'],region_id=rid,q=q,feature_local_delta=feat[0],feature_depth_delta=feat[1],feature_size_delta=feat[2],active_ports=len(ports),structural_sensitivity=sensitivity,noop_choice_derivative=q*sensitivity,raw_local_original=b['predicted_local_abc_delay'],raw_local_candidate=x['predicted_local_abc_delay'],feature_hash_identical=x['feature_hash']==b['feature_hash']))
 return out
