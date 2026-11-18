"""Replay fixed Liberty timing without altering the archived mapped circuits."""
from pathlib import Path
import argparse
import json
import os
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from global_timing.common import run_tool, save_json, tool_path, digest
from global_timing.liberty_cover import Liberty, TimingGraph, mapped_graph

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('abc','library','reference','original','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    torch.set_num_threads(1)
    out=a.output
    path=out/'tables'
    run_tool([a.abc,'-c',f'read_lib {tool_path(a.library)}; read_aiger {tool_path(a.original)}; map; topo; stime'],
             out,'export',900,dict(os.environ,CULS_LIB_TABLES=str(path)))
    results=[]
    for dev in ('cpu','cuda'):
        model=Liberty(path,dev)
        for file in sorted(a.reference.glob('case_*/trace.json')):
            frozen=json.loads(file.read_text())['timing']
            ins,outs,gates=mapped_graph(file.with_name('mapped.v'),model)
            graph=TimingGraph(model,gates,ins,outs)
            pred,area,_,_=graph.forward(torch.empty(0,device=dev),True)
            ref=torch.tensor([[r['rise_ps'],r['fall_ps']] for r in frozen['po_rise_fall']],device=dev)
            results.append(dict(case=file.parent.name,device=dev,max_error=float((pred-ref).abs().max()),
                               area_error=float(area)-frozen['area'],outputs=len(outs),
                               mapped_sha256=digest(file.with_name('mapped.v'))))
            torch.testing.assert_close(pred,ref,atol=.015,rtol=3.e-5)
            if abs(float(area)-frozen['area'])>max(.02,3.e-5*frozen['area']):
                raise ValueError((float(area),frozen['area']))
    save_json(out/'parity.json',results)
