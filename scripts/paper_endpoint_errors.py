"""Locate direction errors at identifiable primary outputs for already measured edits."""
import csv
import sys
from pathlib import Path
import torch
from paper_joint_evidence import FrozenCase, ROOT, read, save_json
from global_timing.boolean_candidates import parse_library
from global_timing.feasibility import CandidateFeasibility
from global_timing.liberty_cover import Liberty
from global_timing.load_bank import LoadBank
from global_timing.structure import parse_aig


def arrivals(record):
    data=read(Path(record['work'])/'evaluation/result.json')
    assert data['cec']=='PASS' and data['input_sha256']==record['input_sha256']
    return torch.tensor([[p['rise_ps'],p['fall_ps']] for p in data['po_rise_fall']],dtype=torch.float64)


def run(out, file):
    case=FrozenCase(file.parent.name,out/file.parent.name)
    records=read(file)
    direct,noop=map(arrivals,records[:2])
    rules=parse_library(ROOT/'projects/culs/src/algorithms/rewrite_library.inc')
    aig=parse_aig(case.source)
    rule=CandidateFeasibility(aig,case.cat,rules)
    bank=LoadBank(aig,case.cat,rule,read(case.prep/'scores.json')['scores'],rules,
                  Liberty(case.old/'original/tables','cpu'),case.tools['abc'],case.tools['liberty'],case.out/'bank',.5)
    def evaluate(v):
        with torch.no_grad():
            return bank.forward(bank.hard(v),detail=True)[0].double()
    base=evaluate(case.zero)
    rows=[]
    with (case.out/'transition_deltas.csv').open('x',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['proposal','po','transition','abc_noop','abc_after','proxy_noop','proxy_after'])
        for record in records[2:]:
            edits=record['edits']
            if not edits or record['status']!='PASS':
                continue
            v=case.zero.copy()
            for i,c in edits:v[i]=c
            pred=evaluate(v)
            real=arrivals(record)
            assert real.shape==pred.shape==base.shape
            # Take the transition attaining the true candidate maximum; preserve other observations in CSV.
            k=int(real.flatten().argmax())
            predicted_change=float((pred-base).flatten()[k])
            observed_change=float((real-noop).flatten()[k])
            rows.append(dict(label=record['label'],real_ps=float(real.max()),raw_proxy_ps=float(pred.max()),
                    anchored_direct_proxy_ps=float((direct+pred-base).max()),
                    anchored_noop_proxy_ps=float((noop+pred-base).max()),
                    true_critical_po=k//2,transition=['rise','fall'][k%2],
                    predicted_delta_at_true_transition=predicted_change,real_delta_at_true_transition=observed_change))
            for po in range(real.shape[0]):
                for t,edge in enumerate(('rise','fall')):
                    writer.writerow([record['label'],po,edge,float(noop[po,t]),float(real[po,t]),float(base[po,t]),float(pred[po,t])])
    save_json(case.out/'endpoint_summary.json',dict(rows=rows,original_max=float(direct.max()),noop_max=float(noop.max()),
                    proxy_noop_max=float(base.max()),proxy_noop_critical_po=int(base.flatten().argmax())//2,
                    true_noop_critical_po=int(noop.flatten().argmax())//2,device='cpu'))


if __name__=='__main__':
    torch.set_num_threads(1)
    out=Path(sys.argv[1])
    out.mkdir()
    for arg in sys.argv[2:]:run(out,Path(arg))
