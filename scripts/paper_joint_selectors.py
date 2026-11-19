"""Freeze equal hard-evaluation budgets before evaluating the selected methods and their oracles."""
import sys
import time
from pathlib import Path
import torch
from paper_joint_evidence import FrozenCase, ROOT, read, save_json, digest
from global_timing.boolean_candidates import parse_library
from global_timing.feasibility import CandidateFeasibility
from global_timing.liberty_cover import Liberty
from global_timing.load_bank import LoadBank
from global_timing.load_search import optimize
from global_timing.load_marginals import marginal_greedy
from global_timing.structure import parse_aig

SIZES=(0,1,2,4,6,8,12,16)

def sparse(v):
    return [(i,c) for i,c in enumerate(v) if c]

def informed(bank,rule,case,scores,block):
    before={r['decision_index']:r['predicted_local_abc_delay'] for r in scores if not r['choice_index']}
    values={(r['decision_index'],r['choice_index']):r['predicted_local_abc_delay'] for r in scores}
    options=[]
    for group in bank.groups:
        gain,i,c=max(((1-values[i,c]/max(before[i],1.e-12),i,c) for i,c in group[1:]),key=lambda x:(x[0],-x[1],-x[2]))
        if gain>0:options.append((gain,i,c))
    options.sort(key=lambda x:(-x[0],x[1],x[2]))
    full=rule.project(case.zero,options,16)
    top=[x for x in options if full[x[1]]==x[2]]
    best=case.zero; best_ps=float('inf'); trace=[];offset=0
    with torch.no_grad():
        for j,size in enumerate(SIZES):
            if block:
                edits=top[offset:size];offset=size
                trial=rule.project(best,edits,len(edits))
            else:trial=rule.project(case.zero,top[:size],size)
            ps=float(bank.forward(bank.hard(trial))[0])
            trace.append(dict(vector=trial,delay_ps=ps))
            tolerance=max(2e-5,2e-6*abs(best_ps)) if j else 0
            if ps<best_ps-tolerance or (abs(ps-best_ps)<=tolerance and len(sparse(trial))<len(sparse(best))):
                best=trial;best_ps=ps
    return full,best,trace

def run(name,out,device,offline=True,reference_work=None,marginal_only=False):
    start=time.monotonic()
    case=FrozenCase(name,out/name,reference_work)
    rules=parse_library(ROOT/'projects/culs/src/algorithms/rewrite_library.inc')
    aig=parse_aig(case.source)
    rule=CandidateFeasibility(aig,case.cat,rules)
    if not any(any(mask[1:]) for mask in rule.allowed()):
        original = case.measure('original',None)
        controls = ('gradient','gradient_no_litecut','local_greedy','global_prefix','global_block','greedy','random','global_marginal')
        save_json(case.out/'selected.json',[original,case.measure('noop',case.zero)]+[dict(original,label=m) for m in controls])
        save_json(case.out/'frozen.json',dict(no_legal_edits=True,methods={m:dict(selection=[],seconds=0,trace=[]) for m in controls}))
        return
    scores=read(case.prep/'scores.json')['scores']
    bank=LoadBank(aig,case.cat,rule,scores,rules,Liberty(case.old/'original/tables',device),
                  case.tools['abc'],case.tools['liberty'],case.out/'bank',.5)
    assert digest(case.out/'bank/forest.aig')==digest(case.old/'templates/forest.aig')
    if marginal_only:
        v,trace,seconds=marginal_greedy(bank,rule,8,16)
        frozen=dict(selection=sparse(v),seconds=seconds,trace=[dict(edits=sparse(t['vector']),proxy_ps=t['delay_ps']) for t in trace])
        save_json(case.out/'marginal_freeze.json',frozen)
        results=[case.measure('original',None),case.measure('noop',case.zero),
                 case.measure('global_marginal',v if any(v) else None)]
        save_json(case.out/'marginal.json',results)
        return
    methods={};traces={};times={}
    for method in ('gradient','greedy','random','gradient_no_litecut'):
        begin=time.monotonic()
        if method=='gradient_no_litecut':bank.set_alpha(0.)
        v,data=optimize(bank,rule,'gradient' if method.startswith('gradient') else method,8,None,16,None,
                        case.inputs['seed'],record_priority=method=='gradient')
        methods[method]=v;traces[method]=data['trace'];times[method]=time.monotonic()-begin
    bank.set_alpha(.5)
    begin=time.monotonic()
    full,best,t=informed(bank,rule,case,scores,False)
    times['local_greedy']=times['global_prefix']=time.monotonic()-begin
    methods.update(local_greedy=full,global_prefix=best)
    traces.update(local_greedy=t,global_prefix=t)
    begin=time.monotonic()
    _,methods['global_block'],traces['global_block']=informed(bank,rule,case,scores,True)
    times['global_block']=time.monotonic()-begin
    if not offline:
        methods['global_marginal'],traces['global_marginal'],times['global_marginal']=marginal_greedy(bank,rule,8,16)
    assert all(1<=len(t)<=8 for t in traces.values())
    if reference_work is None:
        assert all(len(t)==8 for m,t in traces.items() if m!='global_marginal')
    old=read(case.old/'proposal.json')
    parity=methods['gradient']==case.selected
    frozen={m:dict(selection=sparse(v),seconds=times[m],
              trace=[dict(edits=sparse(x['vector']),proxy_ps=x['delay_ps'],priority=x.get('priority')) for x in traces[m]])
            for m,v in methods.items()}
    save_json(case.out/'frozen.json',dict(methods=frozen,device=device,seed=case.inputs['seed'],
              matches_historical_selection=parity,historical_steps_matched=sum(x['vector']==y['vector'] for x,y in zip(old['trace'],traces['gradient'])),
              preprocessing_and_search_s=time.monotonic()-start))
    print(name,'frozen',parity,flush=True)
    measurements=[case.measure('original',None),case.measure('noop',case.zero)]
    for m,v in methods.items():
        measurements.append(case.measure(m,v if any(v) else None))
    if offline and name=='systemcaes_comb':
        match=next((x for x in traces['gradient'] if x['vector']==case.selected),None)
        if match is None: raise RuntimeError('Historical vector did not reproduce; cannot infer priority')
        order=match['priority']
        assert len(order)==16
        for size in (1,4,8,16):
            v=case.zero.copy()
            for i,c in order[:size]:v[i]=c
            rule.validate(v)
            measurements.append(case.measure('nested_'+str(size),v))
    save_json(case.out/'selected.json',measurements)
    if not offline:
        return
    # Offline diagnostics only: labels do not participate in the selectors above.
    for m,trace in traces.items():
        rows=[case.measure(f'{m}_step{i}',x['vector'] if any(x['vector']) else None) for i,x in enumerate(trace)]
        save_json(case.out/(m+'_oracle.json'),rows)

if __name__=='__main__':
    out=Path(sys.argv[1]);out.mkdir()
    torch.set_num_threads(1)
    names=sys.argv[3:]
    mode=names[0] if names[0] in ('--selected-only','--marginal-only') else None
    for name in names[int(mode is not None):]:
        run(name,out,sys.argv[2],offline=mode is None,marginal_only=mode=='--marginal-only')
