"""Recover the frozen historical edit order and finish labels for the previously frozen selectors."""
import importlib.util
from pathlib import Path
import sys
import torch
from paper_joint_evidence import FrozenCase, read, save_json, ROOT, digest
from global_timing.boolean_candidates import parse_library
from global_timing.feasibility import CandidateFeasibility
from global_timing.liberty_cover import Liberty
from global_timing.load_bank import LoadBank
from global_timing.load_search import optimize
from global_timing.structure import parse_aig
import global_timing.load_bank as load_bank


def main(out, freeze):
    out.mkdir()
    case = FrozenCase('systemcaes_comb',out/'systemcaes_comb')
    fixed = read(freeze)
    historical = read(case.old/'proposal.json')
    step = next(r['step'] for r in historical['trace'] if r['vector']==case.selected)
    archived = case.old.parents[2]/'src/global_timing/liberty_cover.py'
    spec = importlib.util.spec_from_file_location('archived_timing',archived)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Use the original autograd implementation on this small diagnostic. Preserve the production default.
    load_bank.TimingGraph = module.TimingGraph
    rules = parse_library(ROOT/'projects/culs/src/algorithms/rewrite_library.inc')
    aig = parse_aig(case.source)
    rule = CandidateFeasibility(aig,case.cat,rules)
    bank = LoadBank(aig,case.cat,rule,read(case.prep/'scores.json')['scores'],rules,
                    Liberty(case.old/'original/tables','cuda'),case.tools['abc'],case.tools['liberty'],
                    case.out/'bank',.5)
    assert digest(case.out/'bank/forest.aig')==digest(case.old/'templates/forest.aig')
    _,trace = optimize(bank,rule,'gradient',step+1,None,16,None,case.inputs['seed'],
                       priority_target=(step,case.selected))
    current = trace['trace'][step]
    order = current['target_order']
    assert {(i,c) for _,i,c in order}=={(i,c) for i,c in enumerate(case.selected) if c}
    assert len(order)==16
    save_json(case.out/'priority.json',dict(values=order,step=step,
              archived_source=str(archived),source_sha256=digest(archived),selector_freeze_sha256=digest(freeze),
              matches_archived_vector=(current['vector']==case.selected),
              historical_proxy=historical['trace'][step]['delay_ps'],rebuilt_proxy=current['delay_ps']))
    def expand(pairs):
        v=case.zero.copy()
        for i,c in pairs:v[i]=c
        rule.validate(v)
        return v
    results=[case.measure('original',None),case.measure('noop',case.zero)]
    for name, record in fixed['methods'].items():
        v=expand(record['selection'])
        results.append(case.measure(name,v if any(v) else None))
    for size in (1,4,8,16):
        results.append(case.measure('nested_'+str(size),expand([(i,c) for _,i,c in order[:size]])))
    assert results[-1]['delay_ps']==read(case.old/'final/result.json')['delay_ps']
    save_json(case.out/'selected.json',results)
    for name, record in fixed['methods'].items():
        rows=[]
        for i, item in enumerate(record['trace']):
            v=expand(item['edits'])
            rows.append(case.measure(f'{name}_step{i}',v if any(v) else None))
        save_json(case.out/(name+'_oracle.json'),rows)
    print('historical vector match',current['vector']==case.selected,flush=True)


if __name__=='__main__':
    torch.set_num_threads(1)
    main(Path(sys.argv[1]),Path(sys.argv[2]))
