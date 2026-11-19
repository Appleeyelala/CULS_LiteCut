"""Paired size and PO calibration pilot with one submitted proposal per arm and separate offline labels."""
import sys
from pathlib import Path
import torch
from paper_joint_evidence import FrozenCase, ROOT, read, save_json, digest
from global_timing.boolean_candidates import parse_library
from global_timing.evaluation import parse_timing
from global_timing.feasibility import CandidateFeasibility
from global_timing.liberty_cover import Liberty
from global_timing.load_bank import LoadBank
from global_timing.load_search import optimize
from global_timing.structure import parse_aig


def run(name, out):
    case = FrozenCase(name, out / name)
    rules = parse_library(ROOT / 'projects/culs/src/algorithms/rewrite_library.inc')
    aig = parse_aig(case.source)
    rule = CandidateFeasibility(aig, case.cat, rules)
    bank = LoadBank(aig, case.cat, rule, read(case.prep / 'scores.json')['scores'], rules,
                    Liberty(case.old / 'original/tables', 'cuda'), case.tools['abc'], case.tools['liberty'],
                    case.out / 'bank', .5)
    assert digest(case.out/'bank/forest.aig') == digest(case.old/'templates/forest.aig')
    original = parse_timing((case.old / 'original/mapping.log').read_text())
    observations = [[po['rise_ps'], po['fall_ps']] for po in original['po_rise_fall']]
    selected = {}
    traces = {}
    frozen = {}
    for anchored in (False, True):
        for adaptive in (False, True):
            method = ('anchored' if anchored else 'raw') + ('_adaptive' if adaptive else '_fixed')
            v, trace = optimize(bank, rule, 'gradient', 8, None, 16, None, case.inputs['seed'],
                                adaptive_sizes=adaptive, observed_po=observations if anchored else None)
            selected[method] = v
            traces[method] = trace['trace']
            assert trace['hard_calls'] == 8
            frozen[method] = dict(trace=[dict(pairs=[(i,c) for i,c in enumerate(r['vector']) if c],
                                                proxy_ps=r['delay_ps'], capacity=r.get('edit_capacity',16 if r['step'] else 0))
                                          for r in trace['trace']], seconds=trace['seconds'],
                                  selected=[(i,c) for i,c in enumerate(v) if c])
    save_json(case.out / 'frozen.json', dict(methods=frozen, historical_match=selected['raw_fixed']==case.selected))
    results = [case.measure('original', None), case.measure('noop',case.zero)]
    for method, v in selected.items():
        results.append(case.measure(method, v if any(v) else None))
    save_json(case.out / 'selected.json', results)
    for method, trace in traces.items():
        rows = [case.measure(f'{method}_step{i}', r['vector'] if any(r['vector']) else None)
                for i,r in enumerate(trace)]
        save_json(case.out / (method + '_oracle.json'), rows)
    print(name,[(r['label'],r.get('delay_ps')) for r in results],flush=True)


if __name__ == '__main__':
    torch.set_num_threads(1)
    out = Path(sys.argv[1])
    out.mkdir()
    for name in sys.argv[2:]:
        run(name, out)
