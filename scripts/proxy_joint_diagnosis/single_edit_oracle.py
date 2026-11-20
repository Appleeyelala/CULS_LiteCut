"""Compare additive selection using measured, LiteCut-on and LiteCut-off single edits."""
import argparse
from itertools import combinations
import json
from math import fsum, isclose, isfinite
from pathlib import Path
import sys

import torch

from diagnostic_io import ROOT, RESULTS, new_output, read_csv, read_json, sha256, write_csv
sys.path.insert(0, str(ROOT / 'scripts'))
from paper_joint_evidence import FrozenCase
from global_timing.boolean_candidates import parse_library
from global_timing.feasibility import CandidateFeasibility
from global_timing.liberty_cover import Liberty
from global_timing.load_bank import LoadBank
from global_timing.structure import parse_aig


def vector(zero, pairs):
    result = zero.copy()
    for decision, choice in pairs:
        if result[decision] or not choice:
            raise ValueError('Invalid duplicate or empty pool edit')
        result[decision] = choice
    return result


def best_subset(pool, gains, case, rule):
    # Positive terms alone attain an upper bound even when edits can conflict.
    positive = tuple(p for p in pool if gains[p] > 0)
    try:
        rule.validate(vector(case.zero, positive))
        return positive
    except ValueError:
        # The pools have at most 16 edits. Prefer fewer edits, then sorted IDs
        # at a tied objective. The empty combination is feasible.
        ordered = sorted((s for k in range(len(pool)+1) for s in combinations(pool, k)),
                         key=lambda s: (-fsum(gains[p] for p in s), len(s), s))
        for subset in ordered:
            try:
                rule.validate(vector(case.zero, subset))
                return subset
            except ValueError:
                continue
        raise ValueError('No legal subset, including empty')


def verify_measurement(work, pairs, case, primary=False):
    result = read_json(work / ('final/result.json' if primary else 'evaluation/result.json'))
    assert result['cec'] == 'PASS'
    for key, digest in [('original', case.inputs['original_sha256']),
                        ('library', case.inputs['tools']['liberty']['sha256']),
                        ('abc', case.inputs['tools']['frozen_abc']['sha256'])]:
        assert result['provenance'][key]['sha256'] == digest
    if pairs is not None:
        replay = read_json(work / 'replay/replay.json')
        assert [(i,c) for i,c in enumerate(replay['choices']) if c] == list(pairs)
        assert replay['status'] == 'PASS' and replay['applied_rewrites'] == len(pairs)
        assert result['input_sha256'] == sha256(Path(replay['aig']))
    else:
        assert result['input_sha256'] == case.inputs['original_sha256']
    return result


def archive(case, name):
    path = RESULTS / ('challenger_ablation.csv' if name == 'cabac_ucontext' else 'isolated.csv')
    history = {}
    for row in read_csv(path):
        label = row['method']
        if row['circuit'] != name or not (label in ('original','noop','joint') or label.startswith('single_')):
            continue
        pairs = json.loads(row['edit_pairs'])
        pairs = None if pairs is None else tuple(map(tuple, pairs))
        work = Path(row['work'])
        m = verify_measurement(work, pairs, case)
        assert (m['input_sha256'], m['delay_ps']) == (row['raw_aig_sha256'], float(row['raw_ps']))
        history[label] = (pairs, m, work)
    return history


@torch.no_grad()
def singles(bank, zero, pool, alpha):
    bank.set_alpha(alpha)
    base = float(bank.forward(bank.hard(zero))[0])
    values = {p: float(bank.forward(bank.hard(vector(zero, [p])))[0]) for p in pool}
    assert base > 0 and all(isfinite(v) and v > 0 for v in values.values())
    return base, values


def run(out, name):
    case = FrozenCase(name, out / name)
    raw = read_json(case.old / 'result.json')
    history = archive(case, name)
    pool = tuple(sorted(history['joint'][0]))
    measured = {pairs[0]: m['delay_ps'] for label, (pairs, m, _) in history.items() if label.startswith('single_')}
    assert len(set(pool)) == len(pool) == (10 if name == 'cabac_ucontext' else 16)
    assert set(pool) == set(measured)
    original = history['original'][1]['delay_ps']
    noop = history['noop'][1]['delay_ps']
    print(name, 'verified archive, building frozen timing graph', flush=True)
    aig = parse_aig(case.source)
    rules = parse_library(ROOT / 'projects/culs/src/algorithms/rewrite_library.inc')
    rule = CandidateFeasibility(aig, case.cat, rules)
    rule.validate(vector(case.zero, pool))
    bank = LoadBank(aig, case.cat, rule, read_json(case.prep/'scores.json')['scores'], rules,
                    Liberty(case.old/'original/tables', 'cuda'), case.tools['abc'],
                    case.tools['liberty'], case.out/'bank', .5)
    assert sha256(case.out/'bank/forest.aig') == sha256(case.old/'templates/forest.aig')
    estimates = {'real': (noop, measured),
                 'litecut_on': singles(bank, case.zero, pool, .5),
                 'litecut_off': singles(bank, case.zero, pool, 0.)}
    assert isclose(estimates['litecut_on'][0], read_json(case.old/'proposal.json')['original_model_delay'],
                   abs_tol=2e-5, rel_tol=2e-6)
    del bank
    torch.cuda.empty_cache()
    gains = {tag: {p: base - delays[p] for p in pool} for tag, (base, delays) in estimates.items()}
    selected = {tag: best_subset(pool, values, case, rule) for tag, values in gains.items()}
    # Commit the three choices to disk before measuring any combined candidate.
    (case.out/'selections.json').write_text(json.dumps(selected, indent=2)+'\n')
    print(name, 'frozen selections', {a:len(s) for a,s in selected.items()}, flush=True)
    individual = []
    for p in pool:
        row = dict(circuit=name, decision=p[0], choice=p[1], mffc=case.cat.decisions[p[0]]['decision_owner_region'],
                   source_work=str(history[f'single_{p[0]}'][2]))
        for tag, (base, delays) in estimates.items():
            row.update({f'{tag}_noop_ps':base, f'{tag}_single_ps':delays[p],
                        f'{tag}_gain_ps':gains[tag][p], f'{tag}_chosen':p in selected[tag]})
        individual.append(row)
    write_csv(case.out/'oracle_single_edits.csv', individual)
    arms = []
    def add(tag, pairs, m, work, deployed):
        arms.append(dict(circuit=name, group=tag, pool_edits=len(pool), selected_edits=len(pairs or ()),
                         pairs=json.dumps(pairs), additive_gain_ps=fsum(gains[tag][p] for p in pairs) if tag in gains else None,
                         original_ps=original, noop_ps=noop, raw_ps=m['delay_ps'], delta_vs_noop_ps=m['delay_ps']-noop,
                         area=m['area'], cec=m['cec'], returned_ps=deployed, input_sha256=m['input_sha256'], work=str(work)))
    for tag, pairs in selected.items():
        m = case.measure(tag, vector(case.zero, pairs))
        assert m['status'] == 'PASS'
        work = Path(m['work'])
        validated = verify_measurement(work, pairs, case)
        assert validated['input_sha256'] == m['input_sha256'] and validated['delay_ps']==m['delay_ps']
        # When a selection coincides with an archived measured candidate, its
        # fresh replay and mapping must reproduce the stored file and delay.
        for h_pairs, previous, _ in history.values():
            if pairs == h_pairs:
                assert (m['input_sha256'], m['delay_ps']) == (previous['input_sha256'], previous['delay_ps'])
        add(tag, pairs, validated, work, min(original,m['delay_ps']) if pairs else original)
    best = min(pool,key=lambda p:(measured[p],p))
    for label, source in [('original','original'),('noop','noop'),('original_pool_joint','joint'),
                          ('best_measured_single',f'single_{best[0]}')]:
        pairs, m, work = history[source]
        add(label, pairs, m, work, min(original,m['delay_ps']) if pairs else original)
    adam_pairs = tuple((i,c) for i,c in enumerate(case.selected) if c)
    adam = verify_measurement(case.old, adam_pairs, case, primary=True)
    assert raw['candidate_delay_ps'] == adam['delay_ps']
    add('original_adam', adam_pairs, adam, case.old, raw['delay_ps'])
    write_csv(case.out/'oracle_arms.csv', arms)
    (case.out/'source.json').write_text(json.dumps(dict(primary_inputs=case.inputs, prepared=read_json(case.prep/'prepared.json'),
        original_template_forest_sha256=sha256(case.old/'templates/forest.aig'),
        original_tables_sha256=sha256(case.old/'original/tables'), torch=torch.__version__, gpu=torch.cuda.get_device_name(),
        peak_cuda_bytes=torch.cuda.max_memory_allocated()),indent=2)+'\n')
    print(name, 'complete', [(r['group'],r['raw_ps']) for r in arms[:3]], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('circuits', nargs='+', choices=['c5315','intra_pred_PE','cabac_ucontext'])
    args = parser.parse_args()
    out = new_output(args.output)
    torch.set_num_threads(1)
    for name in args.circuits:
        torch.cuda.reset_peak_memory_stats()
        run(out, name)
