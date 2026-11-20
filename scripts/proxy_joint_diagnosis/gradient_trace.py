"""Task 3: observe production Adam steps, then trace each hard candidate offline."""
import argparse
from functools import wraps
import json
import math
import os
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import torch

from diagnostic_io import ROOT, new_output, primary_runs, read_json, sha256, write_csv
from discrete_path import critical_path

sys.path.insert(0, str(ROOT / 'scripts'))
from paper_joint_evidence import FrozenCase
from global_timing.boolean_candidates import parse_library
from global_timing.feasibility import CandidateFeasibility
from global_timing.liberty_cover import Liberty
from global_timing.load_bank import LoadBank
from global_timing.load_search import optimize
from global_timing.structure import parse_aig


def close_score(a, b):
    return math.isclose(a, b, rel_tol=2e-6, abs_tol=2e-5)


def search(bank, rule, inputs):
    assert inputs['seconds'] is None and inputs['mode'] == 'delay' and inputs['alpha'] == .5
    assert not inputs.get('adaptive_sizes') and not inputs.get('anchor_outputs')
    vector, data = optimize(bank, rule, 'gradient', inputs['steps'], None,
                            inputs['max_edits'], None, inputs['seed'])
    return dict(vector=vector, trace=data['trace'])


def observed_search(bank, rule, inputs):
    gradients, norms = [], []
    original_step = torch.optim.Adam.step

    @wraps(original_step)
    def record_before_update(optimizer, *args, **kwargs):
        (group,) = optimizer.param_groups
        (parameter,) = group['params']
        grad = parameter.grad.detach()
        assert grad.numel() == bank.states
        gradients.append(grad.cpu().numpy().copy())
        norms.append(float(grad.norm()))
        return original_step(optimizer, *args, **kwargs)

    with patch.object(torch.optim.Adam, 'step', record_before_update):
        data = search(bank, rule, inputs)
    assert norms == [t['gradient_norm'] for t in data['trace'] if 'gradient_norm' in t]
    return data, gradients


def sparse(vector):
    return [(i, c) for i, c in enumerate(vector) if c]


def compact(data):
    return dict(selection=sparse(data['vector']), trace=[
        {'step': t['step'], 'edits': sparse(t['vector']), 'delay_ps': t['delay_ps'],
         'gradient_norm': t.get('gradient_norm')} for t in data['trace']])


def compare(name, data, reference, label):
    trace = data['trace']
    assert len(trace) == len(reference['trace'])
    first_diff = next((t['step'] for t, u in zip(trace, reference['trace'])
                       if t['vector'] != u['vector']), None)
    rows = []
    for t, u in zip(trace, reference['trace']):
        assert t['step'] == u['step']
        rows.append(dict(circuit=name, comparison=label, step=t['step'],
                         same_step_vector=t['vector'] == u['vector'],
                         diagnostic_ps=t['delay_ps'], reference_ps=u['delay_ps'],
                         within_score_tolerance=close_score(t['delay_ps'], u['delay_ps']),
                         diagnostic_grad_norm=t.get('gradient_norm'), reference_grad_norm=u.get('gradient_norm'),
                         first_different_vector_step=first_diff, same_final_vector=data['vector'] == reference['vector']))
    return rows


def group_layout(bank, case, pi_count):
    decision, choice = np.asarray([pair for g in bank.groups for pair in g], dtype=np.int32).T
    group_ids = bank.group_ids.cpu().numpy().astype(np.int32)
    roots = np.asarray([case.cat.decisions[g[1][0]]['root'] for g in bank.groups], dtype=np.int32)
    assert np.all(np.diff(roots) > 0)
    for root, group in zip(roots, bank.groups):
        assert all(case.cat.decisions[i]['root'] == root for i, _ in group[1:])
    positions = np.asarray([bank.graph.positions.get(f'pin{pi_count + j}', -1)
                            for j in range(len(bank.groups))], dtype=np.int64)
    return decision, choice, group_ids, roots, positions


def analyze_steps(name, case, bank, data, gradients, pi_count, out):
    decision, choice, group, roots, positions = group_layout(bank, case, pi_count)
    valid = choice != 0
    final = np.asarray(data['vector'])
    final_indices = np.flatnonzero(valid & (final[decision] == choice))
    assert len(final_indices) == np.count_nonzero(final)
    rows, options, path_memberships = [], [], []
    for step, t in enumerate(data['trace']):
        vector = t['vector']
        hard_ps, endpoints, marked = critical_path(bank, vector)
        if not close_score(t['delay_ps'], hard_ps):
            raise ValueError(f'{name} step {step}: replayed hard score differs from search')
        by_net = marked.any(dim=1).cpu().numpy()
        on_root = np.zeros(len(roots), dtype=bool)
        visible = positions >= 0
        on_root[visible] = by_net[positions[visible]]
        path_memberships.append(on_root)
        on_path = on_root[group]
        current = np.asarray(vector)
        current_mask = valid & (current[decision] == choice)
        assert int(current_mask.sum()) == np.count_nonzero(current)
        grad = gradients[step] if step < len(gradients) else None
        row = dict(circuit=name, step=step, hard_score_ps=t['delay_ps'], replay_hard_ps=hard_ps,
                   path_state='discrete_candidate', gradient_state='probability_mixture_logits' if grad is not None else None,
                   hard_critical_po_transition='|'.join(f'{i}.{"rise" if e == 0 else "fall"}' for i, e in endpoints),
                   editable_roots=len(roots), reachable_editable_roots=int(visible.sum()), path_roots=int(on_root.sum()),
                   legal_edit_options=int(valid.sum()), path_options=int((valid & on_path).sum()),
                   off_path_options=int((valid & ~on_path).sum()), noop_options=int((~valid).sum()),
                   step_edits=int(current_mask.sum()), step_edits_on_path=int((current_mask & on_path).sum()),
                   final_proposal_edits=len(final_indices), final_edits_on_path=int(on_path[final_indices].sum()),
                   final_edits_on_path_active_here=int((on_path[final_indices] & current_mask[final_indices]).sum()))
        categories = {'legal_edit': valid, 'on_path_edit': valid & on_path, 'off_path_edit': valid & ~on_path,
                      'noop': ~valid}
        for label, mask in categories.items():
            row[label + '_grad_nonzero'] = int(np.count_nonzero(grad[mask])) if grad is not None else None
            row[label + '_grad_abs_gt_1e_8'] = int((np.abs(grad[mask]) > 1e-8).sum()) if grad is not None else None
        rows.append(row)
        for k in final_indices:
            i, c = int(decision[k]), int(choice[k])
            options.append(dict(circuit=name, step=step, final_decision=i, final_choice=c,
                                original_aig_root=int(roots[group[k]]),
                                mffc_owner=case.cat.decisions[i]['decision_owner_region'],
                                on_discrete_path=bool(on_path[k]), active_in_this_hard_vector=bool(current_mask[k]),
                                soft_logit_gradient=float(grad[k]) if grad is not None else None))
    # One state index addresses a unique option (including group-specific no-op).
    # Seven rows of gradients, eight rows of hard-path membership by root.
    np.savez_compressed(out / 'all_options.npz', decision=decision, choice=choice, group=group,
                        original_root_by_group=roots, hard_path_by_step_group=np.stack(path_memberships),
                        soft_logit_grad_by_step_state=np.stack(gradients))
    write_csv(out / 'gradient_steps.csv', rows)
    write_csv(out / 'gradient_path_options.csv', options)


def run(name, out):
    cohort, _, family, work, expected_sha, _, _ = next(r for r in primary_runs() if r[1] == name)
    case = FrozenCase(name, out / name, reference_work=work)
    assert expected_sha == case.inputs['original_sha256']
    inputs = case.inputs
    historical = read_json(work / 'proposal.json')
    aig = parse_aig(case.source)
    rules = parse_library(ROOT / 'projects/culs/src/algorithms/rewrite_library.inc')
    rule = CandidateFeasibility(aig, case.cat, rules)
    print(name, 'preparing production timing graph', flush=True)
    bank = LoadBank(aig, case.cat, rule, read_json(case.prep / 'scores.json')['scores'], rules,
                    Liberty(work / 'original/tables', 'cuda'), case.tools['abc'], case.tools['liberty'],
                    case.out / 'templates', inputs['alpha'])
    forest_hash = sha256(case.out / 'templates/forest.aig')
    assert sha256(work / 'templates/forest.aig') == forest_hash
    print(name, 'unobserved search, then observed search', flush=True)
    control = search(bank, rule, inputs)
    recorded, grads = observed_search(bank, rule, inputs)
    traces = {'observed': compact(recorded), 'control': compact(control)}
    assert len(grads) == 7 and len(recorded['trace']) == 8
    pair = compare(name, recorded, control, 'observed_vs_control')
    parity = pair + compare(name, recorded, historical, 'observed_vs_archive')
    if any(not r['same_step_vector'] or not r['within_score_tolerance'] for r in pair):
        repeat = search(bank, rule, inputs)
        traces['repeat_control'] = compact(repeat)
        parity += compare(name, repeat, control, 'unobserved_repeat_vs_control')
    write_csv(case.out / 'trace_parity.csv', parity)
    (case.out / 'sparse_traces.json').write_text(json.dumps(traces) + '\n')
    print(name, 'tracing hard paths after search', flush=True)
    analyze_steps(name, case, bank, recorded, grads, aig.pi_count, case.out)
    provenance = dict(cohort=cohort, family=family, historical_work=str(work), historical_inputs=inputs,
                      prepared=read_json(case.prep / 'prepared.json'), historical_proposal_sha256=sha256(work / 'proposal.json'),
                      original_tables_sha256=sha256(work / 'original/tables'), shared_forest_sha256=forest_hash,
                      slurm_job=os.environ['SLURM_JOB_ID'], torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                      peak_torch_bytes=torch.cuda.max_memory_allocated())
    (case.out / 'source.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(name, 'complete', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('circuits', nargs='+', choices=['c5315', 'intra_pred_PE', 'cabac_ucontext', 'unreg'])
    args = parser.parse_args()
    out = new_output(args.output)
    torch.set_num_threads(1)
    for circuit in args.circuits:
        torch.cuda.reset_peak_memory_stats()
        run(circuit, out)
