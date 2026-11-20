"""Check per-state gradient records and assemble the four circuit traces."""
import argparse
import json
from pathlib import Path

import numpy as np

from diagnostic_io import new_output, read_csv, read_json, sha256, write_csv


def check_case(case):
    steps = read_csv(case / 'gradient_steps.csv')
    selected = read_csv(case / 'gradient_path_options.csv')
    parity = read_csv(case / 'trace_parity.csv')
    traced = read_json(case / 'sparse_traces.json')['observed']
    chosen = {tuple(pair) for pair in traced['selection']}
    assert len(steps) == len(traced['trace']) == 8
    assert len(selected) == 8 * len(chosen)
    with np.load(case / 'all_options.npz') as saved:
        group, choice, decision = (saved[key] for key in ('group', 'choice', 'decision'))
        legal = choice > 0
        roots = saved['original_root_by_group']
        paths = saved['hard_path_by_step_group']
        grads = saved['soft_logit_grad_by_step_state']
        assert np.isfinite(grads).all() and grads.shape == (7, len(choice))
        assert paths.shape == (8, len(roots))
        lookup = {}
        for pair in chosen:
            (idx,) = np.flatnonzero((decision == pair[0]) & (choice == pair[1]))
            lookup[pair] = idx
        for i, row in enumerate(steps):
            assert row['circuit'] == case.name and int(row['step']) == i
            assert float(row['hard_score_ps']) == traced['trace'][i]['delay_ps']
            on_path = paths[i, group]
            assert int(row['path_roots']) == paths[i].sum()
            assert int(row['path_options']) == (on_path & legal).sum()
            assert int(row['off_path_options']) == (~on_path & legal).sum()
            assert int(row['legal_edit_options']) == legal.sum()
            for label, mask in [('legal_edit', legal), ('noop', ~legal),
                                ('on_path_edit', on_path & legal), ('off_path_edit', ~on_path & legal)]:
                nonzero = row[label + '_grad_nonzero']
                significant = row[label + '_grad_abs_gt_1e_8']
                if i == 7:
                    assert nonzero == significant == ''
                else:
                    assert int(nonzero) == (grads[i, mask] != 0).sum()
                    assert int(significant) == (abs(grads[i, mask]) > 1e-8).sum()
            sub = [o for o in selected if int(o['step']) == i]
            assert {(int(o['final_decision']), int(o['final_choice'])) for o in sub} == chosen
            assert sum(o['on_discrete_path'] == 'True' for o in sub) == int(row['final_edits_on_path'])
            current = {tuple(pair) for pair in traced['trace'][i]['edits']}
            for o in sub:
                pair = int(o['final_decision']), int(o['final_choice'])
                k = lookup[pair]
                assert int(o['original_aig_root']) == roots[group[k]]
                assert (o['on_discrete_path'] == 'True') == on_path[k]
                assert (o['active_in_this_hard_vector'] == 'True') == (pair in current)
                assert (o['soft_logit_gradient'] == '' if i == 7 else float(o['soft_logit_gradient']) == grads[i, k])
    for comparison in ('observed_vs_control', 'observed_vs_archive'):
        assert [int(r['step']) for r in parity if r['comparison'] == comparison] == list(range(8))
    return steps, selected, parity


def main(output, directories):
    dest = new_output(output)
    names = ('c5315', 'intra_pred_PE', 'cabac_ucontext', 'unreg')
    tables = [[], [], []]
    sources = {}
    for name in names:
        case, = [d / name for d in directories if (d / name).is_dir()]
        for table, rows in zip(tables, check_case(case)):
            table.extend(rows)
        sources[name] = dict(records=str(case), metadata=read_json(case / 'source.json'),
                             all_options_sha256=sha256(case / 'all_options.npz'))
    for filename, rows in zip(('gradient_steps.csv', 'gradient_path_options.csv', 'trace_parity.csv'), tables):
        write_csv(dest / filename, rows)
    (dest / 'gradient_sources.json').write_text(json.dumps(sources, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('directories', type=Path, nargs='+')
    args = parser.parse_args()
    main(args.output, args.directories)
