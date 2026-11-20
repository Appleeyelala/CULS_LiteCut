"""Tasks 1 and 2: score existing primary proposals without mapping or searching."""
import argparse
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

from diagnostic_io import new_output, primary_runs, read_json, sha256, write_csv


def gains(reference, candidate):
    return 100 * (reference - candidate) / reference


def collect():
    baselines, edited = [], []
    for cohort, name, family, work, expected_sha, original, noop in primary_runs():
        inputs = read_json(work / 'inputs.json')
        proposal = read_json(work / 'proposal.json')
        result = read_json(work / 'result.json')
        assert inputs['method'] == result['method'] == 'gradient' and inputs['alpha'] == .5
        assert expected_sha == inputs['original_sha256'] == sha256(Path(inputs['original']))
        assert result['original_delay_ps'] == original and result['cec'] == 'PASS'
        assert result['vector'] == proposal['vector']
        trace = proposal['trace']
        d0 = proposal['original_model_delay']
        assert not any(trace[0]['vector']) and trace[0]['delay_ps'] == d0
        count = sum(bool(c) for c in proposal['vector'])
        ranked = proposal['rank_delay']
        assert any(t['vector'] == proposal['vector'] and t['delay_ps'] == ranked for t in trace)
        assert min(noop, original, d0, ranked) > 0
        common = dict(cohort=cohort, circuit=name, family=family, aig_sha256=expected_sha,
                      edits=count, delivery_status=result['status'], source_run=str(work),
                      proposal_sha256=sha256(work / 'proposal.json'),
                      result_sha256=sha256(work / 'result.json'))
        error = 100 * (d0 - original) / original
        baselines.append(dict(**common, model_original_ps=d0, abc_original_ps=original,
                              signed_relative_error_pct=error, abs_relative_error_pct=abs(error)))
        if count:
            measured = result['candidate_delay_ps']
            assert result['candidate_cec'] == 'PASS' and measured > 0
            predicted = gains(d0, ranked)
            actual = gains(original, measured)
            vs_noop = gains(noop, measured)
            edited.append(dict(**common, model_original_ps=d0, rank_delay_ps=ranked,
                               abc_original_ps=original, candidate_delay_ps=measured,
                               abc_noop_ps=noop, delivered_ps=result['delay_ps'],
                               predicted_gain_pct=predicted, real_gain_pct=actual,
                               real_gain_vs_noop_pct=vs_noop,
                               predicted_better_actual_worse=predicted > 0 and actual < 0,
                               predicted_better_actual_worse_vs_noop=predicted > 0 and vs_noop < 0))
    assert len({r['aig_sha256'] for r in baselines}) == len(baselines) == 38
    assert len(edited) == 32
    return baselines, edited


def correlation(rows, x, y):
    left = [r[x] for r in rows]
    right = [r[y] for r in rows]
    if min(len(set(left)), len(set(right))) < 2:
        return None, None
    return float(pearsonr(left, right)[0]), float(spearmanr(left, right)[0])


def summarize(baselines, edited):
    base_stats, edit_stats = [], []
    for cohort in ('development', 'external', 'pooled'):
        base = [r for r in baselines if cohort in (r['cohort'], 'pooled')]
        edits = [r for r in edited if cohort in (r['cohort'], 'pooled')]
        pearson, spearman = correlation(base, 'model_original_ps', 'abc_original_ps')
        base_stats.append(dict(cohort=cohort, n=len(base),
                               mean_abs_relative_error_pct=mean(r['abs_relative_error_pct'] for r in base),
                               mean_signed_relative_error_pct=mean(r['signed_relative_error_pct'] for r in base),
                               pearson=pearson, spearman=spearman))
        for reference, key in [('direct_original', 'real_gain_pct'), ('zero_edit_replay', 'real_gain_vs_noop_pct')]:
            pearson, spearman = correlation(edits, 'predicted_gain_pct', key)
            worse = [r['circuit'] for r in edits if r['predicted_gain_pct'] > 0 and r[key] < 0]
            edit_stats.append(dict(cohort=cohort, actual_reference=reference, circuits=len(base), edited=len(edits),
                                   predicted_better=sum(r['predicted_gain_pct'] > 0 for r in edits),
                                   actual_better=sum(r[key] > 0 for r in edits), actual_tied=sum(r[key] == 0 for r in edits),
                                   pearson=pearson, spearman=spearman,
                                   predicted_better_actual_worse=len(worse), wrong_direction_circuits='|'.join(worse)))
    return base_stats, edit_stats


def scatter(path, rows):
    fig, ax = plt.subplots(figsize=(6.5, 5), constrained_layout=True)
    for cohort, color, marker in [('development', '#b54330', 's'), ('external', '#126782', 'o')]:
        points = [r for r in rows if r['cohort'] == cohort]
        ax.scatter([r['predicted_gain_pct'] for r in points], [r['real_gain_pct'] for r in points],
                   label=f'{cohort.capitalize()} (n={len(points)})', c=color, marker=marker,
                   alpha=.85, edgecolors='white', linewidths=.6, s=45, zorder=3)
    ax.axhline(0, color='0.45', lw=.85)
    ax.axvline(0, color='0.45', lw=.85)
    lo, hi = ax.get_xlim()
    ax.set_autoscale_on(False)
    ax.plot([lo, hi], [lo, hi], '--', color='0.55', linewidth=.9, label='Equal improvement')
    ax.grid(alpha=.16)
    ax.set(xlabel='Predicted delay improvement (%)', ylabel='Measured candidate improvement over original (%)')
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(path, metadata={'CreationDate': None})
    plt.close(fig)


def main(path):
    out = new_output(path)
    base, edited = collect()
    bstats, estats = summarize(base, edited)
    write_csv(out / 'baseline_by_circuit.csv', base)
    write_csv(out / 'baseline_summary.csv', bstats)
    write_csv(out / 'edit_by_circuit.csv', edited)
    write_csv(out / 'edit_summary.csv', estats)
    scatter(out / 'edit_scatter.pdf', edited)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    main(parser.parse_args().output)
