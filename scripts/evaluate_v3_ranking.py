#!/usr/bin/env python3
"""Read-only ranking of the frozen v3 pilot. Never imports the model or ABC."""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics as stats

ROOT = Path(__file__).resolve().parents[1]
BASE = Path('analysis/joint_gradient_v3')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ranks(values, tolerance):
    """Bounded-width, minimum-anchored equivalence classes and average ranks."""
    if not values or tolerance < 0 or not all(math.isfinite(v) for v in values):
        raise ValueError('Nonempty finite values and nonnegative tolerance required')
    ordered = sorted(range(len(values)), key=lambda i: (values[i], i))
    result = [0.] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] - values[ordered[start]] <= tolerance:
            end += 1
        for i in ordered[start:end]:
            result[i] = (start + 1 + end) / 2
        start = end
    return result


def pearson(a, b):
    aa = [x - stats.mean(a) for x in a]
    bb = [x - stats.mean(b) for x in b]
    denominator = math.sqrt(sum(x*x for x in aa) * sum(x*x for x in bb))
    return sum(x*y for x, y in zip(aa, bb)) / denominator if denominator else None


def evaluate(pred, actual, indices, tolerance, near_percent):
    if len(pred) != len(actual) or len(pred) != len(indices) or len(set(indices)) != len(indices):
        raise ValueError('Unmatched or duplicate combinations')
    pr, ar = ranks(pred, tolerance), ranks(actual, tolerance)
    counts = dict(correct=0, wrong=0, predicted_tie_only=0, true_tie_only=0, both_tied=0)
    pairs = []
    for i, j in itertools.combinations(range(len(pred)), 2):
        p, a = pr[i] - pr[j], ar[i] - ar[j]
        kind = ('both_tied' if not p and not a else 'predicted_tie_only' if not p
                else 'true_tie_only' if not a else 'correct' if p*a > 0 else 'wrong')
        counts[kind] += 1
        pairs.append(dict(index_a=indices[i], index_b=indices[j], classification=kind))
    c, w, pt, tt, bt = (counts[k] for k in counts)
    denominator = math.sqrt((c+w+pt)*(c+w+tt))
    best = [i for i, r in enumerate(ar) if r == min(ar)]
    top = [i for i, r in enumerate(pr) if r == min(pr)]
    selected = min(top, key=lambda i: indices[i])
    optimum = min(actual)
    if optimum <= 0:
        raise ValueError('Positive Mapping delay required for relative regret')
    regrets = [actual[i] - optimum for i in top]
    regret = actual[selected] - optimum
    total = len(pairs)
    m = dict(spearman=pearson(pr, ar), kendall_tau_b=(c-w)/denominator if denominator else None,
             **counts, pairs=total, true_distinct_pairs=c+w+pt, true_tied_pairs=tt+bt,
             predicted_tied_pairs=pt+bt, predicted_tie_ratio=(pt+bt)/total,
             pairwise_accuracy=c/(c+w+pt) if c+w+pt else None,
             decisive_accuracy=c/(c+w) if c+w else None,
             predicted_groups=len(set(pr)), true_groups=len(set(ar)),
             indistinguishable_combinations=sum(pr.count(r)>1 for r in pr),
             excess_indistinguishable_combinations=len(pr)-len(set(pr)),
             top1_index=indices[selected], true_best_indices=';'.join(str(indices[i]) for i in best),
             predicted_top_indices=';'.join(str(indices[i]) for i in top), top_set_size=len(top),
             top1_hit=int(selected in best), actual_best_delay_ps=optimum,
             selected_actual_delay_ps=actual[selected], top1_regret_ps=regret,
             top1_regret_percent=100*regret/optimum,
             near_optimal=int(100*regret/optimum <= near_percent),
             uniform_tie_top1_hit=sum(i in best for i in top)/len(top),
             uniform_tie_regret_ps=stats.mean(regrets),
             uniform_tie_regret_percent=100*stats.mean(regrets)/optimum,
             top_set_best_regret_percent=100*min(regrets)/optimum,
             top_set_worst_regret_percent=100*max(regrets)/optimum)
    return m, pr, ar, pairs


def aggregate(rows):
    result = dict(circuit_seed_cases=len(rows))
    for k in ('spearman', 'kendall_tau_b'):
        defined = [r[k] for r in rows if r[k] is not None]
        filled = [r[k] if r[k] is not None else 0. for r in rows]
        result.update({k+'_defined': len(defined), k+'_mean_zero_filled': stats.mean(filled),
                       k+'_median_zero_filled': stats.median(filled),
                       k+'_mean_defined_only': stats.mean(defined) if defined else None})
    for k in ('pairwise_accuracy', 'decisive_accuracy'):
        vals = [r[k] for r in rows if r[k] is not None]
        result[k+'_defined'] = len(vals)
        result[k+'_macro'] = stats.mean(vals) if vals else None
    for k in ('correct', 'wrong', 'predicted_tie_only', 'true_tie_only', 'both_tied',
              'true_distinct_pairs', 'true_tied_pairs', 'predicted_tied_pairs', 'top1_hit', 'near_optimal'):
        result[k+'_count'] = sum(r[k] for r in rows)
    for k in ('predicted_tie_ratio', 'indistinguishable_combinations', 'predicted_groups',
              'top_set_size', 'top1_hit', 'near_optimal', 'uniform_tie_top1_hit',
              'top1_regret_ps', 'top1_regret_percent', 'uniform_tie_regret_percent'):
        result[k+'_mean'] = stats.mean(r[k] for r in rows)
    for k in ('top1_regret_ps', 'top1_regret_percent'):
        result[k+'_median'] = stats.median(r[k] for r in rows)
        result[k+'_max'] = max(r[k] for r in rows)
    return result


def write_csv(path, rows):
    with path.open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    protocol_path = ROOT/BASE/'ranking_evaluation/PROTOCOL.json'
    protocol = read(protocol_path)
    manifest_path = ROOT/BASE/'learning_manifest.json'
    selected = [c for c in read(manifest_path)['selected'] if c['role'] == 'validation']
    assert len(selected) == protocol['circuits'] == 12
    ids = [c['circuit_id'] for c in selected]
    names = {c['circuit_id']: c['input'] for c in selected}
    measured_path = ROOT/BASE/'pilot/measured_combinations.csv'
    measured = list(csv.DictReader(measured_path.open()))
    truth = {}
    for r in measured:
        if r['role'] != 'validation' or int(r['index']) == 0:
            continue
        key = (int(r['circuit_id']), int(r['index']))
        assert key not in truth and key[0] in ids
        assert r['status'] == r['cec'] == 'PASS' and r['area_feasible'] == 'True' and int(r['edits']) > 0
        truth[key] = float(r['candidate_delay_ps'])
    expected = {(cid, i) for cid in ids for i in range(1, 9)}
    assert set(truth) == expected and len(truth) == 96
    paths = [protocol_path, manifest_path, measured_path, ROOT/BASE/'pilot/evidence_index.json',
             ROOT/BASE/'pilot/summary.json', ROOT/BASE/'diagnostics/SERIALIZED_NOOP.md']
    cases, combinations, pair_rows, seeds = [], [], [], []
    prediction_audits = []
    for variant in ('fixed', 'ports', 'global', 'combined'):
        for seed in ([protocol['fixed_seed']] if variant == 'fixed' else protocol['seeds']):
            path = ROOT/BASE/'pilot'/f'{variant}_{seed}_validation.json'
            paths.append(path)
            saved = read(path)['predictions']
            records = [r for r in saved if r['domain'] == protocol['domain']]
            keyed = {(r['circuit_id'], r['index']): r for r in records}
            assert len(keyed) == len(records) == 96 and set(keyed) == expected
            for key, r in keyed.items():
                assert r['actual_delay_ps'] == truth[key], f'Label mismatch: {path}:{key}'
                assert r['area_feasible'] is True and math.isfinite(r['predicted_delay_ps'])
            prediction_audits.append(dict(variant=variant, seed=seed, sampled_records=len(records),
                                          excluded_other_domain_records=len(saved)-len(records), exact_label_matches=96))
            group = []
            for cid in ids:
                records = [keyed[cid, i] for i in range(1, 9)]
                pred = [r['predicted_delay_ps'] for r in records]
                actual = [truth[cid, i] for i in range(1, 9)]
                m, pr, ar, pairs = evaluate(pred, actual, list(range(1, 9)),
                    protocol['absolute_tie_tolerance_ps'], protocol['near_optimal_percent'])
                prefix = dict(variant=variant, seed=seed, circuit_id=cid, input=names[cid])
                row = dict(**prefix, **m)
                group.append(row)
                cases.append(row)
                pair_rows.extend(dict(**prefix, **p) for p in pairs)
                for i, r in enumerate(records):
                    combinations.append(dict(**prefix, index=r['index'], predicted_delay_ps=pred[i],
                        actual_delay_ps=actual[i], predicted_rank=pr[i], true_rank=ar[i],
                        predicted_top=int(pr[i] == min(pr)), actual_best=int(ar[i] == min(ar)),
                        selected=int(r['index'] == m['top1_index'])))
            seeds.append(dict(variant=variant, seed=seed, **aggregate(group)))
    overall, stability = [], []
    for variant in ('fixed', 'ports', 'global', 'combined'):
        rows = [r for r in cases if r['variant'] == variant]
        seed_rows = [r for r in seeds if r['variant'] == variant]
        overall.append(dict(variant=variant, seeds=len(seed_rows), **aggregate(rows)))
        for cid in ids:
            cr = [r for r in rows if r['circuit_id'] == cid]
            stability.append(dict(variant=variant, circuit_id=cid, input=names[cid],
                seed_count=len(cr), top1_indices=';'.join(str(r['top1_index']) for r in cr),
                distinct_top1_indices=len({r['top1_index'] for r in cr}),
                spearman_min_zero_filled=min(r['spearman'] or 0. for r in cr),
                spearman_max_zero_filled=max(r['spearman'] or 0. for r in cr),
                pairwise_accuracy_min=min(r['pairwise_accuracy'] for r in cr if r['pairwise_accuracy'] is not None)
                    if any(r['pairwise_accuracy'] is not None for r in cr) else None,
                pairwise_accuracy_max=max(r['pairwise_accuracy'] for r in cr if r['pairwise_accuracy'] is not None)
                    if any(r['pairwise_accuracy'] is not None for r in cr) else None,
                regret_percent_min=min(r['top1_regret_percent'] for r in cr),
                regret_percent_max=max(r['top1_regret_percent'] for r in cr)))
    assert len(cases) == 120 and len(combinations) == 960 and len(pair_rows) == 3360
    args.output.mkdir(parents=True, exist_ok=False)
    for name, rows in [('overall_metrics', overall), ('seed_metrics', seeds), ('circuit_metrics', cases),
                       ('combination_ranks', combinations), ('pair_classifications', pair_rows), ('seed_stability', stability)]:
        write_csv(args.output/(name+'.csv'), rows)
    evidence = dict(protocol=protocol, prediction_audits=prediction_audits,
        counts=dict(unique_circuits=12, unique_combinations=96, model_seed_evaluations=10,
                    circuit_seed_cases=120, ranked_prediction_records=960, classified_pairs=3360,
                    missing_records=0, label_mismatches=0, new_mapping_calls=0),
        provenance=read(ROOT/BASE/'pilot/evidence_index.json'),
        pilot_source_commits=read(ROOT/BASE/'pilot/summary.json'),
        files=[dict(path=str(p.relative_to(ROOT)), sha256=sha(p)) for p in paths],
        output_sha256={p.name: sha(p) for p in args.output.glob('*.csv')})
    (args.output/'evidence_index.json').write_text(json.dumps(evidence, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(dict(counts=evidence['counts'], overall=overall, seeds=seeds), indent=2))


if __name__ == '__main__':
    main()
