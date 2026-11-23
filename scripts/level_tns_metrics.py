#!/usr/bin/env python3
"""Step 3 v3 part A: TNS-oriented local metrics for the seven schemes.

Reads the exported candidates of the whole v3 cohort and, for every chosen
candidate, measures how the near-critical leaf paths move:

    a_l      = level[leaf] + leaf_depths[l]          (candidate)
    a_l_orig = level[leaf] + d_orig(l)               (original structure)
    T_k      = required[node] - k,  k = 0..3
    TNS_k    = sum_l max(0, a_l - T_k)
    N_k      = count_l [a_l > T_k]

Selection rules, shifted soft weights and the relative tie tolerance are the
same as step3_v2. Every assertion from step 3 is repeated here and aborts the
run; the 76 known self-replacement +1 cases stay whitelisted.
"""
import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from global_timing.level_weights import original_leaf_depths, propagate_levels
from global_timing.structure import parse_aig, references

SCHEMES = ('uniform', 'hard', 'tau0.5', 'tau1', 'tau2', 'rootlevel', 'abc_default')
WEIGHTED = ('hard', 'tau0.5', 'tau1', 'tau2')
SOFT = ('tau0.5', 'tau1', 'tau2')
TAUS = {'tau0.5': 0.5, 'tau1': 1.0, 'tau2': 2.0}
BINS = [('0', 0, 0), ('1-2', 1, 2), ('3-5', 3, 5), ('>5', 6, None)]
KS = (0, 1, 2, 3)
TOL = 1e-9
CSV_LIMIT = 20 * 1024 * 1024
CSV_COLUMNS = ['circuit', 'decision_id', 'node_abc', 'node_aiger', 'slack_bin', 'node_slack',
               'n_candidates', 'n_nodes_saved', 'abc_default_keep']
for _scheme in SCHEMES:
    CSV_COLUMNS += [f'{_scheme}_subgraph', f'{_scheme}_root_level', f'{_scheme}_delta_and']
    CSV_COLUMNS += [f'{_scheme}_tns{k}_delta' for k in KS]


def need(condition, message):
    if not condition:
        raise SystemExit('ASSERTION FAILED: ' + message)


def slack_bin(slack):
    for name, low, high in BINS:
        if slack >= low and (high is None or slack <= high):
            return name
    raise SystemExit(f'slack {slack} outside the bins')


def weights(scheme, slacks):
    if scheme == 'uniform':
        return [1.0] * len(slacks)
    if scheme == 'hard':
        tightest = min(slacks)
        return [1.0 if slack == tightest else 0.0 for slack in slacks]
    tau = TAUS[scheme]
    lowest = min(slacks)
    return [math.exp(-(slack - lowest) / tau) for slack in slacks]


def bounded_mffc_size(root, leaves, aig, fanouts):
    boundary = set(leaves)
    remaining = {}
    count = 0
    pending = [root]
    while pending:
        node = pending.pop()
        if node in boundary or node <= aig.pi_count:
            continue
        count += 1
        for literal in aig.fanin_literals[node]:
            source = literal >> 1
            if source in boundary:
                continue
            value = remaining.get(source, fanouts[source]) - 1
            remaining[source] = value
            if value == 0:
                pending.append(source)
    return count


def pick(values, rows, tolerance):
    best = min(values)
    limit = tolerance * max(1.0, abs(best))
    tied = [index for index, value in enumerate(values) if abs(value - best) <= limit]
    choice = min(tied, key=lambda index: (rows[index]['added'], rows[index]['subgraph']))
    return choice, len(tied)


def median_of(counter):
    total = sum(counter.values())
    if not total:
        return None
    seen = 0
    for value in sorted(counter):
        seen += counter[value]
        if seen * 2 >= total:
            return value
    return None


def empty_cell():
    return dict(n=0, tns_sum=0, tns_hist=Counter(), tns_neg=0, tns_zero=0, tns_pos=0,
                n_sum=0, n_hist=Counter(), n_neg=0, n_zero=0, n_pos=0)


def run_circuit(name, group, aig_path, export_path, out_dir, csv_dir):
    aig = parse_aig(aig_path)
    levels = propagate_levels(aig)
    fanouts = references(aig)
    nodes_rows = {}
    with (export_path.parent / 'nodes.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            nodes_rows[row['node']] = row
    offset = aig.po_count

    def to_aig(node_id):
        return node_id if node_id <= aig.pi_count else node_id - offset

    counts = dict(decisions=0, empty_decisions=0, nonempty=0, candidates=0)
    per_node = Counter()
    assertions = dict(rebuild=0, order=0, mffc=0, self_known=0, self_other=0, subgraph_order=0)
    cells = {label: {scheme: {k: empty_cell() for k in KS} for scheme in SCHEMES}
             for label, _, _ in BINS}
    compare = {scheme: {label: {k: dict(n=0, smaller=0, equal=0, larger=0,
                                        same_candidate=0, and_smaller=0, and_equal=0, and_larger=0,
                                        and_hist=Counter()) for k in KS}
                        for label, _, _ in BINS} for scheme in WEIGHTED}

    csv_path = csv_dir / f'{name}.decisions.csv'
    with export_path.open() as stream, csv_path.open('w', newline='') as sink:
        writer = csv.writer(sink)
        writer.writerow(CSV_COLUMNS)
        for line in stream:
            record = json.loads(line)
            counts['decisions'] += 1
            node_abc = record['node']
            per_node[node_abc] += 1
            decision_id = f'{node_abc}#{per_node[node_abc] - 1}'
            leaves_abc = [leaf['id'] for leaf in record['leaves']]
            need(len(leaves_abc) == 4, f'{name}: {node_abc} has {len(leaves_abc)} leaves')
            root = to_aig(node_abc)
            leaves = [to_aig(leaf) for leaf in leaves_abc]
            required = levels.required[root]
            need(required is not None, f'{name}: node {node_abc} has no required level')
            depths, _ = original_leaf_depths(aig, root, leaves, levels.level)
            need(all(depth is not None for depth in depths),
                 f'{name}: {node_abc} has a leaf outside the cut boundary')
            rebuilt = max(levels.level[leaf] + depth for leaf, depth in zip(leaves, depths))
            need(rebuilt == levels.level[root], f'{name}: {node_abc} rebuild mismatch')
            assertions['rebuild'] += 1
            slacks = [required - (levels.level[leaf] + depth) for leaf, depth in zip(leaves, depths)]
            node_level = levels.level[root]
            node_slack = required - node_level
            label = slack_bin(node_slack)
            orig_paths = [levels.level[leaf] + depth for leaf, depth in zip(leaves, depths)]
            orig_tns = {k: sum(max(0, a - (required - k)) for a in orig_paths) for k in KS}
            orig_n = {k: sum(1 for a in orig_paths if a > required - k) for k in KS}

            got_mffc = bounded_mffc_size(root, leaves, aig, fanouts)
            need(got_mffc == record['n_nodes_saved'],
                 f'{name}: {node_abc} MFFC {got_mffc} != {record["n_nodes_saved"]}')
            assertions['mffc'] += 1

            candidates = record['candidates']
            counts['candidates'] += len(candidates)
            subgraphs = [candidate['subgraph'] for candidate in candidates]
            need(sorted(subgraphs) == list(range(len(candidates))),
                 f'{name}: {node_abc} subgraph ids are not evaluation order')
            assertions['subgraph_order'] += 1
            if not candidates:
                counts['empty_decisions'] += 1
                writer.writerow([name, decision_id, node_abc, root, label, node_slack, 0,
                                 record['n_nodes_saved'], 0] + [''] * (len(CSV_COLUMNS) - 9))
                continue
            counts['nonempty'] += 1

            rows = []
            for candidate in candidates:
                leaf_depths = candidate['leaf_depths']
                need(len(leaf_depths) == 4, f'{name}: {node_abc} candidate depth length')
                order_level = max((levels.level[leaf] + depth
                                   for leaf, depth in zip(leaves, leaf_depths) if depth >= 0),
                                  default=0)
                need(order_level == candidate['root_level'],
                     f'{name}: {node_abc} subgraph {candidate["subgraph"]} root_level mismatch')
                assertions['order'] += 1
                if candidate['self']:
                    drift = [i for i in range(4)
                             if leaf_depths[i] >= 0 and leaf_depths[i] != depths[i]]
                    if drift:
                        known = (candidate['root_level'] == node_level + 1
                                 and all(leaf_depths[i] == depths[i] + 1
                                         for i in range(4) if depths[i] is not None))
                        if known:
                            assertions['self_known'] += 1
                        else:
                            assertions['self_other'] += 1
                            raise SystemExit(f'ASSERTION FAILED: {name}: {node_abc} self drift {drift}')
                values = {scheme: sum(w * d for w, d in zip(weights(scheme, slacks), leaf_depths)
                                      if d >= 0) for scheme in SCHEMES[:5]}
                paths = [levels.level[leaf] + depth
                         for leaf, depth in zip(leaves, leaf_depths) if depth >= 0]
                rows.append(dict(subgraph=candidate['subgraph'],
                                 added=0 if candidate['self'] else candidate['nodes_added'],
                                 root_level=candidate['root_level'],
                                 nodes_added=candidate['nodes_added'],
                                 self=candidate['self'],
                                 accepted=candidate['accepted_by_default'],
                                 values=values, paths=paths))

            chosen = {}
            for scheme in SCHEMES[:5]:
                tolerance = TOL if scheme in SOFT else 0.0
                chosen[scheme] = pick([row['values'][scheme] for row in rows], rows, tolerance)[0]
            chosen['rootlevel'] = pick([row['root_level'] for row in rows], rows, 0.0)[0]
            accepted_rows = [(index, row) for index, row in enumerate(rows)
                             if row['accepted'] and not row['self']]
            if accepted_rows:
                gains = [record['n_nodes_saved'] - row['nodes_added'] for _, row in accepted_rows]
                best_gain = max(gains)
                tied = [accepted_rows[k][0] for k, gain in enumerate(gains) if gain == best_gain]
                chosen['abc_default'] = min(tied)
                keep = 0
            else:
                chosen['abc_default'] = None
                keep = 1

            selection = {}
            for scheme in SCHEMES:
                if chosen[scheme] is None:
                    selection[scheme] = dict(subgraph='', root_level=node_level, delta_and=0,
                                             tns={k: 0 for k in KS}, n={k: 0 for k in KS})
                    continue
                row = rows[chosen[scheme]]
                tns = {k: sum(max(0, a - (required - k)) for a in row['paths']) for k in KS}
                nov = {k: sum(1 for a in row['paths'] if a > required - k) for k in KS}
                selection[scheme] = dict(subgraph=row['subgraph'], root_level=row['root_level'],
                                         delta_and=row['nodes_added'] - record['n_nodes_saved'],
                                         tns=tns, n=nov)

            for scheme in SCHEMES:
                picked = selection[scheme]
                for k in KS:
                    cell = cells[label][scheme][k]
                    delta = picked['tns'][k] - orig_tns[k]
                    ndelta = picked['n'][k] - orig_n[k]
                    cell['n'] += 1
                    cell['tns_sum'] += delta
                    cell['tns_hist'][delta] += 1
                    cell['tns_neg'] += delta < 0
                    cell['tns_zero'] += delta == 0
                    cell['tns_pos'] += delta > 0
                    cell['n_sum'] += ndelta
                    cell['n_hist'][ndelta] += 1
                    cell['n_neg'] += ndelta < 0
                    cell['n_zero'] += ndelta == 0
                    cell['n_pos'] += ndelta > 0

            for scheme in WEIGHTED:
                base = selection['rootlevel']
                other = selection[scheme]
                if base['root_level'] != other['root_level']:
                    continue
                same = base['subgraph'] == other['subgraph']
                dand = other['delta_and'] - base['delta_and']
                for k in KS:
                    bucket = compare[scheme][label][k]
                    bucket['n'] += 1
                    bucket['same_candidate'] += same
                    a, b = other['tns'][k], base['tns'][k]
                    bucket['smaller'] += a < b
                    bucket['equal'] += a == b
                    bucket['larger'] += a > b
                    bucket['and_smaller'] += dand < 0
                    bucket['and_equal'] += dand == 0
                    bucket['and_larger'] += dand > 0
                    bucket['and_hist'][dand] += 1

            writer.writerow([name, decision_id, node_abc, root, label, node_slack,
                             len(candidates), record['n_nodes_saved'], keep]
                            + [value for scheme in SCHEMES
                               for value in (selection[scheme]['subgraph'],
                                             selection[scheme]['root_level'],
                                             selection[scheme]['delta_and'])
                               + [selection[scheme]['tns'][k] - orig_tns[k] for k in KS]])

    need(assertions['rebuild'] == counts['decisions'], f'{name}: rebuild coverage')
    need(assertions['mffc'] == counts['decisions'], f'{name}: MFFC coverage')
    need(assertions['order'] == counts['candidates'], f'{name}: candidate order coverage')

    stats = []
    for label, _, _ in BINS:
        for scheme in SCHEMES:
            for k in KS:
                cell = cells[label][scheme][k]
                if not cell['n']:
                    continue
                stats.append(dict(circuit=name, group=group, band=label, scheme=scheme, k=k,
                                  n=cell['n'], tns_mean=cell['tns_sum'] / cell['n'],
                                  tns_median=median_of(cell['tns_hist']),
                                  tns_smaller=cell['tns_neg'], tns_equal=cell['tns_zero'],
                                  tns_larger=cell['tns_pos'],
                                  n_mean=cell['n_sum'] / cell['n'],
                                  n_median=median_of(cell['n_hist']),
                                  n_smaller=cell['n_neg'], n_equal=cell['n_zero'],
                                  n_larger=cell['n_pos']))
    stats_path = out_dir / 'stats' / f'{name}.tsv'
    stats_path.parent.mkdir(exist_ok=True)
    with stats_path.open('w', newline='') as sink:
        writer = csv.DictWriter(sink, fieldnames=list(stats[0].keys()), delimiter='\t')
        writer.writeheader()
        writer.writerows(stats)

    size = csv_path.stat().st_size
    return dict(circuit=name, group=group, ands=aig.and_count, counts=counts,
                assertions=assertions,
                bins={label: {scheme: {k: _cell_json(cells[label][scheme][k]) for k in KS}
                              for scheme in SCHEMES} for label, _, _ in BINS},
                hists={label: {scheme: {k: dict(tns=dict(cells[label][scheme][k]['tns_hist']),
                                                n=dict(cells[label][scheme][k]['n_hist']))
                                        for k in KS} for scheme in SCHEMES}
                       for label, _, _ in BINS},
                compare={scheme: {label: {k: _compare_json(compare[scheme][label][k])
                                          for k in KS} for label, _, _ in BINS}
                         for scheme in WEIGHTED},
                stats_tsv=str(stats_path),
                csv=dict(path=str(csv_path), bytes=size, keep=size <= CSV_LIMIT))


def _cell_json(cell):
    if not cell['n']:
        return dict(n=0)
    return dict(n=cell['n'], tns_mean=cell['tns_sum'] / cell['n'],
                tns_median=median_of(cell['tns_hist']),
                tns_smaller=cell['tns_neg'], tns_equal=cell['tns_zero'], tns_larger=cell['tns_pos'],
                n_mean=cell['n_sum'] / cell['n'], n_median=median_of(cell['n_hist']),
                n_smaller=cell['n_neg'], n_equal=cell['n_zero'], n_larger=cell['n_pos'])


def _compare_json(bucket):
    if not bucket['n']:
        return dict(n=0)
    return dict(n=bucket['n'], same_candidate=bucket['same_candidate'],
                tns_smaller=bucket['smaller'], tns_equal=bucket['equal'],
                tns_larger=bucket['larger'],
                and_smaller=bucket['and_smaller'], and_equal=bucket['and_equal'],
                and_larger=bucket['and_larger'], and_median=median_of(bucket['and_hist']),
                and_hist={str(key): value for key, value in bucket['and_hist'].items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--cohort', required=True, type=Path)
    parser.add_argument('--export-manifest', type=Path, default=None,
                        help='export_manifest.json of the export run; defaults to this run')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    run = args.run.resolve()
    if not os.getenv('SLURM_JOB_ID'):
        raise SystemExit('run this only inside the committed /data Slurm job')
    out = run / 'output'
    (out / 'stats').mkdir(parents=True, exist_ok=True)
    cohort = json.loads(args.cohort.read_text())
    manifest_path = args.export_manifest or (out / 'export_manifest.json')
    manifest = json.loads(Path(manifest_path).read_text())

    cohort_paths = {}
    for row in cohort['pilots'] + cohort['development_new'] + cohort['sampled']:
        cohort_paths[row['name']] = row['path']

    prepared = []
    for row in manifest['results']:
        if row.get('status') != 'PASS':
            continue
        name = row['circuit']
        prepared.append((name, row['group'], Path(cohort_paths[name]),
                         Path(row['export']) / 'cuts.jsonl', out, out / 'decisions'))
    (out / 'decisions').mkdir(exist_ok=True)

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_circuit, name, group, aig, export, out, csv_dir)
                   for name, group, aig, export, out, csv_dir in prepared]
        for future in futures:
            result = future.result()
            results.append(result)
            print(f"{result['circuit']}: cuts={result['counts']['decisions']} "
                  f"csv={result['csv']['bytes']}", flush=True)

    totals = {}
    for label, _, _ in BINS:
        totals[label] = {}
        for scheme in SCHEMES:
            totals[label][scheme] = {}
            for k in KS:
                tns = Counter()
                nov = Counter()
                for result in results:
                    tns.update({int(key): value for key, value in
                                result['hists'][label][scheme][k]['tns'].items()})
                    nov.update({int(key): value for key, value in
                                result['hists'][label][scheme][k]['n'].items()})
                count = sum(tns.values())
                if not count:
                    totals[label][scheme][k] = dict(n=0)
                    continue
                total = sum(value * weight for value, weight in tns.items())
                totals[label][scheme][k] = dict(
                    n=count, tns_mean=total / count, tns_median=median_of(tns),
                    tns_smaller=sum(w for v, w in tns.items() if v < 0),
                    tns_equal=tns.get(0, 0),
                    tns_larger=sum(w for v, w in tns.items() if v > 0),
                    n_mean=sum(v * w for v, w in nov.items()) / count,
                    n_median=median_of(nov),
                    n_smaller=sum(w for v, w in nov.items() if v < 0),
                    n_equal=nov.get(0, 0),
                    n_larger=sum(w for v, w in nov.items() if v > 0))

    compare_totals = {}
    for scheme in WEIGHTED:
        compare_totals[scheme] = {}
        for label, _, _ in BINS:
            compare_totals[scheme][label] = {}
            for k in KS:
                merged = dict(n=0, same_candidate=0, tns_smaller=0, tns_equal=0, tns_larger=0,
                              and_smaller=0, and_equal=0, and_larger=0, hist=Counter())
                for result in results:
                    cell = result['compare'][scheme][label][k]
                    if not cell.get('n'):
                        continue
                    merged['n'] += cell['n']
                    merged['same_candidate'] += cell['same_candidate']
                    for key in ('tns_smaller', 'tns_equal', 'tns_larger',
                                'and_smaller', 'and_equal', 'and_larger'):
                        merged[key] += cell[key]
                    merged['hist'].update({int(v): w for v, w in cell['and_hist'].items()})
                compare_totals[scheme][label][k] = dict(
                    n=merged['n'], same_candidate=merged['same_candidate'],
                    tns_smaller=merged['tns_smaller'], tns_equal=merged['tns_equal'],
                    tns_larger=merged['tns_larger'], and_smaller=merged['and_smaller'],
                    and_equal=merged['and_equal'], and_larger=merged['and_larger'],
                    and_median=median_of(merged['hist']) if merged['n'] else None)

    summary = dict(step='3_v3', part='A', circuits=results, totals=totals,
                   compare_totals=compare_totals,
                   cohort=dict(total=cohort['total_unique'], shortfalls=cohort['shortfalls'],
                               skipped=cohort['skipped'], seed=cohort['seed']),
                   slurm_job_id=os.getenv('SLURM_JOB_ID'))
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print('done', len(results), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
