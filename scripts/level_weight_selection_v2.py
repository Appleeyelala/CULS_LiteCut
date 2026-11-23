#!/usr/bin/env python3
"""Step 3 v2: recompute the selection with stable soft weights and seven schemes.

No ABC run, no cut re-enumeration, no install and no mapping. This step re-reads
the frozen step-2 export, the four original AIGs and the step-1 CSV, and only
recomputes the per-cut choice.

Changes against v1 (``scripts/level_weight_selection.py``):

* soft weights are shifted by the cut's minimum slack,
  ``w_l = exp(-(slack_l - m) / tau)`` with ``m = min(slack)``. The argmin is the
  same as v1's unnormalised form, but no weight underflows;
* J ties use a relative tolerance ``1e-9 * max(1, |J|)`` for the soft schemes;
  the integer objectives (uniform, hard, rootlevel, abc_default) stay exact;
* two controls are added: ``rootlevel`` (minimise the candidate root level) and
  ``abc_default`` (replay ABC's own acceptance rule, including "keep the
  original" when the cut has no accepted candidate);
* the stratified table by node slack becomes the main output.
"""
import argparse
import csv
import gzip
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

from global_timing.level_weights import original_leaf_depths, propagate_levels
from global_timing.structure import parse_aig, references

CIRCUITS = [
    ('c1355', '2602b10f131a6ba5996521036ba130aab128a4aa15e3ab49ce44f2f02047f0b0'),
    ('c5315', 'd1c0cea55baf5a407495d82eb4385a9ddea8528cde21a762f46bb0774fc93afd'),
    ('intra_pred_PE', '889b40b408ff0e3333a81f73e99887ee379fb60c6e8c1b5f23d1056ce7c088a4'),
    ('cabac_ucontext', '77fdf62aa46fa9a5bdb67a49ede156bff75bc8ea121faff67b74fa89a2a78707'),
]
SCHEMES = ('uniform', 'hard', 'tau0.5', 'tau1', 'tau2', 'rootlevel', 'abc_default')
WEIGHTED = ('hard', 'tau0.5', 'tau1', 'tau2')
SOFT = ('tau0.5', 'tau1', 'tau2')
TAUS = {'tau0.5': 0.5, 'tau1': 1.0, 'tau2': 2.0}
PAIRS = [(a, b) for i, a in enumerate(SCHEMES) for b in SCHEMES[i + 1:]]
BINS = [('0', 0, 0), ('1-2', 1, 2), ('3-5', 3, 5), ('>5', 6, None)]
TOL = 1e-9
EXAMPLE_CAP = 20
CSV_COLUMNS = ['circuit', 'decision_id', 'line_no', 'node_abc', 'node_aiger', 'is_region_root',
               'node_level', 'node_required', 'node_slack', 'slack_bin', 'n_candidates',
               'n_nodes_saved', 'abc_default_keep']
for _scheme in SCHEMES:
    CSV_COLUMNS += [f'{_scheme}_subgraph', f'{_scheme}_root_level', f'{_scheme}_nodes_added',
                    f'{_scheme}_delta', f'{_scheme}_accepted', f'{_scheme}_J']


def file_digest(path):
    import hashlib
    sha = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            sha.update(chunk)
    return sha.hexdigest()


def need(condition, message):
    if not condition:
        raise SystemExit('ASSERTION FAILED: ' + message)


def slack_bin(slack):
    for name, low, high in BINS:
        if slack >= low and (high is None or slack <= high):
            return name
    raise SystemExit(f'internal: slack {slack} fell outside the bins')


def shifted_weights(scheme, slacks):
    """w_l = exp(-(slack_l - min(slack)) / tau); the tightest leaf keeps weight 1."""
    if scheme == 'uniform':
        return [1.0] * len(slacks)
    if scheme == 'hard':
        tightest = min(slacks)
        return [1.0 if slack == tightest else 0.0 for slack in slacks]
    tau = TAUS[scheme]
    lowest = min(slacks)
    return [math.exp(-(slack - lowest) / tau) for slack in slacks]


def v1_weights(scheme, slacks):
    """The v1 form, unnormalised, only used to enumerate the degenerate cuts."""
    if scheme == 'uniform':
        return [1.0] * len(slacks)
    if scheme == 'hard':
        tightest = min(slacks)
        return [1.0 if slack == tightest else 0.0 for slack in slacks]
    tau = TAUS[scheme]
    return [math.exp(-slack / tau) for slack in slacks]


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
    """Smallest value, then nodes_added (self -> 0), then the subgraph number."""
    best = min(values)
    limit = tolerance * max(1.0, abs(best))
    tied = [index for index, value in enumerate(values) if abs(value - best) <= limit]
    choice = min(tied, key=lambda index: (rows[index]['added'], rows[index]['subgraph']))
    return choice, len(tied)


def quantile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    position = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[position]


def empty_bin():
    return dict(nonempty=0, lower=0, equal=0, higher=0, over=0,
                delta=Counter(), delta_positive=0, lowered_acc1=0, lowered_acc0=0)


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


def load_v1(folder, name):
    """v1 per-decision CSV: decision_id -> per-scheme subgraph."""
    path = folder / f'{name}.decisions.csv'
    if not path.exists():
        path = folder / f'{name}.decisions.csv.gz'
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as stream:
        return {row['decision_id']: row for row in csv.DictReader(stream)}


def run_circuit(name, aig_path, export_path, stats, step1_rows, v1_rows, out_dir):
    aig = parse_aig(aig_path)
    levels = propagate_levels(aig)
    fanouts = references(aig)
    offset = stats['identifier_mapping']['po_count']
    need(offset == aig.po_count, f'{name}: offset {offset} differs from AIG PO count {aig.po_count}')
    need(stats['identifier_mapping']['and_nodes'] == aig.and_count,
         f'{name}: recorded AND count differs from the AIG')
    need(stats['identifier_mapping']['verified'] == aig.and_count,
         f'{name}: identifier mapping was not verified on every AND')

    def to_aig(node_id):
        return node_id if node_id <= aig.pi_count else node_id - offset

    roots = {row['root']: row for row in step1_rows}
    for root_id, row in roots.items():
        need(levels.level[root_id] == row['level'], f'{name}: step-1 level of {root_id} disagrees')
        required = levels.required[root_id]
        need(required is not None, f'{name}: step-1 root {root_id} has no required level')
        need(required == row['req'], f'{name}: step-1 required of {root_id} disagrees')
        need(required - levels.level[root_id] == row['slack'], f'{name}: step-1 slack of {root_id} disagrees')

    counters = stats['counters']
    base = Path(stats['_folder'])
    nodes_rows = {}
    with (base / f'{name}.nodes.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            nodes_rows[row['node']] = row
    high_fanout = sorted(node for node, row in nodes_rows.items() if row['reason'] == 'high_fanout')
    need(len(high_fanout) == counters['nodes_high_fanout'],
         f'{name}: high-fanout node count disagrees with counters')
    excluded = []
    for node in high_fanout:
        row = nodes_rows[node]
        need(row['cuts_exported'] == 0 and row['candidates'] == 0,
             f'{name}: high-fanout node {node} unexpectedly carries cuts')
        aiger_id = to_aig(node)
        required = levels.required[aiger_id]
        need(required is not None, f'{name}: high-fanout node {node} has no required level')
        excluded.append(dict(circuit=name, abc_id=node, aiger_id=aiger_id, reason=row['reason'],
                             level=levels.level[aiger_id], required=required,
                             slack=required - levels.level[aiger_id]))

    totals = dict(decisions=0, empty_decisions=0, candidates=0, keep_decisions=0,
                  excluded_high_fanout_nodes=len(excluded))
    assertions = dict(rebuild_checked=0, candidate_order_checked=0, mffc_checked=0,
                      self_known=0, self_other=0, subgraph_order_checked=0)
    bins = {label: {scheme: empty_bin() for scheme in SCHEMES} for label, _, _ in BINS}
    scheme_totals = {scheme: dict(lower=0, equal=0, higher=0, over=0, nonempty=0,
                                  delta=Counter(), delta_positive=0, lowered_acc1=0,
                                  lowered_acc0=0, ties=0) for scheme in SCHEMES}
    pairwise = {f'{a}|{b}': 0 for a, b in PAIRS}
    degenerate = {scheme: 0 for scheme in SOFT}
    agreement = {scheme: dict(same=0, differ=0, tie=0, material=0, examples=[]) for scheme in SOFT}
    slack_zero_area = {scheme: dict(lowered=0, area_up=0) for scheme in WEIGHTED}
    v1_found = 0
    compared = 0
    per_node = Counter()

    csv_path = out_dir / f'{name}.decisions.csv'
    with export_path.open() as stream, csv_path.open('w', newline='') as sink:
        writer = csv.writer(sink)
        writer.writerow(CSV_COLUMNS)
        for line_no, line in enumerate(stream, 1):
            record = json.loads(line)
            totals['decisions'] += 1
            node_abc = record['node']
            per_node[node_abc] += 1
            decision_id = f'{node_abc}#{per_node[node_abc] - 1}'
            need(node_abc in nodes_rows, f'{name}: exported node {node_abc} missing from nodes.jsonl')
            need(nodes_rows[node_abc]['reason'] == 'evaluated',
                 f'{name}: exported node {node_abc} is not evaluated')
            leaves_abc = [leaf['id'] for leaf in record['leaves']]
            need(len(leaves_abc) == 4, f'{name}: {decision_id} has {len(leaves_abc)} leaves')
            need(len(set(leaves_abc)) == 4, f'{name}: {decision_id} repeats a leaf')
            root = to_aig(node_abc)
            leaves = [to_aig(leaf) for leaf in leaves_abc]
            need(root in aig.fanin_literals, f'{name}: node {node_abc} is not an AND')
            need(root not in leaves, f'{name}: {decision_id} has the node as a leaf')
            required = levels.required[root]
            need(required is not None, f'{name}: node {node_abc} has no required level')

            depths, _outside = original_leaf_depths(aig, root, leaves, levels.level)
            need(all(depth is not None for depth in depths),
                 f'{name}: {decision_id} has a leaf outside the cut boundary')
            rebuilt = max(levels.level[leaf] + depth for leaf, depth in zip(leaves, depths))
            need(rebuilt == levels.level[root],
                 f'{name}: {decision_id} rebuilds {rebuilt}, original level {levels.level[root]}')
            assertions['rebuild_checked'] += 1
            slacks = [required - (levels.level[leaf] + depth) for leaf, depth in zip(leaves, depths)]
            need(all(slack >= 0 for slack in slacks), f'{name}: {decision_id} negative path slack')
            node_level = levels.level[root]
            node_slack = required - node_level
            label = slack_bin(node_slack)

            got_mffc = bounded_mffc_size(root, leaves, aig, fanouts)
            need(got_mffc == record['n_nodes_saved'],
                 f'{name}: {decision_id} bounded MFFC {got_mffc} != recorded {record["n_nodes_saved"]}')
            assertions['mffc_checked'] += 1

            candidates = record['candidates']
            totals['candidates'] += len(candidates)
            subgraphs = [candidate['subgraph'] for candidate in candidates]
            need(sorted(subgraphs) == list(range(len(candidates))),
                 f'{name}: {decision_id} subgraph ids are not 0..n-1 in evaluation order')
            assertions['subgraph_order_checked'] += 1
            if not candidates:
                totals['empty_decisions'] += 1
                writer.writerow([name, decision_id, line_no, node_abc, root, int(root in roots),
                                 node_level, required, node_slack, label, 0,
                                 record['n_nodes_saved'], 0]
                                + [''] * (len(CSV_COLUMNS) - 13))
                continue

            rows = []
            v1_values = {scheme: [] for scheme in SOFT}
            for candidate in candidates:
                leaf_depths = candidate['leaf_depths']
                need(len(leaf_depths) == 4, f'{name}: {decision_id} candidate depth length')
                order_level = max((levels.level[leaf] + depth
                                   for leaf, depth in zip(leaves, leaf_depths) if depth >= 0),
                                  default=0)
                need(order_level == candidate['root_level'],
                     f'{name}: {decision_id} subgraph {candidate["subgraph"]} root_level mismatch')
                assertions['candidate_order_checked'] += 1
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
                            raise SystemExit(
                                f'ASSERTION FAILED: {name}: {decision_id} subgraph '
                                f'{candidate["subgraph"]} self depth drift {drift} is not the known +1')
                values = {scheme: sum(w * d for w, d in zip(shifted_weights(scheme, slacks), leaf_depths)
                                      if d >= 0) for scheme in SCHEMES[:5]}
                for scheme in SOFT:
                    v1_values[scheme].append(
                        sum(w * d for w, d in zip(v1_weights(scheme, slacks), leaf_depths) if d >= 0))
                rows.append(dict(subgraph=candidate['subgraph'],
                                 added=0 if candidate['self'] else candidate['nodes_added'],
                                 root_level=candidate['root_level'],
                                 nodes_added=candidate['nodes_added'],
                                 self=candidate['self'],
                                 accepted=candidate['accepted_by_default'],
                                 values=values))

            chosen = {}
            ties = {}
            for scheme in SCHEMES[:5]:
                tolerance = TOL if scheme in SOFT else 0.0
                index, tie_count = pick([row['values'][scheme] for row in rows], rows, tolerance)
                chosen[scheme] = index
                ties[scheme] = tie_count
            index, tie_count = pick([row['root_level'] for row in rows], rows, 0.0)
            chosen['rootlevel'] = index
            ties['rootlevel'] = tie_count
            accepted_rows = [(index, row) for index, row in enumerate(rows)
                             if row['accepted'] and not row['self']]
            if accepted_rows:
                gains = [record['n_nodes_saved'] - row['nodes_added'] for _, row in accepted_rows]
                best_gain = max(gains)
                tied = [accepted_rows[k][0] for k, gain in enumerate(gains) if gain == best_gain]
                chosen['abc_default'] = min(tied)
                ties['abc_default'] = len(tied)
                keep = 0
            else:
                chosen['abc_default'] = None
                ties['abc_default'] = 0
                keep = 1
                totals['keep_decisions'] += 1

            for scheme in SOFT:
                v1_best = min(v1_values[scheme])
                if v1_best < TOL:
                    degenerate[scheme] += 1
            if min(slacks) <= 5:
                compared += 1
                need(decision_id in v1_rows, f'{name}: {decision_id} missing from the v1 CSV')
                v1_found += 1
                for scheme in SOFT:
                    v1_choice = int(v1_rows[decision_id][f'{scheme}_subgraph'])
                    v2_index = chosen[scheme]
                    if v1_choice == rows[v2_index]['subgraph']:
                        agreement[scheme]['same'] += 1
                    else:
                        agreement[scheme]['differ'] += 1
                        left = v1_values[scheme][v1_choice]
                        right = v1_values[scheme][v2_index]
                        gap = abs(left - right) / max(1.0, abs(left), abs(right))
                        kind = 'tie' if gap <= TOL else 'material'
                        agreement[scheme][kind] += 1
                        if len(agreement[scheme]['examples']) < EXAMPLE_CAP:
                            agreement[scheme]['examples'].append(dict(
                                decision=decision_id, kind=kind,
                                v1_subgraph=v1_choice,
                                v2_subgraph=rows[v2_index]['subgraph'],
                                slack=slacks, min_slack=min(slacks),
                                v1_J=left, v1_J_other=right, v2_J=rows[v2_index]['values'][scheme],
                                relative_gap=gap))

            selection = {}
            for scheme in SCHEMES:
                if chosen[scheme] is None:
                    selection[scheme] = dict(subgraph='', root_level=node_level, nodes_added=0,
                                             delta=0, accepted='', J='', level=node_level,
                                             self=None)
                    continue
                row = rows[chosen[scheme]]
                selection[scheme] = dict(
                    subgraph=row['subgraph'], root_level=row['root_level'],
                    nodes_added=row['nodes_added'],
                    delta=row['nodes_added'] - record['n_nodes_saved'],
                    accepted=int(row['accepted']), J=round(row['values'][scheme], 12)
                    if scheme in SCHEMES[:5] else '',
                    level=row['root_level'], self=row['self'])

            for scheme in SCHEMES:
                picked = selection[scheme]
                cell = bins[label][scheme]
                cell['nonempty'] += 1
                if picked['level'] < node_level:
                    cell['lower'] += 1
                elif picked['level'] == node_level:
                    cell['equal'] += 1
                else:
                    cell['higher'] += 1
                    if picked['level'] - node_level > node_slack:
                        cell['over'] += 1
                cell['delta'][picked['delta']] += 1
                if picked['delta'] > 0:
                    cell['delta_positive'] += 1
                if picked['level'] < node_level:
                    if picked['accepted'] == 1:
                        cell['lowered_acc1'] += 1
                    elif picked['accepted'] == 0:
                        cell['lowered_acc0'] += 1
                total = scheme_totals[scheme]
                total['nonempty'] += 1
                if picked['level'] < node_level:
                    total['lower'] += 1
                elif picked['level'] == node_level:
                    total['equal'] += 1
                else:
                    total['higher'] += 1
                    if picked['level'] - node_level > node_slack:
                        total['over'] += 1
                total['delta'][picked['delta']] += 1
                if picked['delta'] > 0:
                    total['delta_positive'] += 1
                if picked['level'] < node_level:
                    if picked['accepted'] == 1:
                        total['lowered_acc1'] += 1
                    elif picked['accepted'] == 0:
                        total['lowered_acc0'] += 1
                if ties[scheme] > 1:
                    total['ties'] += 1
                if label == '0' and scheme in WEIGHTED and picked['level'] < node_level:
                    slack_zero_area[scheme]['lowered'] += 1
                    if picked['delta'] > 0:
                        slack_zero_area[scheme]['area_up'] += 1

            for a, b in PAIRS:
                if selection[a]['subgraph'] != selection[b]['subgraph']:
                    pairwise[f'{a}|{b}'] += 1

            writer.writerow([name, decision_id, line_no, node_abc, root, int(root in roots),
                             node_level, required, node_slack, label, len(candidates),
                             record['n_nodes_saved'], keep]
                            + [value for scheme in SCHEMES
                               for value in (selection[scheme]['subgraph'],
                                             selection[scheme]['root_level'],
                                             selection[scheme]['nodes_added'],
                                             selection[scheme]['delta'],
                                             selection[scheme]['accepted'],
                                             selection[scheme]['J'])])

    need(assertions['rebuild_checked'] == totals['decisions'],
         f'{name}: only {assertions["rebuild_checked"]} of {totals["decisions"]} cuts were rebuilt')
    need(assertions['mffc_checked'] + 0 == totals['decisions'],
         f'{name}: bounded MFFC checked on {assertions["mffc_checked"]} of {totals["decisions"]} cuts')
    nonempty = totals['decisions'] - totals['empty_decisions']
    need(sum(row['nonempty'] for row in scheme_totals.values()) == nonempty * len(SCHEMES),
         f'{name}: scheme accounting does not cover every non-empty decision')
    need(v1_found == compared, f'{name}: compared {v1_found} decisions with v1 but expected {compared}')

    return dict(
        circuit=name,
        counts=dict(decisions=totals['decisions'], candidates=totals['candidates'],
                    empty_decisions=totals['empty_decisions'],
                    keep_decisions=totals['keep_decisions'],
                    excluded_high_fanout_nodes=totals['excluded_high_fanout_nodes'],
                    nonempty=nonempty),
        assertions=assertions,
        excluded_nodes=excluded,
        bins=_bin_summary(bins),
        schemes=_scheme_summary(scheme_totals),
        pairwise=pairwise,
        degenerate=degenerate,
        agreement={scheme: dict(same=agreement[scheme]['same'],
                                differ=agreement[scheme]['differ'],
                                tie=agreement[scheme]['tie'],
                                material=agreement[scheme]['material'],
                                examples=agreement[scheme]['examples']) for scheme in SOFT},
        slack_zero_area={scheme: dict(lowered=slack_zero_area[scheme]['lowered'],
                                      area_up=slack_zero_area[scheme]['area_up'])
                         for scheme in WEIGHTED},
        csv=dict(path=str(csv_path), bytes=csv_path.stat().st_size),
    )


def _bin_summary(bins):
    out = {}
    for label, _, _ in BINS:
        out[label] = {}
        for scheme in SCHEMES:
            cell = bins[label][scheme]
            denominator = cell['nonempty'] or 1
            out[label][scheme] = dict(
                nonempty=cell['nonempty'], lower=cell['lower'], equal=cell['equal'],
                higher=cell['higher'], over=cell['over'],
                lower_fraction=cell['lower'] / denominator,
                equal_fraction=cell['equal'] / denominator,
                higher_fraction=cell['higher'] / denominator,
                over_fraction=cell['over'] / denominator,
                delta_median=median_of(cell['delta']),
                delta_hist={str(k): v for k, v in cell['delta'].items()},
                delta_positive=cell['delta_positive'],
                delta_positive_fraction=cell['delta_positive'] / denominator,
                lowered_acc1=cell['lowered_acc1'], lowered_acc0=cell['lowered_acc0'])
    return out


def _scheme_summary(totals):
    out = {}
    for scheme, row in totals.items():
        denominator = row['nonempty'] or 1
        out[scheme] = dict(
            nonempty=row['nonempty'], lower=row['lower'], equal=row['equal'],
            higher=row['higher'], over=row['over'],
            lower_fraction=row['lower'] / denominator,
            equal_fraction=row['equal'] / denominator,
            higher_fraction=row['higher'] / denominator,
            over_fraction=row['over'] / denominator,
            delta_median=median_of(row['delta']),
            delta_hist={str(k): v for k, v in row['delta'].items()},
            delta_positive=row['delta_positive'],
            delta_positive_fraction=row['delta_positive'] / denominator,
            lowered_acc1=row['lowered_acc1'], lowered_acc0=row['lowered_acc0'],
            tie_fraction=row['ties'] / denominator)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--out', default=None, type=Path)
    parser.add_argument('--circuits', default=None,
                        help='comma-separated subset for a pre-flight; default is all four')
    args = parser.parse_args()
    wanted = [item.strip() for item in (args.circuits or '').split(',') if item.strip()]
    known = {name for name, _ in CIRCUITS}
    if wanted and set(wanted) - known:
        raise SystemExit(f'unknown circuit(s): {sorted(set(wanted) - known)}')
    run = args.run.resolve()
    if not os.getenv('SLURM_JOB_ID'):
        raise SystemExit('run this only inside the committed /data Slurm job')
    if not run.is_relative_to('/data/wushangge'):
        raise SystemExit('run this only under /data/wushangge')
    source = run / 'src'
    out_dir = (args.out or run / 'output').resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    step2 = source / 'analysis/level_weighted/step2'
    step1 = source / 'analysis/level_weighted/step1'
    v1_dir = source / 'analysis/level_weighted/step3'
    stats_all = json.loads((step2 / 'stats.json').read_text())

    inputs = {}
    circuits = {}
    for name, aig_sha in CIRCUITS:
        if wanted and name not in wanted:
            continue
        circuit_stats = stats_all['circuits'][name]
        need(circuit_stats['input_sha256'] == aig_sha,
             f'{name}: stats.json digest differs from the frozen cohort')
        export_path = run / 'input/exports' / name / 'cuts.jsonl'
        aig_path = run / 'input/circuits' / f'{name}.aig'
        export_sha = file_digest(export_path)
        need(export_sha == circuit_stats['full_export']['sha256'],
             f'{name}: export digest differs from stats.json')
        need(file_digest(aig_path) == aig_sha, f'{name}: AIG digest differs from step 1')
        circuit_stats['_folder'] = str(step2)
        step1_rows = [{key: int(value) for key, value in row.items()}
                      for row in csv.DictReader((step1 / f'{name}.csv').open())]
        v1_path = v1_dir / f'{name}.decisions.csv'
        if not v1_path.exists():
            v1_path = v1_dir / f'{name}.decisions.csv.gz'
        need(v1_path.exists(), f'{name}: v1 decisions CSV is missing')
        inputs[name] = dict(aig_sha256=aig_sha, export_sha256=export_sha,
                            export_cuts=circuit_stats['full_export']['lines'],
                            step1_csv_sha256=file_digest(step1 / f'{name}.csv'),
                            v1_csv_sha256=file_digest(v1_path), v1_csv=str(v1_path))
        circuits[name] = run_circuit(name, aig_path, export_path, circuit_stats,
                                     step1_rows, load_v1(v1_dir, name), out_dir)
        print(f'{name}: decisions={circuits[name]["counts"]["decisions"]} '
              f'keep={circuits[name]["counts"]["keep_decisions"]} '
              f'self_known={circuits[name]["assertions"]["self_known"]} '
              f'degen={circuits[name]["degenerate"]}', flush=True)

    totals = _totals(circuits)
    summary = dict(
        step='3_v2',
        purpose='recompute the per-cut selection with shifted soft weights, two controls '
                'and a slack-stratified table',
        coverage=dict(
            cuts='only the 4-leaf cuts that ABC rewrite evaluates',
            selection='minimum objective inside each cut only; no cross-cut comparison and no '
                      'per-node final candidate',
            graph='original AIG; levels, required and leaf depths recomputed in this run',
            global_note='a lower local root_level is not a global level reduction; proving that '
                        'needs joint installation, which this step does not run'),
        schemes=list(SCHEMES),
        soft_weights='w_l = exp(-(slack_l - min(slack)) / tau), tau in {0.5,1,2}; not normalised',
        tie_rule='|a-b| <= 1e-9 * max(1,|a|) for the soft schemes, exact for the integer ones',
        inputs=inputs,
        circuits=circuits,
        totals=totals,
        slurm_job_id=os.getenv('SLURM_JOB_ID'),
        source_commit=json.loads((run / 'run.json').read_text())['source_commit'],
    )
    (out_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print('totals:', json.dumps(totals['counts'], ensure_ascii=False), flush=True)
    for label in totals['bins']:
        print('bin', label, {s: totals['bins'][label][s]['nonempty'] for s in SCHEMES}, flush=True)
    return 0


def _totals(circuits):
    counts = Counter()
    for row in circuits.values():
        counts.update(row['counts'])
    assertions = Counter()
    for row in circuits.values():
        assertions.update(row['assertions'])
    bins = {}
    for label, _, _ in BINS:
        bins[label] = {}
        for scheme in SCHEMES:
            cells = [row['bins'][label][scheme] for row in circuits.values()]
            merged = empty_bin()
            for cell in cells:
                merged['nonempty'] += cell['nonempty']
                merged['lower'] += cell['lower']
                merged['equal'] += cell['equal']
                merged['higher'] += cell['higher']
                merged['over'] += cell['over']
                merged['delta_positive'] += cell['delta_positive']
                merged['lowered_acc1'] += cell['lowered_acc1']
                merged['lowered_acc0'] += cell['lowered_acc0']
                merged['delta'].update({int(k): v for k, v in cell['delta_hist'].items()})
            denominator = merged['nonempty'] or 1
            bins[label][scheme] = dict(
                nonempty=merged['nonempty'], lower=merged['lower'], equal=merged['equal'],
                higher=merged['higher'], over=merged['over'],
                lower_fraction=merged['lower'] / denominator,
                equal_fraction=merged['equal'] / denominator,
                higher_fraction=merged['higher'] / denominator,
                over_fraction=merged['over'] / denominator,
                delta_median=median_of(merged['delta']), delta_positive=merged['delta_positive'],
                delta_positive_fraction=merged['delta_positive'] / denominator,
                lowered_acc1=merged['lowered_acc1'], lowered_acc0=merged['lowered_acc0'])
    schemes = {}
    for scheme in SCHEMES:
        rows = [row['schemes'][scheme] for row in circuits.values()]
        nonempty = sum(row['nonempty'] for row in rows)
        merged_delta = Counter()
        for row in rows:
            merged_delta.update({int(k): v for k, v in row['delta_hist'].items()})
        schemes[scheme] = dict(
            nonempty=nonempty,
            lower=sum(row['lower'] for row in rows),
            equal=sum(row['equal'] for row in rows),
            higher=sum(row['higher'] for row in rows),
            over=sum(row['over'] for row in rows),
            lower_fraction=sum(row['lower'] for row in rows) / (nonempty or 1),
            equal_fraction=sum(row['equal'] for row in rows) / (nonempty or 1),
            higher_fraction=sum(row['higher'] for row in rows) / (nonempty or 1),
            over_fraction=sum(row['over'] for row in rows) / (nonempty or 1),
            delta_median=median_of(merged_delta),
            delta_positive=sum(row['delta_positive'] for row in rows),
            delta_positive_fraction=sum(row['delta_positive'] for row in rows) / (nonempty or 1),
            lowered_acc1=sum(row['lowered_acc1'] for row in rows),
            lowered_acc0=sum(row['lowered_acc0'] for row in rows),
            tie_fraction=sum(row['tie_fraction'] for row in rows) / len(rows))
    pairwise = Counter()
    for row in circuits.values():
        pairwise.update(row['pairwise'])
    degenerate = {scheme: sum(row['degenerate'][scheme] for row in circuits.values())
                  for scheme in SOFT}
    agreement = {}
    for scheme in SOFT:
        same = sum(row['agreement'][scheme]['same'] for row in circuits.values())
        differ = sum(row['agreement'][scheme]['differ'] for row in circuits.values())
        tie = sum(row['agreement'][scheme]['tie'] for row in circuits.values())
        material = sum(row['agreement'][scheme]['material'] for row in circuits.values())
        examples = []
        for name, row in circuits.items():
            examples += [dict(circuit=name, **item) for item in row['agreement'][scheme]['examples']]
        examples.sort(key=lambda item: (item['kind'] != 'material', -item['relative_gap']))
        agreement[scheme] = dict(same=same, differ=differ, tie=tie, material=material,
                                 rate=same / (same + differ) if same + differ else None,
                                 material_rate=(same + tie) / (same + differ)
                                 if same + differ else None,
                                 examples=examples[:EXAMPLE_CAP])
    slack_zero_area = {scheme: dict(
        lowered=sum(row['slack_zero_area'][scheme]['lowered'] for row in circuits.values()),
        area_up=sum(row['slack_zero_area'][scheme]['area_up'] for row in circuits.values()))
        for scheme in WEIGHTED}
    for scheme in WEIGHTED:
        cell = slack_zero_area[scheme]
        cell['fraction'] = cell['area_up'] / cell['lowered'] if cell['lowered'] else None
    return dict(counts=dict(counts), assertions=dict(assertions), bins=bins, schemes=schemes,
                pairwise=dict(pairwise), degenerate=degenerate, agreement=agreement,
                slack_zero_area=slack_zero_area,
                rootlevel_vs_hard=dict(
                    rootlevel_lower=schemes['rootlevel']['lower'],
                    hard_lower=schemes['hard']['lower'],
                    difference=schemes['rootlevel']['lower'] - schemes['hard']['lower']))


if __name__ == '__main__':
    sys.exit(main())
