#!/usr/bin/env python3
"""Step 3: compare five leaf-weighting schemes on the exported ABC rewrite candidates.

Pure CPU. This step reads the step-2 export, the four original AIGs and the
step-1 level/slack CSV. It never installs a candidate, never re-enumerates cuts,
never maps and never runs the joint ABC replacement.

Definitions used here (see analysis/level_weighted/step3/README.md):

* one decision = one exported (node, cut) pair;
* ``d_orig(l)`` is the reverse longest path from the node to leaf ``l`` on the
  original AIG, stopping at the cut boundary, computed by the same
  ``global_timing.level_weights.original_leaf_depths`` used in step 1;
* ``slack_l = required[node] - (level[l] + d_orig(l))`` -- the node's required
  level, not the leaf's own slack;
* the objective is ``J(c) = sum_l w_l * leaf_depths(c)[l]`` with ``-1`` depths
  contributing nothing, and the original structure uses ``d_orig`` instead of
  ``leaf_depths``.

Every condition that must hold is asserted in code and aborts the run. Only the
two conditions the task explicitly asks to count (self-candidate depth drift and
bounded-MFFC size drift) collect examples and continue.
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
SCHEMES = ('uniform', 'hard', 'tau0.5', 'tau1', 'tau2')
TAUS = {'tau0.5': 0.5, 'tau1': 1.0, 'tau2': 2.0}
WEIGHTED = ('hard', 'tau0.5', 'tau1', 'tau2')
PAIRS = [(a, b) for i, a in enumerate(SCHEMES) for b in SCHEMES[i + 1:]]
EXAMPLE_TARGET = 10
EXAMPLE_DETAIL = 20
CSV_COLUMNS = [
    'circuit', 'decision_id', 'line_no', 'node_abc', 'node_aiger', 'is_region_root',
    'node_level', 'node_required', 'node_slack', 'n_candidates', 'n_nodes_saved',
    'tie_uniform', 'tie_hard', 'tie_tau0.5', 'tie_tau1', 'tie_tau2',
]
for _scheme in SCHEMES:
    CSV_COLUMNS += [f'{_scheme}_subgraph', f'{_scheme}_J', f'{_scheme}_J0',
                    f'{_scheme}_root_level', f'{_scheme}_nodes_added',
                    f'{_scheme}_self', f'{_scheme}_accepted']


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


def scheme_weights(scheme, slacks):
    if scheme == 'uniform':
        return [1.0] * len(slacks)
    if scheme == 'hard':
        tightest = min(slacks)
        return [1.0 if slack == tightest else 0.0 for slack in slacks]
    tau = TAUS[scheme]
    return [math.exp(-slack / tau) for slack in slacks]


def bounded_mffc_size(root, leaves, aig, fanouts):
    """MFFC of the node with the cut leaves held as an extra fanout.

    ABC calls ``Abc_NodeMffcLabelAig`` after incrementing each cut leaf's fanout
    counter by one (rwrEva.c, cut loop). A leaf therefore never reaches zero
    references: it is excluded from the count and its fanins are not visited.
    The returned value is exactly what the export records as ``n_nodes_saved``.
    """
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


def quantiles(values):
    ordered = sorted(values)
    if not ordered:
        return {}
    def at(fraction):
        position = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
        return ordered[position]
    return dict(min=ordered[0], p10=at(0.10), p25=at(0.25), median=at(0.50),
                p75=at(0.75), p90=at(0.90), max=ordered[-1])


def empty_scheme_stats():
    return dict(
        J_smaller=0, J_equal=0, J_greater=0,
        root_lower=0, root_equal=0, root_higher=0,
        acc_one=0, acc_zero=0, lower_and_dropped=0, lower_total=0,
        delta=Counter(), delta_list=[],
        ties=0, single_choice=0,
    )


def pairs_key(a, b):
    return f'{a}|{b}'


def run_circuit(name, aig_path, export_path, stats, step1_rows, out_dir, examples):
    aig = parse_aig(aig_path)
    levels = propagate_levels(aig)
    fanouts = references(aig)
    offset = stats['identifier_mapping']['po_count']
    need(offset == aig.po_count,
         f'{name}: abc/aiger offset {offset} differs from AIG PO count {aig.po_count}')
    need(stats['identifier_mapping']['and_nodes'] == aig.and_count,
         f'{name}: recorded AND count differs from AIG')
    need(stats['identifier_mapping']['verified'] == aig.and_count,
         f'{name}: identifier mapping was not verified on every AND')

    def to_aig(node_id):
        return node_id if node_id <= aig.pi_count else node_id - offset

    roots = {row['root']: row for row in step1_rows}
    for root_id, row in roots.items():
        need(levels.level[root_id] == row['level'],
             f'{name}: step-1 level of root {root_id} is {row["level"]}, recomputed '
             f'{levels.level[root_id]}')
        required = levels.required[root_id]
        need(required is not None, f'{name}: step-1 root {root_id} has no required level')
        need(required == row['req'],
             f'{name}: step-1 required of root {root_id} is {row["req"]}, recomputed {required}')
        need(required - levels.level[root_id] == row['slack'],
             f'{name}: step-1 slack of root {root_id} disagrees with the recomputation')

    counters = stats['counters']
    nodes_rows = {}
    with (Path(stats['_folder']) / f'{name}.nodes.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            nodes_rows[row['node']] = row
    high_fanout = sorted(node for node, row in nodes_rows.items() if row['reason'] == 'high_fanout')
    need(len(high_fanout) == counters['nodes_high_fanout'],
         f'{name}: {len(high_fanout)} high-fanout nodes in nodes.jsonl but '
         f'{counters["nodes_high_fanout"]} recorded')
    need(sum(1 for row in nodes_rows.values() if row['reason'] == 'evaluated')
         == counters['nodes_evaluated'],
         f'{name}: evaluated node count disagrees with the export')
    excluded = []
    for node in high_fanout:
        row = nodes_rows[node]
        need(row['cuts_exported'] == 0 and row['candidates'] == 0,
             f'{name}: high-fanout node {node} unexpectedly carries cuts')
        aiger_id = to_aig(node)
        required = levels.required[aiger_id]
        need(required is not None,
             f'{name}: high-fanout node {node} has no required level on the original AIG')
        excluded.append(dict(circuit=name, abc_id=node, aiger_id=aiger_id,
                             reason=row['reason'], level=levels.level[aiger_id],
                             required=required, slack=required - levels.level[aiger_id],
                             abc_required=row['required'], cuts_exported=row['cuts_exported']))

    totals = dict(decisions=0, empty_decisions=0, candidates=0,
                  region_root_decisions=0, internal_decisions=0,
                  zero_slack_decisions=0, positive_slack_decisions=0)
    schemes = {scheme: empty_scheme_stats() for scheme in SCHEMES}
    pairwise = {pairs_key(a, b): 0 for a, b in PAIRS}
    strata = {
        'region_root': dict(decisions=0, nonempty=0, ties=0, diff=Counter()),
        'internal': dict(decisions=0, nonempty=0, ties=0, diff=Counter()),
        'slack_zero': dict(decisions=0, nonempty=0, ties=0, diff=Counter()),
        'slack_positive': dict(decisions=0, nonempty=0, ties=0, diff=Counter()),
    }
    per_node = Counter()
    assertion = dict(rebuild_checked=0, candidate_order_checked=0,
                     self_mismatches=0, self_examples=[],
                     mffc_mismatches=0, mffc_examples=[],
                     unreachable_required=0)

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
                 f'{name}: exported node {node_abc} is not an evaluated node')
            leaves_abc = [leaf['id'] for leaf in record['leaves']]
            need(len(leaves_abc) == 4, f'{name}: decision {decision_id} has {len(leaves_abc)} leaves')
            need(len(set(leaves_abc)) == 4, f'{name}: decision {decision_id} repeats a leaf')
            root = to_aig(node_abc)
            leaves = [to_aig(leaf) for leaf in leaves_abc]
            need(root in aig.fanin_literals, f'{name}: node {node_abc} is not an AND of the AIG')
            need(root not in leaves, f'{name}: decision {decision_id} has the node as a leaf')
            required = levels.required[root]
            if required is None:
                assertion['unreachable_required'] += 1
                raise SystemExit(f'ASSERTION FAILED: {name}: node {node_abc} has no required level')

            depths, _outside = original_leaf_depths(aig, root, leaves, levels.level)
            need(all(depth is not None for depth in depths),
                 f'{name}: decision {decision_id} has a leaf outside the cut boundary {depths}')
            rebuilt = max(levels.level[leaf] + depth for leaf, depth in zip(leaves, depths))
            need(rebuilt == levels.level[root],
                 f'{name}: decision {decision_id} rebuilds {rebuilt} but the original level is '
                 f'{levels.level[root]}')
            assertion['rebuild_checked'] += 1
            slacks = [required - (levels.level[leaf] + depth) for leaf, depth in zip(leaves, depths)]
            need(all(slack >= 0 for slack in slacks),
                 f'{name}: decision {decision_id} has a negative path slack {slacks}')

            got_mffc = bounded_mffc_size(root, leaves, aig, fanouts)
            if got_mffc != record['n_nodes_saved']:
                assertion['mffc_mismatches'] += 1
                if len(assertion['mffc_examples']) < EXAMPLE_DETAIL:
                    assertion['mffc_examples'].append(dict(
                        decision=decision_id, node_abc=node_abc, recomputed=got_mffc,
                        recorded=record['n_nodes_saved'], leaves=leaves_abc))

            node_level = levels.level[root]
            node_slack = required - node_level
            is_root = root in roots
            strata_keys = ('region_root' if is_root else 'internal',
                           'slack_zero' if node_slack == 0 else 'slack_positive')
            totals['region_root_decisions' if is_root else 'internal_decisions'] += 1
            totals['zero_slack_decisions' if node_slack == 0 else 'positive_slack_decisions'] += 1
            for key in strata_keys:
                strata[key]['decisions'] += 1

            candidates = record['candidates']
            totals['candidates'] += len(candidates)
            if not candidates:
                totals['empty_decisions'] += 1
                continue
            for key in strata_keys:
                strata[key]['nonempty'] += 1

            weights = {scheme: scheme_weights(scheme, slacks) for scheme in SCHEMES}
            base = {scheme: sum(w * d for w, d in zip(weights[scheme], depths))
                    for scheme in SCHEMES}
            rows = []
            for candidate in candidates:
                leaf_depths = candidate['leaf_depths']
                need(len(leaf_depths) == 4,
                     f'{name}: decision {decision_id} candidate {candidate["subgraph"]} has '
                     f'{len(leaf_depths)} depths')
                order_level = max((levels.level[leaf] + depth
                                   for leaf, depth in zip(leaves, leaf_depths) if depth >= 0),
                                  default=0)
                need(order_level == candidate['root_level'],
                     f'{name}: decision {decision_id} subgraph {candidate["subgraph"]} has '
                     f'root_level {candidate["root_level"]}, array order gives {order_level}')
                assertion['candidate_order_checked'] += 1
                if candidate['self']:
                    drift = [i for i in range(4)
                             if leaf_depths[i] >= 0 and leaf_depths[i] != depths[i]]
                    if drift:
                        assertion['self_mismatches'] += 1
                        if len(assertion['self_examples']) < EXAMPLE_DETAIL:
                            assertion['self_examples'].append(dict(
                                decision=decision_id, node_abc=node_abc,
                                recorded=leaf_depths, original=list(depths), positions=drift))
                values = {scheme: sum(w * d for w, d in zip(weights[scheme], leaf_depths)
                                      if d >= 0) for scheme in SCHEMES}
                rows.append(dict(candidate=candidate, values=values,
                                 added=0 if candidate['self'] else candidate['nodes_added']))

            chosen = {}
            ties = {}
            for scheme in SCHEMES:
                best = min(row['values'][scheme] for row in rows)
                ties[scheme] = sum(1 for row in rows if row['values'][scheme] == best)
                pick = min((row for row in rows if row['values'][scheme] == best),
                           key=lambda row: (row['added'], row['candidate']['subgraph']))
                chosen[scheme] = pick
                stats_row = schemes[scheme]
                value = pick['values'][scheme]
                reference = base[scheme]
                if value < reference:
                    stats_row['J_smaller'] += 1
                elif value == reference:
                    stats_row['J_equal'] += 1
                else:
                    stats_row['J_greater'] += 1
                root_level = pick['candidate']['root_level']
                if root_level < node_level:
                    stats_row['root_lower'] += 1
                    if not pick['candidate']['accepted_by_default']:
                        stats_row['lower_and_dropped'] += 1
                elif root_level == node_level:
                    stats_row['root_equal'] += 1
                else:
                    stats_row['root_higher'] += 1
                if pick['candidate']['accepted_by_default']:
                    stats_row['acc_one'] += 1
                else:
                    stats_row['acc_zero'] += 1
                if root_level < node_level:
                    stats_row['lower_total'] += 1
                delta = pick['candidate']['nodes_added'] - record['n_nodes_saved']
                stats_row['delta'][delta] += 1
                stats_row['delta_list'].append(delta)
                if ties[scheme] > 1:
                    stats_row['ties'] += 1
                else:
                    stats_row['single_choice'] += 1

            uniform_pick = chosen['uniform']['candidate']['subgraph']
            for scheme in WEIGHTED:
                if chosen[scheme]['candidate']['subgraph'] != uniform_pick:
                    for key in strata_keys:
                        strata[key]['diff'][scheme] += 1
            for key in strata_keys:
                if ties['uniform'] > 1:
                    strata[key]['ties'] += 1

            for a, b in PAIRS:
                if chosen[a]['candidate']['subgraph'] != chosen[b]['candidate']['subgraph']:
                    pairwise[pairs_key(a, b)] += 1

            for scheme in WEIGHTED:
                if (chosen[scheme]['candidate']['subgraph'] != uniform_pick
                        and len(examples[scheme]) < EXAMPLE_TARGET):
                    examples[scheme].append(dict(
                        circuit=name, decision=decision_id, node_abc=node_abc,
                        node_aiger=root, is_region_root=is_root, node_slack=node_slack,
                        uniform_subgraph=uniform_pick,
                        uniform_J=chosen['uniform']['values']['uniform'],
                        uniform_root_level=chosen['uniform']['candidate']['root_level'],
                        uniform_nodes_added=chosen['uniform']['candidate']['nodes_added'],
                        weighted_subgraph=chosen[scheme]['candidate']['subgraph'],
                        weighted_J=chosen[scheme]['values'][scheme],
                        weighted_root_level=chosen[scheme]['candidate']['root_level'],
                        weighted_nodes_added=chosen[scheme]['candidate']['nodes_added'],
                        self=chosen[scheme]['candidate']['self'],
                        accepted=chosen[scheme]['candidate']['accepted_by_default'],
                        weights={s: [round(x, 6) for x in weights[s]] for s in SCHEMES},
                        slacks=slacks, leaf_depths=chosen[scheme]['candidate']['leaf_depths']))

            writer.writerow(_csv_row(name, decision_id, line_no, node_abc, root, is_root,
                                     node_level, required, node_slack, len(candidates),
                                     record['n_nodes_saved'], ties, chosen, base))

    need(assertion['rebuild_checked'] == totals['decisions'],
         f'{name}: only {assertion["rebuild_checked"]} of {totals["decisions"]} cuts were rebuilt')
    need(sum(s['J_smaller'] + s['J_equal'] + s['J_greater'] for s in schemes.values())
         == len(SCHEMES) * (totals['decisions'] - totals['empty_decisions']),
         f'{name}: scheme accounting does not cover every non-empty decision')

    return dict(
        circuit=name,
        counts=dict(decisions=totals['decisions'], candidates=totals['candidates'],
                    empty_decisions=totals['empty_decisions'],
                    region_root_decisions=totals['region_root_decisions'],
                    internal_decisions=totals['internal_decisions'],
                    zero_slack_decisions=totals['zero_slack_decisions'],
                    positive_slack_decisions=totals['positive_slack_decisions'],
                    excluded_high_fanout_nodes=len(excluded)),
        assertions=assertion,
        excluded_nodes=excluded,
        pairwise={key: value for key, value in pairwise.items()},
        schemes=_scheme_summary(schemes, totals),
        strata=_strata_summary(strata, totals),
        examples={scheme: examples[scheme] for scheme in WEIGHTED},
        csv=dict(path=str(csv_path), bytes=csv_path.stat().st_size),
    )


def _csv_row(circuit, decision_id, line_no, node_abc, root, is_root, node_level,
             required, node_slack, n_candidates, n_nodes_saved, ties, chosen, base):
    row = [circuit, decision_id, line_no, node_abc, root, int(is_root),
           node_level, required, node_slack, n_candidates, n_nodes_saved,
           ties['uniform'], ties['hard'], ties['tau0.5'], ties['tau1'], ties['tau2']]
    for scheme in SCHEMES:
        pick = chosen[scheme]
        row += [pick['candidate']['subgraph'], round(pick['values'][scheme], 9),
                round(base[scheme], 9), pick['candidate']['root_level'],
                pick['candidate']['nodes_added'], int(pick['candidate']['self']),
                int(pick['candidate']['accepted_by_default'])]
    return row


def _scheme_summary(schemes, totals):
    decisions = totals['decisions'] - totals['empty_decisions']
    summary = {}
    for scheme, row in schemes.items():
        summary[scheme] = dict(
            J_smaller=row['J_smaller'], J_equal=row['J_equal'], J_greater=row['J_greater'],
            J_smaller_fraction=row['J_smaller'] / decisions,
            root_lower=row['root_lower'], root_equal=row['root_equal'],
            root_higher=row['root_higher'],
            root_lower_fraction=row['root_lower'] / decisions,
            accepted_one=row['acc_one'], accepted_zero=row['acc_zero'],
            accepted_zero_fraction=row['acc_zero'] / decisions,
            lower_and_dropped=row['lower_and_dropped'],
            lower_and_dropped_fraction=(row['lower_and_dropped'] / row['lower_total']
                                        if row['lower_total'] else None),
            tie_decisions=row['ties'], tie_fraction=row['ties'] / decisions,
            delta_counts=dict(negative=sum(v for k, v in row['delta'].items() if k < 0),
                              zero=row['delta'].get(0, 0),
                              positive=sum(v for k, v in row['delta'].items() if k > 0)),
            delta_quantiles=quantiles(row['delta_list']),
            delta_common=[[value, count] for value, count in
                          sorted(row['delta'].items(), key=lambda item: (-item[1], item[0]))[:5]],
        )
    return summary


def _strata_summary(strata, totals):
    summary = {}
    for key, row in strata.items():
        denominator = row['nonempty']
        summary[key] = dict(decisions=row['decisions'], nonempty=row['nonempty'],
                            ties=row['ties'],
                            tie_fraction=row['ties'] / denominator if denominator else None,
                            diff_counts={scheme: row['diff'][scheme] for scheme in WEIGHTED},
                            diff_vs_uniform={scheme: (row['diff'][scheme] / denominator
                                                      if denominator else None)
                                             for scheme in WEIGHTED})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--out', default=None, type=Path)
    parser.add_argument('--circuits', default=None,
                        help='comma-separated subset for a pre-flight; default is all four')
    args = parser.parse_args()
    wanted = [name.strip() for name in (args.circuits or '').split(',') if name.strip()]
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
    stats_all = json.loads((step2 / 'stats.json').read_text())

    inputs = {}
    tables = {}
    collected = {scheme: {} for scheme in WEIGHTED}
    for name, aig_sha in CIRCUITS:
        if wanted and name not in wanted:
            continue
        circuit_stats = stats_all['circuits'][name]
        need(circuit_stats['input_sha256'] == aig_sha,
             f'{name}: stats.json input digest differs from the frozen cohort')
        export_path = run / 'input/exports' / name / 'cuts.jsonl'
        aig_path = run / 'input/circuits' / f'{name}.aig'
        export_sha = file_digest(export_path)
        need(export_sha == circuit_stats['full_export']['sha256'],
             f'{name}: export digest {export_sha} differs from stats.json')
        need(file_digest(aig_path) == aig_sha, f'{name}: AIG digest differs from step 1')
        need(export_path.stat().st_size == circuit_stats['full_export']['bytes'],
             f'{name}: export size differs from stats.json')
        circuit_stats['_folder'] = str(step2)
        step1_rows = list(csv.DictReader((step1 / f'{name}.csv').open()))
        for row in step1_rows:
            for field in ('region', 'root', 'level', 'req', 'slack'):
                row[field] = int(row[field])
        inputs[name] = dict(aig_sha256=aig_sha, export_sha256=export_sha,
                            export_bytes=export_path.stat().st_size,
                            export_cuts=circuit_stats['full_export']['lines'],
                            step1_csv_sha256=file_digest(step1 / f'{name}.csv'))
        tables[name] = run_circuit(name, aig_path, export_path, circuit_stats,
                                   step1_rows, out_dir, {s: [] for s in WEIGHTED})
        for scheme in WEIGHTED:
            collected[scheme][name] = tables[name]['examples'][scheme]
        print(f'{name}: decisions={tables[name]["counts"]["decisions"]} '
              f'candidates={tables[name]["counts"]["candidates"]} '
              f'self_bad={tables[name]["assertions"]["self_mismatches"]} '
              f'mffc_bad={tables[name]["assertions"]["mffc_mismatches"]}', flush=True)

    totals = _totals(tables)
    examples = {scheme: [] for scheme in WEIGHTED}
    for scheme in WEIGHTED:
        pools = [collected[scheme].get(name, []) for name, _ in CIRCUITS if name in collected[scheme]]
        index = 0
        while len(examples[scheme]) < EXAMPLE_TARGET and any(index < len(pool) for pool in pools):
            for pool in pools:
                if index < len(pool) and len(examples[scheme]) < EXAMPLE_TARGET:
                    examples[scheme].append(pool[index])
            index += 1
    summary = dict(
        step=3,
        purpose='compare five leaf-weighting schemes on the step-2 ABC rewrite export',
        coverage=dict(
            cuts='only 4-leaf cuts that ABC rewrite evaluates',
            excluded_cuts='cuts with fewer than 4 leaves (rwrEva.c skips nLeaves < 4)',
            selection='minimum J inside each cut only; no comparison across cuts and no '
                      'per-node final candidate',
            graph='original AIG; levels, required and leaf depths recomputed in this run',
        ),
        schemes=list(SCHEMES),
        taus=TAUS,
        inputs=inputs,
        circuits=tables,
        totals=totals,
        examples={scheme: examples[scheme] for scheme in WEIGHTED},
        slurm_job_id=os.getenv('SLURM_JOB_ID'),
        source_commit=json.loads((run / 'run.json').read_text())['source_commit'],
    )
    (out_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')

    failed = [(name, key) for name, row in tables.items()
              for key in ('self_mismatches', 'mffc_mismatches')
              if row['assertions'][key]]
    print('totals:', json.dumps(totals['counts'], ensure_ascii=False), flush=True)
    for name, row in tables.items():
        print(f'{name} counts: {json.dumps(row["counts"], ensure_ascii=False)}', flush=True)
    if failed:
        print('counted mismatches (not fatal):', failed, flush=True)
    return 0


def _totals(tables):
    counts = Counter()
    for row in tables.values():
        counts.update(row['counts'])
    assertions = Counter()
    for row in tables.values():
        for key in ('rebuild_checked', 'candidate_order_checked', 'self_mismatches',
                    'mffc_mismatches', 'unreachable_required'):
            assertions[key] += row['assertions'][key]
    pairwise = Counter()
    for row in tables.values():
        for key, value in row['pairwise'].items():
            pairwise[key] += value
    schemes = {}
    for scheme in SCHEMES:
        schemes[scheme] = dict(
            J_smaller=sum(row['schemes'][scheme]['J_smaller'] for row in tables.values()),
            root_lower=sum(row['schemes'][scheme]['root_lower'] for row in tables.values()),
            accepted_zero=sum(row['schemes'][scheme]['accepted_zero'] for row in tables.values()),
            lower_and_dropped=sum(row['schemes'][scheme]['lower_and_dropped']
                                  for row in tables.values()),
            tie_decisions=sum(row['schemes'][scheme]['tie_decisions']
                              for row in tables.values()),
        )
    return dict(counts=dict(counts), assertions=dict(assertions),
                pairwise=dict(pairwise), schemes=schemes)


if __name__ == '__main__':
    sys.exit(main())
