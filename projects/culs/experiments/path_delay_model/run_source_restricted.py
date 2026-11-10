#!/usr/bin/env python3
"""Validate source-restricted MIO timing, then remeasure verified single rewrites.

All expensive invocations are intended to run under Slurm from a /data run root.
The 32 Gate3B paths and the old replay AIGs are read-only inputs.
"""
import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def abc(binary, script, log):
    p = subprocess.run([str(binary), '-c', script], text=True, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT)
    log.write_text(p.stdout)
    if p.returncode:
        raise RuntimeError(f'ABC exit {p.returncode}; see {log}')
    return p.stdout


def records(output, prefix):
    return [dict(re.findall(r'([a-z_]+)=([^\s]+)', line)) for line in output.splitlines()
            if line.startswith(prefix + ' ')]


def check_result(result, check):
    if result['reachable'] == '1':
        assert check['upper_bound'] == 'PASS', check
        assert check['single_source_consistent'] == 'PASS', check
        assert check['all_pi_ci_zero'] == '1', check
        assert float(result['pair_delay']) <= float(check['global_po_delay']) + 0.0002
    else:
        assert result['pair_delay'] == 'NA', result


def mapped_script(lib, aig, tail):
    return f'read "{lib}"; read "{aig}"; map; topo; stime; {tail}'


def baseline(args, out):
    catalog = read_csv(args.catalog)
    assert len(catalog) == 32
    pi_po = {(int(r['pi']) - 1, int(r['po'])) for r in catalog}
    assert (3, 11) in pi_po
    queries = '; '.join(f'print_pair_delay -v po{po:02d} pi{pi:02d}'
                        for pi, po in sorted(pi_po))
    tail = f'print_delay po11 pi03; {queries}; print_pair_delay -tv po11 pi03'
    output = abc(args.abc, mapped_script(args.lib, args.aig, tail), out / 'baseline.log')
    rr = records(output, 'PAIR_RESULT')
    cc = records(output, 'PAIR_CHECK')
    assert len(rr) == len(pi_po) + 1 and len(cc) == len(rr), (len(rr), len(cc))
    for r, c in zip(rr, cc):
        assert r['source_pi'] == c['source_pi'] and r['target_po'] == c['target_po']
        check_result(r, c)
    indexed = {(r['source_pi'], r['target_po']): r for r in rr}
    target = indexed['pi03', 'po11']
    assert target['reachable'] == '1'
    trace = records(output, 'PAIR_TRACE')
    assert trace[0]['kind'] == 'pi' and trace[0]['name'] == 'pi03'
    assert trace[-1]['kind'] == 'po' and trace[-1]['name'] == 'po11'
    assert len(trace) > 2 and 'PAIR_TRACE_ERROR' not in output
    prev = trace[0]
    for step in trace[1:-1]:
        assert step['kind'] == 'gate' and step['input_id'] == prev['node_id']
        assert abs(float(step['arrival']) - float(step['input_arrival']) -
                   float(step['arc_delay'])) < 0.00002, step
        prev = step
    assert trace[-1]['driver_id'] == prev['node_id']
    assert abs(float(trace[-1]['arrival']) - float(target['pair_delay'])) < 1e-6
    assert re.search(r'Arrival\s*=\s*194\.4', output), 'old display value changed'
    rows = []
    for row in catalog:
        r = indexed[f"pi{int(row['pi'])-1:02d}", f"po{int(row['po']):02d}"]
        rows.append(dict(path_id=row['path_id'], source_pi=r['source_pi'], target_po=r['target_po'],
                         reachable=r['reachable'], pair_delay=r['pair_delay'],
                         rise_arrival=r['rise_arrival'], fall_arrival=r['fall_arrival'],
                         precision=r['precision'], timing_model=r['timing_model']))
    write_csv(out / 'source_restricted_baseline_delays.csv', rows, list(rows[0]))
    (out / 'baseline_trace.json').write_text(json.dumps(trace, indent=2) + '\n')
    (out / 'baseline_summary.json').write_text(json.dumps(dict(
        old_display_ps=194.4, corrected_pair_delay=float(target['pair_delay']),
        trace_gates=len(trace)-2, mapped_input_sha256=sha(args.aig),
        checks=len(cc), all_checks_pass=True), indent=2) + '\n')
    return float(target['pair_delay'])


def unreachable(args, out):
    # Every c1355 input against po11 and po00: look for a genuine disconnected pair.
    queries = '; '.join(f'print_pair_delay -v po{po:02d} pi{pi:02d}'
                        for po in (0, 11) for pi in range(41))
    output = abc(args.abc, mapped_script(args.lib, args.aig, queries), out / 'reachability.log')
    rr, cc = records(output, 'PAIR_RESULT'), records(output, 'PAIR_CHECK')
    assert len(rr) == 82 and len(cc) == 82
    for r, c in zip(rr, cc):
        check_result(r, c)
    unreachable_pairs = [dict(source_pi=r['source_pi'], target_po=r['target_po'])
                         for r in rr if r['reachable'] == '0']
    fixture_script = mapped_script(args.lib, args.fixture,
                                   'print_pair_delay -v po01 pi00; print_pair_delay -tv po00 pi00')
    fixture_out = abc(args.abc, fixture_script, out / 'unreachable_fixture.log')
    fixture_rr = records(fixture_out, 'PAIR_RESULT')
    fixture_cc = records(fixture_out, 'PAIR_CHECK')
    assert len(fixture_rr) == len(fixture_cc) == 2
    for r, c in zip(fixture_rr, fixture_cc):
        check_result(r, c)
    assert fixture_rr[0]['reachable'] == '0' and fixture_rr[0]['pair_delay'] == 'NA'
    assert fixture_rr[1]['reachable'] == '1'
    fixture_trace = records(fixture_out, 'PAIR_TRACE')
    assert fixture_trace[0]['name'] == 'pi00' and fixture_trace[-1]['name'] == 'po00'
    (out / 'reachability_summary.json').write_text(json.dumps(dict(
        tested=82, c1355_unreachable_pairs=unreachable_pairs,
        fixture_disconnected='PASS', fixture_connected='PASS'), indent=2) + '\n')


def choose(rows):
    targets = {6: 3, 9: 5, 24: 5, 37: 1, 38: 2}
    by = defaultdict(list)
    for row in rows:
        if row['rewrite_status'] == 'ACCEPTED' and row['cec_pass'].lower() == 'true':
            by[int(row['region_id'])].append(row)
    selected = []
    for region, n in targets.items():
        vals = sorted(by[region], key=lambda r: (float(r['litecut_delta']), r['candidate_id']))
        assert len(vals) >= n, (region, len(vals))
        indexes = sorted({round(i*(len(vals)-1)/(n-1)) for i in range(n)}) if n > 1 else [0]
        selected.extend(vals[i] for i in indexes)
    assert len(selected) == 16
    return selected


def verify_trace(row):
    path = Path(row['trace_path']) if row['trace_path'] else None
    if path is None:
        aig = Path(row['aig_path'])
        path = aig.parent.parent / 'traces' / (aig.stem + '.jsonl')
    assert path.exists(), path
    events = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
    matching = [e for e in events if str(e.get('root')) == row['root_id']
                and str(e.get('cut_slot')) == row['cut_id']
                and str(e.get('selected_subgraph')) == row['candidate_subgraph_index']
                and e.get('final_status') == 'ACCEPTED']
    assert len(matching) == 1, (path, row['candidate_id'])
    # The old trace schema is preserved verbatim and is inspected in the pilot manifest.
    return str(path), matching


def pilot(args, out, baseline_delay):
    selected = choose(read_csv(args.calibration))
    results = []
    manifest = []
    for i, row in enumerate(selected):
        aig = Path(row['aig_path'])
        assert aig.exists() and sha(aig) == row['aig_sha256']
        trace_path, events = verify_trace(row)
        cec_script = f'read "{args.aig}"; read "{aig}"; cec -n'
        cec_out = abc(args.abc, cec_script, out / f'pilot_{i:02d}_cec.log')
        assert re.search(r'(Networks are equivalent|cec.*proved)', cec_out, re.I), (i, cec_out[-500:])
        timing_script = mapped_script(args.lib, aig, 'print_pair_delay -v po11 pi03')
        output = abc(args.abc, timing_script, out / f'pilot_{i:02d}_timing.log')
        rr, cc = records(output, 'PAIR_RESULT'), records(output, 'PAIR_CHECK')
        assert len(rr) == len(cc) == 1
        r = rr[0]
        check_result(r, cc[0])
        assert (r['source_pi'], r['target_po'], r['reachable']) == ('pi03', 'po11', '1')
        rewritten = float(r['pair_delay'])
        results.append(dict(region_id=row['region_id'], root_id=row['root_id'],
                            cut_id=row['cut_id'], candidate_id=row['candidate_id'],
                            litecut_current=row['litecut_current'],
                            litecut_candidate=row['litecut_candidate'],
                            litecut_delta=row['litecut_delta'],
                            baseline_pair_delay=format(baseline_delay, '.9g'),
                            rewritten_pair_delay=r['pair_delay'],
                            true_pair_delta=format(rewritten-baseline_delay, '.9g'),
                            reachable_before=1, reachable_after=1,
                            cec_pass=True, rewrite_status='ACCEPTED',
                            aig_sha256=row['aig_sha256'], aig_path=str(aig)))
        manifest.append(dict(candidate_id=row['candidate_id'], trace_path=trace_path,
                             matching_root_events=events, cec_script=cec_script,
                             timing_script=timing_script))
    write_csv(out / 'source_restricted_single_candidate_pilot.csv', results, list(results[0]))
    (out / 'pilot_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    x = np.array([float(r['litecut_delta']) for r in results])
    y = np.array([float(r['true_pair_delta']) for r in results])
    from scipy.stats import pearsonr, spearmanr
    # Epsilon is chosen after inspecting raw float32 deltas; not inherited from old 0.01ps.
    nonzero = sorted(abs(float(v)) for v in y if v != 0)
    epsilon = min(0.001, nonzero[0]/2) if nonzero else 0.001
    sign = lambda z: 0 if abs(z) <= epsilon else (1 if z > 0 else -1)
    metrics = dict(n=len(results), true_delta_min=float(y.min()), true_delta_max=float(y.max()),
                   litecut_delta_min=float(x.min()), litecut_delta_max=float(x.max()),
                   pearson=float(pearsonr(x, y).statistic) if np.std(y) and np.std(x) else None,
                   spearman=float(spearmanr(x, y).statistic) if np.std(y) and np.std(x) else None,
                   raw_mae=float(np.mean(abs(x-y))), neutral_epsilon_ps=epsilon,
                   sign_agreement=float(np.mean([sign(a)==sign(b) for a,b in zip(x,y)])),
                   true_neutral_count=sum(sign(v)==0 for v in y),
                   caveat='MIO source-restricted pair delay, not Liberty stime or exact Gate3B AIG path delay')
    (out / 'source_restricted_pilot_metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')


def main():
    p = argparse.ArgumentParser()
    for key in ('abc', 'lib', 'aig', 'catalog', 'calibration', 'fixture', 'out'):
        p.add_argument('--'+key, type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    delay = baseline(args, args.out)
    unreachable(args, args.out)
    pilot(args, args.out, delay)


if __name__ == '__main__':
    main()
