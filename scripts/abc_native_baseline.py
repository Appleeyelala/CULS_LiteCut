#!/usr/bin/env python3
"""Step 3 v3 part C: global baselines from unpatched ABC commands.

For every circuit in the v3 cohort, run seven ABC command sequences from a fresh
``read_aiger`` on the original AIG and measure the global level D, the AND count,
the output TNS at D_orig-1..D_orig-3, the number of outputs above those
thresholds, the wall time and CEC equivalence.

Levels are recomputed with the step-1 ``propagate_levels`` and must match the
``lev`` that ABC prints; a mismatch aborts the circuit.
"""
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from global_timing.level_weights import propagate_levels
from global_timing.structure import parse_aig

# This ABC build ships without abc.rc, so the short aliases are unavailable.
# The script sequences are expanded to full command names:
#   b -> balance, rw -> rewrite, rf -> refactor, rwz -> rewrite -z, rfz -> refactor -z
COMMANDS = [
    ('balance', 'balance'),
    ('rewrite', 'rewrite'),
    ('rewrite_l', 'rewrite -l'),
    ('refactor', 'refactor'),
    ('resub', 'resub'),
    ('resyn2', 'balance; rewrite; refactor; balance; rewrite; rewrite -z; balance; '
               'refactor -z; rewrite -z; balance'),
    ('compress2', 'balance -l; rewrite -l; refactor -l; balance -l; rewrite -l; '
                  'rewrite -z -l; balance -l; refactor -z -l; rewrite -z -l; balance -l'),
]
STAT = None


def sha256(path):
    checksum = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def parse_stats(text):
    ands = lev = None
    for line in text.splitlines():
        if 'and =' in line and 'lev =' in line:
            parts = line.split()
            try:
                and_index = parts.index('and')
                lev_index = parts.index('lev')
                ands = int(parts[and_index + 2].rstrip(','))
                lev = int(parts[lev_index + 2].rstrip(','))
            except (ValueError, IndexError):
                continue
    return ands, lev


def run_abc(binary, commands, log_path):
    started = time.perf_counter()
    proc = subprocess.run([str(binary), '-c', commands], capture_output=True, text=True)
    elapsed = time.perf_counter() - started
    log_path.write_text(proc.stdout + proc.stderr)
    return proc, elapsed


def output_metrics(path, depth, ks=(1, 2, 3)):
    aig = parse_aig(path)
    levels = propagate_levels(aig)
    drivers = [literal >> 1 for literal in aig.po_literals]
    return dict(depth=levels.depth, ands=aig.and_count, po=len(drivers),
                tns={k: sum(max(0, levels.level[node] - (depth - k)) for node in drivers)
                     for k in ks},
                over={k: sum(1 for node in drivers if levels.level[node] > depth - k)
                      for k in ks})


def run_circuit(name, group, aig_path, abc, out_dir, log_dir):
    aig = parse_aig(aig_path)
    levels = propagate_levels(aig)
    depth = levels.depth
    drivers = [literal >> 1 for literal in aig.po_literals]
    baseline = dict(depth=depth, ands=aig.and_count, po=len(drivers),
                    tns={k: sum(max(0, levels.level[node] - (depth - k)) for node in drivers)
                         for k in (1, 2, 3)},
                    over={k: sum(1 for node in drivers if levels.level[node] > depth - k)
                          for k in (1, 2, 3)})
    original_stats = log_dir / f'{name}.original.log'
    proc, elapsed = run_abc(abc, f'read_aiger {aig_path}; print_stats', original_stats)
    _, abc_lev = parse_stats(proc.stdout + proc.stderr)
    if abc_lev is None:
        raise SystemExit(f'{name}: ABC printed no statistics for the original')
    if abc_lev != depth:
        raise SystemExit(f'{name}: ABC lev {abc_lev} != propagate_levels {depth}')

    rows = []
    for tag, sequence in COMMANDS:
        output = out_dir / f'{name}.{tag}.aig'
        log = log_dir / f'{name}.{tag}.log'
        proc, elapsed = run_abc(abc, f'read_aiger {aig_path}; {sequence}; print_stats; '
                                     f'write_aiger {output}', log)
        if proc.returncode != 0 or not output.exists():
            raise SystemExit(f'{name}/{tag}: ABC failed with {proc.returncode}')
        _, abc_lev = parse_stats(proc.stdout + proc.stderr)
        if abc_lev is None:
            raise SystemExit(f'{name}/{tag}: ABC printed no statistics')
        metrics = output_metrics(output, depth)
        if abc_lev != metrics['depth']:
            raise SystemExit(f'{name}/{tag}: ABC lev {abc_lev} != recomputed {metrics["depth"]}')
        cec_log = log_dir / f'{name}.{tag}.cec.log'
        # write_aiger drops PI symbols; compare the networks without names.
        cec_proc, cec_elapsed = run_abc(abc, f'read_aiger {aig_path}; cec -n {output}', cec_log)
        text = cec_log.read_text().lower()
        cec = ('equivalent' in text) and ('not equivalent' not in text)
        if not cec:
            raise SystemExit(f'{name}/{tag}: CEC failed')
        rows.append(dict(circuit=name, group=group, command=tag, sequence=sequence,
                         depth=metrics['depth'], delta_depth=metrics['depth'] - depth,
                         ands=metrics['ands'], delta_ands=metrics['ands'] - aig.and_count,
                         tns1=metrics['tns'][1], tns2=metrics['tns'][2], tns3=metrics['tns'][3],
                         over1=metrics['over'][1], over2=metrics['over'][2], over3=metrics['over'][3],
                         runtime_s=round(elapsed, 3), cec=cec))
    return dict(circuit=name, group=group, ands=aig.and_count, po=len(drivers),
                baseline=baseline, rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--cohort', required=True, type=Path)
    parser.add_argument('--abc', required=True, type=Path)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--limit', type=int, default=None, help='debug: only the first N circuits')
    args = parser.parse_args()
    run = args.run.resolve()
    if not os.getenv('SLURM_JOB_ID'):
        raise SystemExit('run this only inside the committed /data Slurm job')
    abc = args.abc.resolve()
    if not abc.is_file():
        raise SystemExit(f'missing ABC binary: {abc}')
    cohort = json.loads(args.cohort.read_text())
    out = run / 'output'
    (out / 'aig').mkdir(parents=True, exist_ok=True)
    (out / 'logs').mkdir(parents=True, exist_ok=True)

    circuits = []
    for row in cohort['pilots'] + cohort['development_new'] + cohort['sampled']:
        circuits.append((row['name'], row['group'], Path(row['path'])))

    if args.limit:
        circuits = circuits[:args.limit]

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_circuit, name, group, path, abc, out / 'aig', out / 'logs')
                   for name, group, path in circuits]
        for (name, group, _path), future in zip(circuits, futures):
            try:
                result = future.result()
                results.append(result)
                print(f"{result['circuit']}: D={result['baseline']['depth']} "
                      f"ands={result['ands']} rows={len(result['rows'])}", flush=True)
            except Exception as exc:  # record the failure and keep the cohort
                results.append(dict(circuit=name, group=group, status='FAILED',
                                    reason=str(exc)[:400], rows=[]))
                print(f'{name}: FAILED {exc}', flush=True)

    passed = [row for row in results if row.get('status') != 'FAILED']
    summary = dict(step='3_v3', part='C', cohort_size=len(passed),
                   failures=[row for row in results if row.get('status') == 'FAILED'],
                   abc=dict(path=str(abc), sha256=sha256(abc)),
                   commands=[dict(tag=tag, sequence=sequence) for tag, sequence in COMMANDS],
                   results=results, slurm_job_id=os.getenv('SLURM_JOB_ID'))
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    with (out / 'rows.tsv').open('w', newline='') as sink:
        fields = ['circuit', 'group', 'command', 'depth', 'delta_depth', 'ands', 'delta_ands',
                  'tns1', 'tns2', 'tns3', 'over1', 'over2', 'over3', 'runtime_s', 'cec']
        writer = csv.DictWriter(sink, fieldnames=fields, delimiter='\t', extrasaction='ignore')
        writer.writeheader()
        for result in passed:
            for row in result['rows']:
                writer.writerow(row)
    print('done', len(results), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
