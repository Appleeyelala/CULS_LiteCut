#!/usr/bin/env python3
"""Regression of archived selected combinations; Replay and CEC only, no Mapping."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from global_timing.evaluation import cec_passed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-run', type=Path, required=True)
    p.add_argument('--circuit', type=int, required=True)
    p.add_argument('--gpuls', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--prepared', action='store_true')
    args = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not args.output.resolve().is_relative_to('/data/wushangge'):
        p.error('Requires a Slurm allocation and /data output.')
    source = args.reference_run/'output'/f'case_{args.circuit:04d}_Ours'
    choices = source/'work/final/eval_0000/rewrite/choices.txt'
    expected = {tuple(map(int, line.split()[1:])) for line in choices.read_text().splitlines() if line.strip()}
    context = json.loads((source/'context.json').read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, CULS_ALLOW_NONPOSITIVE_CANDIDATES='1', CULS_REPLAY_ALLOW_NONPOSITIVE_GAIN='1')
    env['CULS_REPLAY_PREPARED'] = '1' if args.prepared else '0'
    command = [str(args.gpuls), '-c', f'read {source}/input.aig; gate3e_execution_replay -o {args.output}/raw --region-map {source}/work/partition/owners.csv --config {choices}']
    commands = []
    def run(argv, name):
        start = time.perf_counter()
        with (args.output/f'{name}.log').open('x') as log:
            result = subprocess.run(argv, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=900)
        commands.append(dict(argv=argv, returncode=result.returncode, elapsed_s=time.perf_counter()-start))
        return result.returncode
    code = run(command, 'replay')
    trace = args.output/'raw/traces/choice.jsonl'
    selected = []
    if trace.exists():
        with trace.open() as f:
            for line in f:
                row = json.loads(line)
                if row.get('preselected'):
                    selected.append({k:row[k] for k in ['root','cut_slot','selected_subgraph','final_status','status_reason','damaged_cut_leaves']})
    actual = {(r['root'],r['cut_slot'],r['selected_subgraph']) for r in selected}
    full = code == 0 and actual == expected and all(r['final_status']=='ACCEPTED' for r in selected)
    aig = args.output/'raw/aigs/choice.aig'
    cec = False
    if aig.exists():
        code = run([context['tools']['abc']['path'], '-c', f'cec -n {source}/input.aig {aig}'], 'cec')
        cec = code == 0 and cec_passed((args.output/'cec.log').read_text())
    result = dict(circuit=args.circuit, prepared=args.prepared, expected_count=len(expected),
                  installed_count=sum(r['final_status']=='ACCEPTED' for r in selected),
                  full_install=full, cec_pass=cec, status='PASS' if full and cec else 'FAIL',
                  selected=selected, mapping_calls=0, commands=commands,
                  gpuls_sha256=hashlib.sha256(args.gpuls.read_bytes()).hexdigest(),
                  choices_sha256=hashlib.sha256(choices.read_bytes()).hexdigest(), reference=str(source))
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result), flush=True)
    return 0 if full and cec else 1


if __name__=='__main__':
    raise SystemExit(main())
