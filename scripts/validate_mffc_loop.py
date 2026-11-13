#!/usr/bin/env python3
"""Build/test and run exactly A/B/Ours on three development inputs in Slurm."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--tool-run', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    if not os.environ.get('SLURM_JOB_ID') or not run.is_relative_to('/data/wushangge'):
        parser.error('Requires a committed /data Slurm GPU run.')
    for command in [
        [sys.executable, '-B', str(run / 'src/scripts/build_mffc_gpu.py'), '--run', str(run)],
        [sys.executable, '-B', str(run / 'src/scripts/run_mffc905.py'), '--run', str(run),
         '--tool-run', str(args.tool_run), '--native-build-run', str(run), '--manifest', str(args.manifest),
         '--indices', '47,80,0', '--case-timeout', '1800'],
    ]:
        subprocess.run(command, cwd=run, check=True)
    sys.path.insert(0, str(run / 'src'))
    from global_timing.evaluation import cec_passed, parse_timing
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    evaluations, observations, rows = 0, 0, []
    def evaluation(work, metrics):
        nonlocal evaluations
        actual = json.loads((work / 'result.json').read_text())
        if actual != metrics or not cec_passed((work / 'cec.log').read_text()):
            raise ValueError('Stored metrics/raw CEC do not agree: ' + str(work))
        measured = parse_timing((work / 'mapping.log').read_text())
        for key in ['delay_ps', 'area', 'po_arrivals_ps', 'timing_model']:
            if measured[key] != metrics[key]:
                raise ValueError('Raw Liberty measurement mismatch: ' + key)
        if sha(work / 'mapped.v') != metrics['mapped_sha256']:
            raise ValueError('Mapped netlist identity mismatch.')
        evaluations += 1
    for path in sorted((run / 'output/records').glob('*.json')):
        record = json.loads(path.read_text())
        if record['status'] != 'PASS':
            raise ValueError('A development method failed; inspect ' + str(path))
        result = json.loads(Path(record['result_path']).read_text())
        work = Path(record['detail_path']) / 'work'
        if record['method'] == 'Ours':
            evaluation(work / 'original', result['original'])
            for observation in result['observations']:
                if observation['status'] != 'PASS':
                    if observation.get('metrics'):
                        raise ValueError('Failed request acquired a timing label.')
                    continue
                method = 'replay_noop' if observation['index'] == 0 else 'optimization'
                request = work / method / f"eval_{observation['index']:04d}"
                evaluation(request / 'evaluation', observation['metrics'])
                replay = json.loads((request / 'rewrite/replay.json').read_text())
                trace = [json.loads(line) for line in (request / 'rewrite/raw/traces/choice.jsonl').read_text().splitlines() if line.strip()]
                selected = [t for t in trace if t['preselected']]
                wanted = {tuple(r) for r in replay['requested']}
                got = {(t['root'], t['cut_slot'], t['selected_subgraph']) for t in selected}
                if (wanted != got or len(selected) != len(wanted)
                        or any(t['final_status'] != 'ACCEPTED' for t in selected)
                        or replay['applied_rewrites'] != len(wanted)):
                    raise ValueError('Development combination not exactly installed.')
                if sha(observation['aig']) != observation['metrics']['input_sha256']:
                    raise ValueError('Installed AIG identity mismatch.')
                observations += 1
            if result['selected']['metrics']['area'] > result['replay_noop']['metrics']['area'] * 1.05 + 1e-8:
                raise ValueError('Final actual area constraint violated.')
        else:
            evaluation(work / 'evaluation', result['selected']['metrics'])
        rows.append({'index': record['index'], 'method': record['method'],
                     'delay_ps': result['selected']['metrics']['delay_ps'], 'area': result['selected']['metrics']['area'],
                     'cold_e2e_s': record['end_to_end_runtime_s'], 'source': result['selected']['source'],
                     'backend': result['backend'], 'native_profiles': result.get('native_profiles', []),
                     'requests': result.get('requests_including_noop_failed_repair'),
                     'failed_requests': result.get('failed_requests')})
    if len(rows) != 9:
        raise ValueError('Expected all three methods on three development inputs.')
    payload = {'status': 'PASS', 'method_points': len(rows), 'audited_evaluations': evaluations,
               'audited_successful_replay_observations': observations, 'rows': rows,
               'scope': 'Development correctness only, not full905, not independent GPU performance attribution.'}
    with (run / 'output/mffc_loop_validation.json').open('x') as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
    print(json.dumps({k: payload[k] for k in ['status', 'method_points', 'audited_evaluations']}))


if __name__ == '__main__':
    main()
