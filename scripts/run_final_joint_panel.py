#!/usr/bin/env python3
"""Enforce the frozen panel/protocol, run serial cold controls, then audit."""
import argparse
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--tool-run', type=Path, required=True)
    parser.add_argument('--legacy-tool-run', type=Path, required=True)
    parser.add_argument('--legacy-source', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    if not os.environ.get('SLURM_JOB_ID') or not run.is_relative_to('/data/wushangge'):
        parser.error('Requires committed-source /data workspace and Slurm allocation.')
    from global_timing.common import digest, save_json
    protocol_path = run / 'src/config/gpu_joint_final_protocol.json'
    protocol = json.loads(protocol_path.read_text())
    panel_path = run / 'src' / protocol['panel']
    if digest(panel_path) != protocol['panel_sha256']:
        raise ValueError('Frozen panel checksum changed.')
    panel = json.loads(panel_path.read_text())
    if protocol['seeds'] != panel['seeds']:
        raise ValueError('Protocol seeds differ from the prospective panel.')
    inventory = Path(panel['inventory'])
    if digest(inventory) != panel['inventory_sha256']:
        raise ValueError('Input inventory checksum changed.')
    stored = {r['index']: r for r in json.loads(inventory.read_text())['rows']}
    for row in panel['rows']:
        other = stored[row['index']]
        if (other['relative_path'] != row['relative_path'] or other['sha256'] != row['sha256']
                or digest(Path(other['staged_path'])) != row['sha256']):
            raise ValueError('Frozen input identity changed.')
    if len(panel['rows']) != 16 or len({r['sha256'] for r in panel['rows']}) != 16:
        raise ValueError('Frozen 16-input coverage changed.')
    if protocol['installation_gain_policy'] != 'positive_gain':
        raise ValueError('Primary final panel must preserve default positive-gain installation.')
    command = [sys.executable, '-B', str(run / 'src/scripts/profile_single_method.py'),
        '--run', str(run), '--tool-run', str(args.tool_run.resolve()),
        '--legacy-tool-run', str(args.legacy_tool_run.resolve()), '--legacy-source', str(args.legacy_source.resolve()),
        '--manifest', str(inventory), '--indices', ','.join(str(r['index']) for r in panel['rows']),
        '--seeds', ','.join(map(str, protocol['seeds'])), '--variants', ','.join(protocol['variants']),
        '--evaluations', str(protocol['candidate_budget_including_noop_failed_and_repair']),
        '--proposal-steps', str(protocol['quadratic_steps']), '--trajectories', str(protocol['quadratic_trajectories']),
        '--max-changes', str(protocol['max_changes']), '--validation-workers', str(protocol['cpu_validation_workers']),
        '--tool-timeout', str(protocol['tool_timeout_s']), '--case-timeout', str(protocol['cold_child_timeout_s'])]
    save_json(run / 'input/final_panel_preflight.json', {'protocol_sha256': digest(protocol_path),
        'panel_sha256': digest(panel_path), 'inventory_sha256': digest(inventory),
        'source_commit': json.loads((run / 'run.json').read_text())['source_commit'],
        'profile_command': command, 'expected_cold_children': 16 * len(protocol['seeds']) * len(protocol['variants']),
        'purpose': 'Frozen final comparison, not a development tuning run; serial per-circuit controls and seed-isolated caches.'})
    subprocess.run(command, cwd=run, check=True)
    profile_path = run / 'output/profile_summary.json'
    profiles = json.loads(profile_path.read_text())
    expected = set(itertools.product((r['index'] for r in panel['rows']), protocol['seeds'], protocol['variants']))
    actual = [(r['index'], r['seed'], r['variant']) for r in profiles['records']]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('Final profile coverage differs from the frozen experiment.')
    # Do not remove failures or substitute a native/no-op result for them.
    for row in profiles['records']:
        if row['status'] == 'PASS' and row['returncode'] != 0:
            raise ValueError('Contradictory final child success status.')
        if row['status'] != 'PASS' or row['variant'] in ('original', 'native_abc', 'native_culs'):
            continue
        detail_path = Path(row['detail'])
        config_path = detail_path.parent / f'case_{row["index"]:04d}/config.json'
        actual_config = json.loads(config_path.read_text())
        required = {'seed': row['seed'], 'evaluation_budget': protocol['candidate_budget_including_noop_failed_and_repair'],
                    'warmup_evaluations': protocol['warmup_parameter'], 'feedback_batch': protocol['feedback_batch'],
                    'area_limit_ratio': protocol['area_limit_ratio'], 'timeout_s': protocol['tool_timeout_s']}
        if not row['variant'].startswith(('random_', 'greedy_')):
            required.update(calibration_steps=protocol['calibration_steps'], gradient_restarts=protocol['legacy_restarts'],
                gradient_steps=protocol['legacy_steps'] if row['variant'] == 'legacy_cpu' else protocol['quadratic_steps'])
        if any(actual_config.get(k) != v for k, v in required.items()) or actual_config.get('allow_nonpositive_gain', False):
            raise ValueError('Actual successful child config differs from the frozen protocol.')
    subprocess.run([sys.executable, '-B', str(run / 'src/scripts/analyze_joint_panel.py'),
                    '--run', str(run), '--profile-run', str(run)], cwd=run, check=True)
    analysis = run / 'output/panel_analysis/summary.json'
    save_json(run / 'output/final_panel_completion.json', {
        'status': 'EXECUTED_AND_AUDITED', 'expected_cold_children': len(expected),
        'reported_cold_children': len(actual), 'declared_failed_children': sum(r['status'] != 'PASS' for r in profiles['records']),
        'protocol_sha256': digest(protocol_path), 'profile_summary_sha256': digest(profile_path),
        'analysis_summary_sha256': digest(analysis),
        'scope': 'Execution/coverage/raw-evidence completion only; QoR advantage, runtime goals and scientific contribution require separate assessment.'})


if __name__ == '__main__':
    main()
