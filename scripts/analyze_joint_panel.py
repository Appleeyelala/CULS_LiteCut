#!/usr/bin/env python3
"""Audit completed controls and derive measured delay/runtime statistics."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def comparisons(rows):
    lookup = {}
    for row in rows:
        key = row['index'], row['seed'], row['variant']
        if key in lookup:
            raise ValueError('Ambiguous repeated input/seed/method; analyze separate repetitions explicitly.')
        lookup[key] = row
    results = {}
    for control in ('legacy_cpu', 'random_batch', 'adaptive_batch_cpu', 'greedy_litecut', 'native_abc', 'native_culs'):
        pairs = []
        for row in rows:
            if row['variant'] != 'adaptive_batch_auto':
                continue
            other = lookup.get((row['index'], row['seed'], control))
            pair = {'index': row['index'], 'seed': row['seed']}
            if other is None or row['status'] != 'PASS' or other['status'] != 'PASS':
                pair['status'] = 'MISSING_OR_FAILED'
            elif not row.get('area_feasible_1p05') or not other.get('area_feasible_1p05'):
                pair['status'] = 'AREA_INFEASIBLE_OR_REFERENCE_MISSING'
            else:
                a, b = row['delay_ps'], other['delay_ps']
                shared = row['slurm_job_id'] == other['slurm_job_id']
                pair.update(status='MEASURED', proposed_delay_ps=a, control_delay_ps=b,
                    delay_ratio=a / b if b > 0 else None,
                    outcome='win' if a < b - 2e-5 else 'loss' if a > b + 2e-5 else 'tie',
                    matched_consumed_requests=row['requests'] == other['requests'],
                    shared_allocation=shared,
                    control_over_proposed_runtime=(other['end_to_end_with_input_staging_s'] / row['end_to_end_with_input_staging_s']) if shared else None,
                    same_final_choice=row.get('best_choices_sha256') == other.get('best_choices_sha256') if control == 'adaptive_batch_cpu' else None,
                    same_request_sequence=row.get('request_sequence_sha256') == other.get('request_sequence_sha256') if control == 'adaptive_batch_cpu' else None,
                    proposed_cuda_used='cuda' in (row.get('proposal_devices') or []))
            pairs.append(pair)
        valid = [p for p in pairs if p['status'] == 'MEASURED']
        ratios = sorted(p['delay_ratio'] for p in valid if p['delay_ratio'] is not None and p['delay_ratio'] > 0)
        speed = [p['control_over_proposed_runtime'] for p in valid if p['control_over_proposed_runtime'] is not None]
        q90 = None
        if ratios:
            position = .9 * (len(ratios) - 1)
            left = int(position)
            q90 = ratios[left] + (ratios[min(left + 1, len(ratios) - 1)] - ratios[left]) * (position - left)
        results[control] = {'planned_pairs': len(pairs), 'measured_area_feasible_pairs': len(valid),
            'outcomes': {k: sum(p['outcome'] == k for p in valid) for k in ('win', 'tie', 'loss')},
            'geometric_mean_delay_ratio': math.exp(statistics.mean(map(math.log, ratios))) if ratios else None,
            'p90_delay_ratio': q90, 'median_shared_allocation_speed_ratio': statistics.median(speed) if speed else None,
            'pairs': pairs}
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--profile-run', type=Path, action='append', required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    if not os.environ.get('SLURM_JOB_ID') or not run.is_relative_to('/data/wushangge'):
        parser.error('Requires a new prepared /data run and Slurm allocation.')
    from global_timing.common import save_json
    from global_timing.profile_results import add_runtime_metrics, audit_profiles
    output = run / 'output/panel_analysis'
    output.mkdir(exist_ok=False)
    rows, hashes = audit_profiles(args.profile_run)
    rows = add_runtime_metrics(rows)
    comparison = comparisons(rows)
    save_json(output / 'audited_records.json', rows)
    save_json(output / 'evidence_hashes.json', hashes)
    save_json(output / 'comparisons.json', comparison)
    flat = [{k: v for k, v in row.items() if k not in ('points', 'baseline')} for row in rows]
    fields = sorted({k for row in flat for k in row})
    with (output / 'results.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat)
    summary = {'status': 'PASS', 'records': len(rows), 'failed_or_incomplete_records': sum(r['status'] != 'PASS' for r in rows),
               'audited_successful_measurements': sum(len(r['points']) for r in rows),
               'scope': 'Every successful point audited against exact installation, raw CEC, raw Liberty, netlist and asset/command hashes; failed methods retained without imputation. Runtime cutoffs use actual completion events plus one input staging charge. Paired speed claims require shared allocation; frozen panel and independent circuits/replicates must be identified separately.'}
    save_json(output / 'summary.json', summary)
    header = '| 输入 | seed | 方法 | Delay ps | Area | E2E s | 达标时间 s | 30s / 120s / 600s best ps |\n|---|---:|---|---:|---:|---:|---:|---|\n'
    def number(v):
        return '未达到/缺失' if v is None else f'{v:.6f}'
    table = ''.join('| '+str(r['input'])+' | '+str(r['seed'])+' | '+r['variant']+' | '+number(r.get('delay_ps'))+' | '+number(r.get('area'))+' | '+number(r.get('end_to_end_with_input_staging_s'))+' | '+number(r.get('time_to_quality_s'))+' | '+ ' / '.join(number(r.get(f'best_at_{t}s_ps')) for t in (30,120,600))+' |\n' for r in rows)
    (output / 'REPORT.md').write_text('# 已验证 Delay–Runtime 对照\n\n'+header+table+'\n达标阈值为面积可行的no-op/native ABC较低delay的1.01倍；未达到和缺失不填值。全部失败、真实调用数、逐点时间和来源见CSV/JSON。比较统计与CPU/GPU逐请求一致性见comparisons.json。该审计通过不等于算法取得优势。\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
