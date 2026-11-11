#!/usr/bin/env python3
"""Aggregate saved real runs and compare repeat outcomes without new experiments."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from global_timing.common import save_json
from global_timing.reporting import write_report


def canonical_outcomes(summary):
    """Keep scientific outcomes and chosen IDs; omit paths, timing and logs."""
    fields = ['cec', 'delay_ps', 'area', 'adp', 'po_arrivals_ps', 'critical_po_indices', 'input_sha256']

    def measurement(value):
        return {key: value.get(key) for key in fields}

    def observation(row):
        return {'status': row['status'], 'choices': row['choices'], 'applied_rewrites': row.get('applied_rewrites'),
                'metrics': measurement(row.get('metrics', {})),
                'source': (row.get('prediction_before_evaluation') or {}).get('source')}

    return {'config': summary['config'], 'original': measurement(summary['original']),
            'native_abc': measurement(summary['native_abc']), 'noop': observation(summary['replay_noop']),
            'greedy': [observation(r) for r in summary['greedy_litecut']['observations']],
            'independent': [observation(r) for r in summary['independent_regions']['observations']],
            'random': [observation(r) for r in summary['random_search']],
            'gradient': {mode: [observation(r) for r in item['observations']]
                         for mode, item in summary['gradient_search'].items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--suites', type=Path, nargs='*', default=[])
    parser.add_argument('--benchmarks', type=Path, nargs='*', default=[])
    parser.add_argument('--repeat', type=Path, nargs=2)
    parser.add_argument('--expected-inputs', type=int)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not str(args.output.resolve()).startswith('/data/wushangge/'):
        raise SystemExit('Run batch reports under Slurm in /data.')
    args.output.mkdir(parents=True, exist_ok=False)
    records, comparison = [], []
    for suite in args.suites:
        for line in (suite / 'records.jsonl').read_text().splitlines():
            record = json.loads(line)
            records.append(record)
            comparison.extend(dict(row, input_index=record['index'], input_name=record['relative_path'],
                                   source_summary=record.get('summary')) for row in record.get('comparison', []))
    for benchmark in args.benchmarks:
        tag = hashlib.sha256(str(benchmark.resolve()).encode()).hexdigest()[:12]
        rows = write_report(benchmark, args.output / ('report_' + tag))
        comparison.extend(dict(row, input_index=None, input_name=benchmark.name,
                               source_summary=str(benchmark / 'summary.json')) for row in rows)
    if comparison:
        fields = sorted({key for row in comparison for key in row})
        with (args.output / 'comparison_all.csv').open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader(); writer.writerows(comparison)
    unique = {r['index'] for r in records}
    coverage = {'records': len(records), 'unique_inputs': len(unique), 'status_counts': dict(Counter(r['status'] for r in records)),
                'expected_inputs': args.expected_inputs, 'coverage_complete': args.expected_inputs is not None and
                unique == set(range(args.expected_inputs)),
                'scope': 'Counts reflect saved requests, including failed inputs; PASS is not proof of improved timing.'}
    save_json(args.output / 'coverage.json', coverage)
    save_json(args.output / 'case_records.json', records)
    if args.repeat:
        left, right = [json.loads((p / 'summary.json').read_text()) for p in args.repeat]
        a, b = canonical_outcomes(left), canonical_outcomes(right)
        checks = {key: a[key] == b[key] for key in a}
        save_json(args.output / 'repeat_comparison.json', {'paths': list(map(str, args.repeat)),
                  'exact_outcome_match': all(checks.values()), 'checks': checks,
                  'scope': 'Exact choices, actual AIG hashes, CEC, all PO timings, area and ADP; wall-clock runtime excluded.'})
    lines = ['# 已保存实验汇总', '', f'实际有记录的输入：{len(unique)}；状态：{coverage["status_counts"]}。',
             f'全量覆盖是否完成：{coverage["coverage_complete"]}。', '',
             '完整逐输入状态见 case_records.json；所有方法原始数值见 comparison_all.csv。',
             '固定PI/PO顺序和相同Liberty流程才可比较；历史MIO或mapper mode0指标不并入本表。',
             '同一输入的重跑保留独立来源，不用后一次结果覆盖先前失败；总表可能包含重复输入。']
    (args.output / 'REPORT.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
