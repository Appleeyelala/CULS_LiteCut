#!/usr/bin/env python3
"""Aggregate saved real runs and compare repeat outcomes without new experiments."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from global_timing.common import save_json
from global_timing.cohort import analysis_sources
from global_timing.reporting import write_report
from global_timing.evaluation import native_baseline_metrics
from global_timing.structure import aig_structure_digest


def canonical_outcomes(summary, include_byte_hash=True):
    """Keep scientific outcomes and chosen IDs; omit paths, timing and logs."""
    fields = ['cec', 'delay_ps', 'area', 'adp', 'po_arrivals_ps', 'critical_po_indices']
    if include_byte_hash:
        fields.append('input_sha256')

    def measurement(value):
        return {key: value.get(key) for key in fields}

    def observation(row):
        return {'status': row['status'], 'choices': row['choices'], 'applied_rewrites': row.get('applied_rewrites'),
                'metrics': measurement(row.get('metrics', {})),
                'source': (row.get('prediction_before_evaluation') or {}).get('source'),
                'rejected_requests': row.get('execution_failure', {}).get('rejected_requests')}

    def group(item):
        return {'observations': [observation(row) for row in item['observations']],
                'selected_best': observation(item['best']) if item.get('best') else None}

    native = native_baseline_metrics(summary)
    native_result = (measurement(native) if native is not None else
                     {'status': 'FAIL', 'stage': summary['native_abc_attempt']['stage'], 'metrics': None})
    return {'config': summary['config'], 'original': measurement(summary['original']),
            'native_abc': native_result, 'noop': observation(summary['replay_noop']),
            'greedy': group(summary['greedy_litecut']),
            'independent': group(summary['independent_regions']),
            'random': [observation(r) for r in summary['random_search']],
            'gradient': {mode: group(item) for mode, item in summary['gradient_search'].items()}}


def saved_graphs(benchmark, summary):
    paths = ({'native_abc': benchmark / 'native_abc/rewritten.aig'}
             if native_baseline_metrics(summary) is not None else {})
    paths['original'] = Path(summary['original']['provenance']['original']['path'])
    groups = {'noop': [summary['replay_noop']], 'random': summary['random_search'],
              'greedy': summary['greedy_litecut']['observations'],
              'independent': summary['independent_regions']['observations']}
    groups.update({name: item['observations'] for name, item in summary['gradient_search'].items()})
    for group, rows in groups.items():
        for row in rows:
            if row['status'] == 'PASS':
                paths[f'{group}/{row["index"]}'] = Path(row['aig'])
    return {key: aig_structure_digest(path) for key, path in paths.items()}


def paired_statistics(records):
    """Compare objectives only on matching designs, budgets and feasibility."""
    if len({record['index'] for record in records}) != len(records):
        return {'status': 'NOT_POOLED', 'reason': 'Repeated input indices: select one declared run cohort before computing aggregate performance.'}
    comparisons = {}
    for proposed, control, field, constrained in [
        ('gradient_area_constraint', 'random_search_area_constraint', 'delay_ps', True),
        ('gradient_joint', 'random_search_joint', 'actual_joint_objective', False)]:
        pairs, omitted = [], []
        for record in records:
            rows = {row['method']: row for row in record.get('comparison', [])}
            a, b = rows.get(proposed), rows.get(control)
            reason = None
            if a is None or b is None:
                reason = 'Missing measured method result.'
            elif a['evaluations_including_baseline'] != b['evaluations_including_baseline']:
                reason = 'Different consumed evaluation counts; excluded from this strict paired aggregate.'
            elif constrained and not (a['area_constraint_feasible'] and b['area_constraint_feasible']):
                reason = 'Actual mapped area constraint not satisfied.'
            elif a.get(field) is None or b.get(field) is None:
                reason = 'Objective absent from this saved report version.'
            if reason:
                omitted.append({'index': record['index'], 'status': record['status'], 'reason': reason})
                continue
            av, bv = a[field], b[field]
            tied = math.isclose(av, bv, rel_tol=1e-6, abs_tol=1e-8)
            pairs.append({'index': record['index'], 'proposed': av, 'control': bv,
                'outcome': 'tie' if tied else 'win' if av < bv else 'loss',
                'improvement_pct': 100 * (bv - av) / bv if bv > 0 else None,
                'delay_ratio': a['delay_ps'] / b['delay_ps'] if b['delay_ps'] > 0 and a['delay_ps'] > 0 else None,
                'direct_gradient_evaluations': a.get('direct_gradient_evaluations')})
        improvements = [row['improvement_pct'] for row in pairs if row['improvement_pct'] is not None]
        ratios = [row['delay_ratio'] for row in pairs if row['delay_ratio'] is not None]
        comparisons[proposed] = {'control': control, 'objective': field, 'paired_count': len(pairs),
            'outcomes': dict(Counter(row['outcome'] for row in pairs)), 'omitted_count': len(omitted),
            'median_objective_improvement_pct': statistics.median(improvements) if improvements else None,
            'geometric_mean_delay_ratio': math.exp(statistics.mean(math.log(x) for x in ratios)) if ratios else None,
            'pairs': pairs, 'omitted': omitted}
    return {'status': 'COMPUTED', 'comparisons': comparisons,
        'scope': 'Paired saved real measurements with equal consumed budgets; failures and omissions remain in coverage, never imputed. A win is numerical, not a statistical-significance claim. Joint-objective wins need not reduce delay.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--suites', type=Path, nargs='*', default=[])
    parser.add_argument('--cohort', type=Path, help='Frozen declared retry selection; mutually exclusive with raw input lists.')
    parser.add_argument('--benchmarks', type=Path, nargs='*', default=[])
    parser.add_argument('--repeat', type=Path, nargs=2)
    parser.add_argument('--expected-inputs', type=int)
    args = parser.parse_args()
    if args.cohort and (args.suites or args.benchmarks):
        parser.error('Use --cohort or raw --suites/--benchmarks inputs.')
    if not os.environ.get('SLURM_JOB_ID') or not str(args.output.resolve()).startswith('/data/wushangge/'):
        raise SystemExit('Run batch reports under Slurm in /data.')
    args.output.mkdir(parents=True, exist_ok=False)
    records, comparison = [], []
    batches, cohort = analysis_sources(args.suites, args.cohort)
    if cohort is not None:
        if args.expected_inputs is not None and args.expected_inputs != cohort['expected_inputs']:
            raise ValueError('Expected input count differs from the frozen cohort inventory.')
        args.expected_inputs = cohort['expected_inputs']
    for batch in batches:
        for original_record in batch['records']:
            record = dict(original_record, source_commit=batch['metadata']['source_commit'], source_suite=batch['suite'])
            records.append(record)
            comparison.extend(dict(row, input_index=record['index'], input_name=record['relative_path'],
                                   source_summary=record.get('summary'), source_commit=record['source_commit'],
                                   source_suite=batch['suite']) for row in record.get('comparison', []))
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
                unique == set(range(args.expected_inputs)) and (cohort is None or cohort['coverage_complete']),
                'scope': 'Counts reflect saved requests, including failed inputs; PASS is not proof of improved timing.'}
    coverage['source_versions'] = dict(Counter(record['source_commit'] for record in records))
    if cohort is not None:
        coverage['frozen_cohort'] = {'path': str(args.cohort.resolve()),
            'sha256': hashlib.sha256(args.cohort.read_bytes()).hexdigest(), 'policy': cohort['policy'],
            'pending_declared_attempts': cohort['pending_declared_attempts'],
            'attempt_status_counts': cohort['attempt_status_counts']}
    save_json(args.output / 'coverage.json', coverage)
    save_json(args.output / 'case_records.json', records)
    statistics_report = paired_statistics(records)
    save_json(args.output / 'paired_statistics.json', statistics_report)
    save_json(args.output / 'paired_statistics_by_source_commit.json', {
        commit: paired_statistics([row for row in records if row['source_commit'] == commit])
        for commit in sorted({row['source_commit'] for row in records})})
    if args.repeat:
        left, right = [json.loads((p / 'summary.json').read_text()) for p in args.repeat]
        a, b = canonical_outcomes(left), canonical_outcomes(right)
        checks = {key: a[key] == b[key] for key in a}
        graph_a, graph_b = saved_graphs(args.repeat[0], left), saved_graphs(args.repeat[1], right)
        save_json(args.output / 'repeat_comparison.json', {'paths': list(map(str, args.repeat)),
                  'exact_outcome_match': all(checks.values()), 'checks': checks,
                  'numeric_and_choice_match': canonical_outcomes(left, False) == canonical_outcomes(right, False),
                  'exact_serialized_graph_match': graph_a == graph_b, 'graph_hashes': [graph_a, graph_b],
                  'scope': 'Byte hashes remain checked separately. Graph hashes preserve node IDs and PI/PO order but exclude names/comments, such as ABC write timestamps. Numeric checks include every choice, CEC, PO timing, area and ADP; wall-clock runtime excluded.'})
    lines = ['# 已保存实验汇总', '', f'实际有记录的输入：{len(unique)}；状态：{coverage["status_counts"]}。',
             f'全量覆盖是否完成：{coverage["coverage_complete"]}。', '',
             '完整逐输入状态见 case_records.json；所有方法原始数值见 comparison_all.csv。',
             '固定PI/PO顺序和相同Liberty流程才可比较；历史MIO或mapper mode0指标不并入本表。',
             '同一输入的重跑保留独立来源，不用后一次结果覆盖先前失败；总表可能包含重复输入。']
    if cohort is not None:
        lines += ['', '本表使用已校验的冻结cohort，每个输入选择最后声明且已有终态记录的尝试，失败同样参与选择。',
                  '完整旧尝试及未完成声明保留在cohort.json；不按时序数值选取最优重跑。']
    lines += ['', '逐输入源码版本保留在CSV；版本分层结果见paired_statistics_by_source_commit.json。',
              '多版本总体比较不等于单一最终版本的覆盖或泛化证明。']
    if statistics_report['status'] == 'COMPUTED':
        lines += ['', '配对比较仅使用相同输入、相同已消耗评估数且具备真实指标的结果；失败和缺失单独列出。', '',
                  '| 方法 | 对照 | 配对数 | 数值胜/平/负 | 未纳入数 |', '|---|---|---:|---|---:|']
        for method, item in statistics_report['comparisons'].items():
            counts = item['outcomes']
            lines.append(f"| {method} | {item['control']} | {item['paired_count']} | {counts.get('win', 0)}/{counts.get('tie', 0)}/{counts.get('loss', 0)} | {item['omitted_count']} |")
        lines += ['', 'joint按实际联合目标判定胜负；它不等于delay单项胜负。数值差异不代表统计显著性。']
    (args.output / 'REPORT.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
