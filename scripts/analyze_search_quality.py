#!/usr/bin/env python3
"""Audit saved search quality and proposal origins; no new timing evaluations."""
import argparse
from collections import Counter
import csv
import json
import math
import os
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from global_timing.common import digest, save_json
from global_timing.cohort import analysis_sources
from global_timing.search_analysis import analyze_search, aggregate_search


def write_csv(path, rows):
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: json.dumps(value, sort_keys=True) if isinstance(value, (list, dict)) else value
                          for key, value in row.items()} for row in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--suites', type=Path, nargs='+', default=[])
    inputs.add_argument('--cohort', type=Path, help='Validated frozen retry selection from build_benchmark_cohort.py.')
    parser.add_argument('--expected-inputs', type=int, required=True)
    parser.add_argument('--cohort-label', required=True)
    args = parser.parse_args()
    if args.expected_inputs < 1:
        parser.error('--expected-inputs must be positive.')
    output = args.output.resolve()
    if not os.environ.get('SLURM_JOB_ID') or not output.is_relative_to('/data/wushangge'):
        parser.error('Analysis requires Slurm and a new /data output directory.')
    output.mkdir(parents=True, exist_ok=False)
    snapshots = output / 'input_snapshots'
    snapshots.mkdir()
    coverage, analyzed, methods, predictions, gradients, failures, sources = [], [], [], [], [], [], []
    seen = set()
    batches, cohort = analysis_sources(args.suites, args.cohort)
    if cohort is not None and cohort['expected_inputs'] != args.expected_inputs:
        raise ValueError('Expected input count differs from the frozen cohort inventory.')
    for number, batch in enumerate(batches):
        suite = Path(batch['suite'])
        complete = batch['raw_complete']
        snapshot = snapshots / f'suite_{number:02d}.jsonl'
        with snapshot.open('xb') as stream:
            stream.write(complete)
        metadata = batch['metadata']
        sources.append({'suite': str(suite), 'source_commit': metadata['source_commit'],
                        'records_snapshot': str(snapshot), 'snapshot_sha256': digest(snapshot),
                        'snapshot_scope': 'Declared cohort selection' if cohort else 'Complete live JSONL prefix',
                        'ignored_incomplete_tail_bytes': batch['ignored_incomplete_tail_bytes'],
                        'metadata_path': batch['metadata_path'], 'binaries': metadata.get('binaries')})
        for row in batch['records']:
            if row['index'] in seen:
                raise ValueError('Duplicate input index: choose one declared cohort instead of pooling reruns.')
            seen.add(row['index'])
            identity = {'index': row['index'], 'input_name': row['relative_path'], 'input_sha256': row['sha256'],
                        'case_status': row['status'], 'source_commit': metadata['source_commit']}
            coverage.append(identity)
            if row['status'] not in {'PASS', 'PARTIAL'} or not row.get('summary'):
                failures.append(dict(identity, reason=row.get('reason'), timed_out=row.get('timed_out', False)))
                continue
            path = Path(row['summary']).resolve()
            if not path.is_relative_to(suite):
                raise ValueError('Summary is outside its declared suite.')
            try:
                summary = json.loads(path.read_text())
                if summary['original']['input_sha256'] != row['sha256']:
                    raise ValueError('Case summary input differs from its manifest record.')
                analysis = analyze_search(summary)
                evidence = dict(identity, audit_status='PASS', summary=str(path), summary_sha256=digest(path), analysis=analysis)
                analyzed.append(evidence)
                methods.extend(dict(identity, mode=mode, **value) for mode, value in analysis['methods'].items())
                predictions.extend(dict(identity, **sample) for sample in analysis['prediction_samples'])
                gradient_file = path.parent / 'report/gradients.csv'
                if gradient_file.is_file():
                    with gradient_file.open(newline='') as stream:
                        entries = list(csv.DictReader(stream))
                    for method in sorted({r['method'] for r in entries}):
                        selected = [r for r in entries if r['method'] == method and r['gradient_norm']]
                        norms = [float(r['gradient_norm']) for r in selected]
                        if any(not math.isfinite(x) or x < 0 for x in norms):
                            raise ValueError('Non-finite/negative saved gradient norm.')
                        gradients.append(dict(identity, method=method, recorded_gradient_samples=len(norms),
                            mean_norm=statistics.mean(norms) if norms else None, max_norm=max(norms) if norms else None,
                            zero_norm_samples=sum(x == 0 for x in norms),
                            projected_argmax_changes=sum(r['projection_changed'] == 'True' for r in selected),
                            gradient_csv=str(gradient_file), gradient_csv_sha256=digest(gradient_file)))
                save_json(output / f'case_{row["index"]:04d}.json', evidence)
            except Exception as exc:
                # An audit failure is distinct from a failed benchmark. It
                # cannot silently turn that case into a numeric observation.
                failures.append(dict(identity, audit_error=f'{type(exc).__name__}: {exc}'))
                analyzed = [r for r in analyzed if r['index'] != row['index']]
                methods = [r for r in methods if r['index'] != row['index']]
                predictions = [r for r in predictions if r['index'] != row['index']]
                gradients = [r for r in gradients if r['index'] != row['index']]
    pooled = aggregate_search(analyzed)
    report = {'cohort_label': args.cohort_label, 'sources': sources, 'recorded_inputs': len(seen),
              'expected_inputs': args.expected_inputs, 'coverage_complete': seen == set(range(args.expected_inputs)) and
              (cohort is None or cohort['coverage_complete']),
              'case_status_counts': dict(Counter(r['case_status'] for r in coverage)),
              'analyzed_inputs': len(analyzed), 'audit_failure_count': sum('audit_error' in r for r in failures),
              'aggregate': pooled, 'failures_and_omissions': failures,
              'analysis_source': str(Path(__file__).resolve().parents[1]),
              'script_sha256': digest(Path(__file__)),
              'module_sha256': digest(Path(__file__).resolve().parents[1] / 'global_timing/search_analysis.py'),
              'slurm_job_id': os.environ['SLURM_JOB_ID']}
    report['aggregate_by_source_commit'] = {
        commit: aggregate_search([row for row in analyzed if row['source_commit'] == commit])
        for commit in sorted({row['source_commit'] for row in analyzed})}
    report['heterogeneous_source_versions'] = len({row['source_commit'] for row in coverage}) > 1
    if cohort is not None:
        report['frozen_cohort'] = {'path': str(args.cohort.resolve()), 'sha256': digest(args.cohort),
            'policy': cohort['policy'], 'pending_declared_attempts': cohort['pending_declared_attempts'],
            'attempt_status_counts': cohort['attempt_status_counts']}
    save_json(output / 'summary.json', report)
    save_json(output / 'coverage.json', coverage)
    write_csv(output / 'methods.csv', methods)
    write_csv(output / 'predictions.csv', predictions)
    write_csv(output / 'gradient_diagnostics.csv', gradients)
    write_csv(output / 'failures_and_omissions.csv', failures)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for mode in pooled['modes']:
        rows = [r for r in predictions if r['mode'] == mode and r['normalized_actual_delta'] is not None]
        if rows:
            axes[0].scatter([r['normalized_predicted_delta'] for r in rows], [r['normalized_actual_delta'] for r in rows],
                            s=13, alpha=.45, label=mode)
    axes[0].set(xlabel='Predicted delta / no-op delay', ylabel='Actual delta / no-op delay')
    if axes[0].collections:
        axes[0].legend(fontsize=7)
    modes = list(pooled['modes'])
    origins = sorted({r['selected_source'] for r in methods})
    offsets = [0] * len(modes)
    for origin in origins:
        counts = [pooled['modes'][mode]['final_selected_sources'].get(origin, 0) for mode in modes]
        axes[1].bar(modes, counts, bottom=offsets, label=origin)
        offsets = [a + b for a, b in zip(offsets, counts)]
    axes[1].set(ylabel='Selected final results (design count)')
    if origins:
        axes[1].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(output / 'prediction_and_selected_sources.png', dpi=160)
    fig.savefig(output / 'prediction_and_selected_sources.pdf')
    plt.close(fig)
    lines = ['# 保存搜索的质量审计', '', f'声明cohort：{args.cohort_label}。',
             f'已有记录{len(seen)}/{args.expected_inputs}个输入；分析{len(analyzed)}个；全量覆盖：{report["coverage_complete"]}。',
             f'原始状态：{report["case_status_counts"]}；审计失败：{report["audit_failure_count"]}。', '',
             '| 目标 | 分析设计数 | 按设计平均归一化MAE | 梯度请求成功/总数 | 最终选择来源 |', '|---|---:|---:|---|---|']
    for mode, result in pooled['modes'].items():
        value = result['mean_per_design_normalized_mae']
        lines.append(f'| {mode} | {result["design_count"]} | {value if value is not None else "N/A"} | '
                     f'{result["direct_gradient_successful"]}/{result["direct_gradient_requested"]} | {result["final_selected_sources"]} |')
    lines += ['', '预测来自逐次真实评估前，模型随反馈变化；这不是固定模型的候选排序测试或跨设计泛化证据。',
              '源码版本逐输入保留，并在summary.json中分别汇总；多版本总体统计不冒充单一最终版本的实验。',
              '使用冻结cohort时保留全部旧尝试；尚未产生终态记录的声明任务会使全量完成标记保持false。',
              '基线锚点、失败请求均不用于计算预测误差；没有实际变化的样本单列，不计作方向预测正确。',
              '每个设计等权汇总；零delay参考不做归一化。单种子、自适应样本及相关benchmark变体不支持显著性推断。',
              '最终选择来源描述保存的请求来源，不证明因果贡献；显式修复可能继承梯度提案，其直接parent来源另列。',
              '梯度CSV反映保存的抽样迭代，不宣称包含每一步；完整失败、预算、来源和原始报告hash均保留。']
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({key: report[key] for key in ['recorded_inputs', 'analyzed_inputs', 'coverage_complete', 'audit_failure_count']}))
    return int(report['audit_failure_count'] > 0)


if __name__ == '__main__':
    raise SystemExit(main())
