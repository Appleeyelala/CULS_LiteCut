"""Tables and standalone figures derived only from saved real observations."""
import csv
import json
from pathlib import Path

import torch
import numpy as np

from .common import save_json
from .evaluation import failed_comparisons, native_baseline_metrics
from .surrogate import scalar_objective


def _write_csv(path, rows, fields):
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(benchmark: Path, output: Path):
    summary = json.loads((benchmark / 'summary.json').read_text())
    output.mkdir(parents=True, exist_ok=False)
    original = summary['original']
    baseline = summary['replay_noop']['metrics']
    comparison, observations = [], []

    def joint_value(metrics):
        return float(scalar_objective(torch.tensor(metrics['po_arrivals_ps'], dtype=torch.float64),
            torch.tensor(metrics['area'], dtype=torch.float64), baseline['delay_ps'], baseline['area'],
            'joint', area_weight=summary['config']['area_weight']))

    def add(method, metrics, rows=None, applied=None, runtime=None, direct_gradients=None):
        rows = rows or []
        def percent(value, reference, sign=1):
            return sign * 100 * (value - reference) / reference if reference > 0 else None
        comparison.append({'method': method, 'delay_ps': metrics['delay_ps'], 'area': metrics['area'], 'adp': metrics['adp'],
            'delay_improvement_pct': percent(metrics['delay_ps'], original['delay_ps'], -1),
            'delay_improvement_vs_noop_pct': percent(metrics['delay_ps'], baseline['delay_ps'], -1),
            'area_change_pct': percent(metrics['area'], original['area']),
            'area_change_vs_noop_pct': percent(metrics['area'], baseline['area']),
            'area_constraint_feasible': metrics['area'] <= baseline['area'] * summary['config']['area_limit_ratio'] + 1e-8,
            'evaluations_including_baseline': len(rows) or 1, 'failed_requests': sum(r['status'] != 'PASS' for r in rows),
            'measured_evaluation_runtime_s': sum(r.get('runtime_s', 0) for r in rows) if rows else None,
            'exclusive_method_runtime_s': runtime,
            'direct_gradient_evaluations': direct_gradients,
            'actual_joint_objective': joint_value(metrics),
            'applied_rewrites': applied, 'cec': metrics['cec'],
            'critical_po_indices': json.dumps(metrics['critical_po_indices'])})

    add('Original AIG', original, applied=0, runtime=original.get('runtime_s'))
    add('Replay no-op', baseline, [summary['replay_noop']], 0)
    native = native_baseline_metrics(summary)
    if native is not None:
        add('Native ABC rewrite', native, runtime=summary.get('exclusive_native_runtime_s'))
    failures = failed_comparisons(summary)
    save_json(output / 'method_failures.json', {'failed_comparisons': failures,
        'native_abc_attempt': summary.get('native_abc_attempt'),
        'scope': 'Failed methods have no ranked numeric substitute; original and no-op remain separate controls.'})
    for name in ['greedy_litecut', 'independent_regions']:
        item = summary[name]
        # Support the first historical development run without rewriting it.
        rows = item.get('observations', [item])
        best = item.get('best', item if item.get('status') == 'PASS' else None)
        observations.extend(dict(row, report_method=name) for row in rows)
        if best:
            add(name, best['metrics'], rows, best.get('applied_rewrites'), item.get('exclusive_method_runtime_s'))
    random_rows = summary['random_search']
    observations.extend(dict(row, report_method='random_search') for row in random_rows)
    feasible = [row for row in random_rows if row['status'] == 'PASS' and
                row['metrics']['area'] <= baseline['area'] * summary['config']['area_limit_ratio'] + 1e-8]
    if feasible:
        best = min(feasible, key=lambda row: row['metrics']['delay_ps'])
        add('random_search_area_constraint', best['metrics'], random_rows, best.get('applied_rewrites'), summary.get('exclusive_random_runtime_s'))
    successful = [row for row in random_rows if row['status'] == 'PASS']
    if successful:
        best = min(successful, key=lambda row: joint_value(row['metrics']))
        add('random_search_joint', best['metrics'], random_rows, best.get('applied_rewrites'), summary.get('exclusive_random_runtime_s'))
    for mode, item in summary['gradient_search'].items():
        best, rows = item['best'], item['observations']
        add('gradient_' + mode, best['metrics'], rows, best.get('applied_rewrites'), item.get('exclusive_method_runtime_s'),
            item.get('gradient_proposal_evaluations'))
        observations.extend(dict(row, report_method='gradient_' + mode) for row in rows)
    _write_csv(output / 'comparison.csv', comparison, list(comparison[0]))
    records, predictions, previous_choices = [], [], {}
    for row in observations:
        metrics = row.get('metrics', {})
        choices = row.get('choices', [])
        previous = previous_choices.get(row['report_method'], [0] * len(choices))
        record = {'method': row['report_method'], 'index': row.get('index'), 'status': row['status'],
                  'delay_ps': metrics.get('delay_ps'), 'area': metrics.get('area'), 'adp': metrics.get('adp'),
                  'actual_joint_objective': joint_value(metrics) if row['status'] == 'PASS' else None,
                  'applied_rewrites': row.get('applied_rewrites'), 'runtime_s': row.get('runtime_s'),
                  'active_decisions': sum(choice != 0 for choice in choices),
                  'requested_decisions_changed_from_previous': sum(a != b for a, b in zip(choices, previous)),
                  'proposal_source': (row.get('prediction_before_evaluation') or {}).get('source'),
                  'near_critical_po_indices_within_5pct': json.dumps([i for i, value in enumerate(metrics.get('po_arrivals_ps', []))
                      if value >= .95 * metrics.get('delay_ps', float('inf'))]),
                  'critical_po_indices': json.dumps(metrics.get('critical_po_indices')),
                  'native_trace_endpoints': json.dumps(metrics.get('abc_native_trace_endpoints')),
                  'error': row.get('error')}
        records.append(record)
        previous_choices[row['report_method']] = choices
        prediction = row.get('prediction_before_evaluation') or {}
        if row['status'] == 'PASS' and 'po_arrivals_ps' in prediction:
            actual = np.asarray(metrics['po_arrivals_ps'])
            predicted = np.asarray(prediction['po_arrivals_ps'])
            predictions.append({'method': row['report_method'], 'index': row['index'], 'source': prediction.get('source'),
                'predicted_global_delta_ps': float(max(predicted) - baseline['delay_ps']),
                'actual_global_delta_ps': float(metrics['delay_ps'] - baseline['delay_ps']),
                'endpoint_mae_ps': float(np.abs(actual - predicted).mean()),
                'predicted_area': prediction['area'], 'actual_area': metrics['area'],
                'training_observations': prediction.get('training_observations')})
            indices = prediction.get('training_observation_indices')
            if indices is not None and any(index >= row['index'] for index in indices):
                raise ValueError('Prediction provenance includes its own or a future evaluation label.')
    if records:
        _write_csv(output / 'observations.csv', records, list(records[0]))
    if predictions:
        _write_csv(output / 'prequential_predictions.csv', predictions, list(predictions[0]))
    gradients = []
    for path in sorted(benchmark.glob('gradient_*/gradients_*.json')):
        for row in json.loads(path.read_text()):
            gradients.append({'method': path.parent.name, 'round': path.stem, 'restart': row.get('restart'),
                              'step': row.get('step'), 'gradient_norm': row.get('gradient_norm'),
                              'objective': row.get('objective'), 'projection_changed': row.get('projection_changed_argmax')})
    if gradients:
        _write_csv(output / 'gradients.csv', gradients, list(gradients[0]))
    diagnostics = {'prediction_samples': len(predictions), 'gradient_records': len(gradients),
                   'scope': 'Predictions saved before each new label; adaptive within-design search, not held-out-design generalization.',
                   'runtime_scope': 'Evaluation runtimes exclude calibration/search overhead; complete elapsed time is in run/Slurm records.',
                   'trace_scope': 'Critical PO sets and ABC native trace endpoints; no cross-map logical ID alignment.'}
    diagnostics['critical_po_set_changed_observations'] = sum(row['status'] == 'PASS' and
        row['metrics']['critical_po_indices'] != baseline['critical_po_indices'] for row in observations)
    diagnostics['unsuccessful_deterministic_methods'] = [name for name in ['greedy_litecut', 'independent_regions']
        if summary[name].get('status') == 'FAIL']
    diagnostics['unsuccessful_comparisons'] = failures
    diagnostics['total_runtime_including_preparation_s'] = summary.get('total_runtime_including_preparation_s')
    diagnostics['preparation_runtime_s'] = summary.get('preparation_runtime_s')
    if predictions:
        x = np.array([r['predicted_global_delta_ps'] for r in predictions])
        y = np.array([r['actual_global_delta_ps'] for r in predictions])
        diagnostics['global_delta_mae_ps'] = float(np.mean(np.abs(x - y)))
        diagnostics['global_delta_pearson'] = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 1e-12 and np.std(y) > 1e-12 else None
    save_json(output / 'diagnostics.json', diagnostics)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for method in sorted({row['method'] for row in records if row['method'].startswith(('gradient_', 'random_'))}):
        rows = [r for r in records if r['method'] == method]
        best, curve = baseline['delay_ps'], []
        for row in rows:
            if row['status'] == 'PASS' and row['area'] <= baseline['area'] * summary['config']['area_limit_ratio'] + 1e-8:
                best = min(best, row['delay_ps'])
            curve.append(best)
        axes[0].plot(range(1, len(curve) + 1), curve, label=method)
    axes[0].set(xlabel='Actual request budget (failures included)', ylabel='Best area-feasible delay (ps)')
    axes[0].legend(fontsize=7)
    if predictions:
        axes[1].scatter([r['predicted_global_delta_ps'] for r in predictions],
                        [r['actual_global_delta_ps'] for r in predictions], s=12, alpha=.6)
    axes[1].set(xlabel='Predicted global delay change (ps)', ylabel='Actual global delay change (ps)')
    fig.tight_layout()
    fig.savefig(output / 'convergence_and_prediction.png', dpi=160)
    fig.savefig(output / 'convergence_and_prediction.pdf')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for method in ['random_search', 'gradient_joint']:
        rows = [r for r in records if r['method'] == method]
        if not rows:
            continue
        best, curve = joint_value(baseline), []
        for row in rows:
            if row['status'] == 'PASS':
                best = min(best, row['actual_joint_objective'])
            curve.append(best)
        axes[0].plot(range(1, len(curve) + 1), curve, label=method)
    axes[0].set(xlabel='Actual request budget (failures included)', ylabel='Best actual joint objective')
    axes[0].legend(fontsize=7)
    for method in sorted({r['method'] for r in gradients}):
        values = [r['gradient_norm'] for r in gradients if r['method'] == method and r['gradient_norm'] is not None]
        axes[1].plot(range(len(values)), values, label=method)
    axes[1].set(xlabel='Recorded gradient step (rounds and restarts concatenated)', ylabel='Selection gradient norm')
    if gradients:
        axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / 'joint_objective_and_gradients.png', dpi=160)
    fig.savefig(output / 'joint_objective_and_gradients.pdf')
    plt.close(fig)
    lines = ['# 真实优化运行报告', '', '数值仅来自对应运行的ABC结果；未通过的组合不进入性能排名。', '',
             '| 方法 | Delay ps | Area | ADP | Delay改善% | 评估数 | 失败数 |', '|---|---:|---:|---:|---:|---:|---:|']
    for row in comparison:
        improvement = 'N/A' if row['delay_improvement_pct'] is None else f"{row['delay_improvement_pct']:.3f}"
        lines.append(f"| {row['method']} | {row['delay_ps']:.6f} | {row['area']:.6f} | {row['adp']:.3f} | {improvement} | {row['evaluations_including_baseline']} | {row['failed_requests']} |")
    if failures:
        lines += ['', '未完成的独立对照/方法：' + ', '.join(failures) + '。完整失败记录见method_failures.json；数值表不填补其缺失指标。']
    lines += ['', '预测评估为自适应搜索中的预先预测，不代表跨设计泛化。基线锚定不计作预测准确性的证明。',
              '面积约束参考no-op replay面积；所有失败、warmup和显式修复都计入实际搜索预算。',
              '原始输入与no-op replay分别保留；Native ABC未导出rewrite数量，不用AIG节点减少量冒充rewrite次数。',
              '收敛图左侧是每种搜索采样集内的最佳面积可行delay；joint最终选点按联合目标，可能不同于其采样集内最小delay点。',
              'CSV中evaluation runtime仅为真实评估耗时；完整运行时间应结合run.json与Slurm accounting，包含拟合和搜索。']
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    return comparison
