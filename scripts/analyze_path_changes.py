#!/usr/bin/env python3
"""Compare retained arrival traces for a frozen benchmark cohort, without mapping."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from global_timing.cohort import read_cohort
from global_timing.common import digest, save_json
from global_timing.evaluation import cec_passed, parse_timing
from global_timing.path_analysis import parse_paths
from global_timing.path_changes import aggregate_cases, analyze_case, endpoint_state, prediction_changes, references
from scripts.analyze_search_quality import write_csv
from scripts.analyze_timing_paths import evaluation_directory, netlist_body


def validate_measurements(summary, report, case_report_path):
    """Re-read CEC, exact timing, input/netlist hashes and transition arithmetic."""
    benchmark = Path(report['source_benchmark']).resolve()
    directory = case_report_path.resolve().parent
    provenance = summary['original']['provenance']
    original = Path(provenance['original']['path'])
    with original.open('rb') as stream:
        header = stream.readline().split()
    if header[0] != b'aig' or len(header) != 6 or int(header[3]) != 0:
        raise ValueError('Expected a combinational binary AIG interface.')
    pi_count = int(header[2])
    artifacts = {}

    def checked(path, expected=None):
        path = Path(path).resolve()
        actual = artifacts.get(str(path))
        if actual is None:
            actual = artifacts[str(path)] = digest(path)
        if expected is not None and actual != expected:
            raise ValueError('Saved evidence checksum changed: ' + str(path))
        return actual

    for name in ['original', 'library']:
        checked(provenance[name]['path'], provenance[name]['sha256'])
    measured, states = {}, {}
    for row in report['measurements']:
        if row['status'] != 'PASS':
            raise ValueError('Complete path audit contains a failed measurement.')
        path = Path(row['paths']).resolve()
        if not path.is_relative_to(directory) or row['paths'] in measured:
            raise ValueError('Duplicate or external trace in path audit.')
        work = path.parent
        for name in ['paths.json', 'result.json', 'cec.log', 'mapping.log', 'mapped.v']:
            checked(work / name)
        actual = json.loads((work / 'result.json').read_text())
        log = (work / 'mapping.log').read_text()
        if (actual['status'] != 'PASS' or actual['cec'] != 'PASS' or
                not cec_passed((work / 'cec.log').read_text())):
            raise ValueError('Retained CEC does not prove equivalence.')
        parsed = parse_timing(log)
        if any(actual[key] != value for key, value in parsed.items()):
            raise ValueError('Saved timing differs from its raw Liberty log.')
        trace = parse_paths(log, actual, pi_count)
        if trace != json.loads(path.read_text()):
            raise ValueError('Saved paths differ from the reconstructed raw trace.')
        checked(row['aig'], row['input_sha256'])
        checked(work / 'mapped.v', actual['mapped_sha256'])
        for name in ['original', 'library']:
            if actual['provenance'][name] != provenance[name]:
                raise ValueError('Observed trace uses another original input or library.')
        observer = actual['provenance']['abc']
        checked(observer['path'], observer['sha256'])
        checked(row['reference_abc']['path'], row['reference_abc']['sha256'])
        if (observer['sha256'] != row['observer_abc_sha256'] or
                actual['input_sha256'] != row['input_sha256'] or
                Path(row['source_benchmark']).resolve() != benchmark or
                row.get('tool_timeout_s', summary['config']['timeout_s']) != summary['config']['timeout_s'] or
                row['fresh_cec'] != 'PASS' or row['measured_metrics_match'] is not True or
                row['mapped_netlist_body_match'] is not True):
            raise ValueError('Path audit provenance/validation flags disagree.')
        state = endpoint_state(trace, actual)
        if (row['near_critical_path_count'] != len(trace['paths']) or
                sorted(row['critical_pairs']) != state['critical_pairs'] or
                any(row['timing'][key] != actual[key] for key in ['area', 'delay_ps', 'po_arrivals_ps'])):
            raise ValueError('Path audit measurements disagree with raw records.')
        body = hashlib.sha256(netlist_body(work / 'mapped.v').encode()).hexdigest()
        if body != row['mapped_reference_body_sha256']:
            raise ValueError('Observed mapped body differs from its reference hash.')
        measured[row['paths']] = (actual, row, body)
        states[row['paths']] = state
    expected = {(row['method'], row['index']): row for row in references(summary)}
    for link in report['selection_path_changes']:
        item = expected[(link['method'], link['index'])]
        reference = item['metrics']
        actual, row, body = measured[link['path_evidence']]
        if item['method'] == 'original':
            aig, mapped = original, benchmark / 'original/mapped.v'
        elif item['method'] == 'native_abc':
            aig, mapped = benchmark / 'native_abc/rewritten.aig', benchmark / 'native_abc/evaluation/mapped.v'
        else:
            aig = Path(item['observation']['aig'])
            mapped = evaluation_directory(aig, benchmark, reference['input_sha256']) / 'mapped.v'
        checked(aig, reference['input_sha256'])
        checked(mapped, reference['mapped_sha256'])
        if (reference['cec'] != 'PASS' or actual['input_sha256'] != reference['input_sha256'] or
                row['reference_abc'] != reference['provenance']['abc'] or
                any(actual[key] != reference[key] for key in
                    ['area', 'delay_ps', 'mapped_gates', 'po_arrivals_ps', 'po_rise_fall', 'critical_po_indices']) or
                hashlib.sha256(netlist_body(mapped).encode()).hexdigest() != body):
            raise ValueError('Successful observation differs from its retained mapping/path evidence.')
    return states, artifacts


def snapshot_reports(audit_runs, output):
    """Freeze completed per-case JSON files, including a live audit's prefix."""
    chosen, sources, invalid = {}, [], []
    if len(set(audit_runs)) != len(audit_runs):
        raise ValueError('Do not list an audit run twice.')
    for number, run in enumerate(audit_runs):
        run = run.resolve()
        snapshot = output / 'input_snapshots' / f'audit_{number:02d}'
        snapshot.mkdir(parents=True)
        metadata = (run / 'run.json').read_bytes()
        (snapshot / 'run.json').write_bytes(metadata)
        parsed_metadata = json.loads(metadata)
        files = sorted((run / 'output/arrival_paths').glob('*/summary.json'))
        seen = set()
        for path in files:
            raw = path.read_bytes()
            target = snapshot / (path.parent.name + '.json')
            target.write_bytes(raw)
            try:
                report = json.loads(raw)
                if report['status'] not in {'PASS', 'FAIL'}:
                    raise ValueError('Nonterminal per-case path report.')
                benchmark = str(Path(report['source_benchmark']).resolve())
            except (ValueError, KeyError) as exc:
                invalid.append({'path': str(path), 'snapshot': str(target), 'reason': str(exc)})
                continue
            if benchmark in seen:
                raise ValueError('Duplicate benchmark inside one audit run.')
            seen.add(benchmark)
            chosen[benchmark] = {'report': report, 'path': path, 'snapshot': str(target),
                                 'sha256': hashlib.sha256(raw).hexdigest(), 'audit_source': number}
        sources.append({'run': str(run), 'source_commit': parsed_metadata['source_commit'],
            'metadata_snapshot': str(snapshot / 'run.json'), 'metadata_sha256': hashlib.sha256(metadata).hexdigest(),
            'state_at_snapshot': parsed_metadata['state'], 'case_report_count': len(files)})
    return chosen, sources, invalid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--audit-runs', type=Path, nargs='+', required=True,
                        help='Declared priority order; last terminal report wins, including failures.')
    args = parser.parse_args()
    output = args.output.resolve()
    if not os.environ.get('SLURM_JOB_ID') or not output.is_relative_to('/data/wushangge'):
        parser.error('Analysis requires Slurm and a new /data output directory.')
    cohort, benchmark_sources = read_cohort(args.cohort)
    output.mkdir(parents=True, exist_ok=False)
    selected, audit_sources, invalid = snapshot_reports(args.audit_runs, output)
    attempts = {row['record']['index']: row for row in cohort['selected_attempts']}
    cases, coverage, observations, transitions, predictions, artifacts = [], [], [], [], [], {}
    used = set()
    for identity in cohort['cases']:
        index = identity['index']
        row = dict(identity, analysis_status='NOT_RECORDED')
        coverage.append(row)
        if identity['selected_attempt'] is None:
            continue
        attempt = attempts[index]
        record = attempt['record']
        row['benchmark_source_commit'] = benchmark_sources[attempt['source']]['source_commit']
        if identity['status'] != 'PASS':
            row.update(analysis_status='BENCHMARK_NOT_PASS', reason=record.get('reason'))
            continue
        summary_path = Path(record['summary']).resolve()
        benchmark = str(summary_path.parent)
        row['benchmark'] = benchmark
        if benchmark not in selected:
            row['analysis_status'] = 'PATH_REPORT_MISSING'
            continue
        saved = selected[benchmark]
        used.add(benchmark)
        row.update(path_report=str(saved['path']), path_report_sha256=saved['sha256'],
                   path_report_snapshot=saved['snapshot'], audit_source=saved['audit_source'])
        if saved['report']['status'] != 'PASS':
            row['analysis_status'] = 'PATH_AUDIT_FAIL'
            continue
        try:
            summary = json.loads(summary_path.read_text())
            states, hashes = validate_measurements(summary, saved['report'], saved['path'])
            analysis = analyze_case(summary, saved['report'], states)
            analysis['prediction_samples'] = prediction_changes(summary, analysis)
            observers = sorted({item['observer_abc_sha256'] for item in saved['report']['measurements']})
            if len(observers) != 1:
                raise ValueError('Case path evidence mixes observer binary versions.')
            row.update(analysis_status='ANALYZED', observer_abc_sha256=observers[0], summary=str(summary_path),
                       summary_sha256=digest(summary_path))
            case = dict(row, analysis=analysis)
            cases.append(case)
            save_json(output / f'case_{index:04d}.json', case)
            identity_fields = {key: row[key] for key in ['index', 'relative_path', 'benchmark_source_commit', 'observer_abc_sha256']}
            # Observation indices belong to each search; keep benchmark indices separate.
            identity_fields['benchmark_index'] = identity_fields.pop('index')
            observations.extend(dict(identity_fields, **value) for value in analysis['observations'])
            transitions.extend(dict(identity_fields, **value) for value in analysis['transitions'])
            predictions.extend(dict(identity_fields, **value) for value in analysis['prediction_samples'])
            artifacts.update(hashes)
        except Exception as exc:
            row.update(analysis_status='ANALYSIS_ERROR', error=f'{type(exc).__name__}: {exc}')
    report = {'cohort': str(args.cohort.resolve()), 'cohort_sha256': digest(args.cohort),
        'expected_inputs': cohort['expected_inputs'], 'recorded_inputs': cohort['recorded_inputs'],
        'benchmark_coverage_complete': cohort['coverage_complete'], 'analyzed_inputs': len(cases),
        'all_selected_pass_cases_analyzed': len(cases) == cohort['case_status_counts'].get('PASS', 0),
        'coverage_counts': dict(Counter(row['analysis_status'] for row in coverage)),
        'audit_failure_count': sum(row['analysis_status'] == 'ANALYSIS_ERROR' for row in coverage),
        'aggregate': aggregate_cases(cases), 'audit_sources': audit_sources, 'invalid_report_snapshots': invalid,
        'unselected_path_reports': sorted(set(selected) - used),
        'path_report_selection_policy': 'last_declared_terminal_report_for_exact_selected_benchmark_path',
        'slurm_job_id': os.environ['SLURM_JOB_ID'],
        'analysis_code': {name: digest(Path(__file__).resolve().parents[1] / name) for name in
            ['scripts/analyze_path_changes.py', 'global_timing/path_changes.py', 'global_timing/path_analysis.py',
             'global_timing/search_analysis.py']}}
    report['aggregate_by_version'] = [dict(benchmark_source_commit=source, observer_abc_sha256=observer,
        aggregate=aggregate_cases([case for case in cases if case['benchmark_source_commit'] == source and
                                   case['observer_abc_sha256'] == observer]))
        for source, observer in sorted({(case['benchmark_source_commit'], case['observer_abc_sha256']) for case in cases})]
    save_json(output / 'summary.json', report)
    save_json(output / 'coverage.json', coverage)
    save_json(output / 'evidence_hashes.json', artifacts)
    write_csv(output / 'observations.csv', observations)
    write_csv(output / 'transitions.csv', transitions)
    write_csv(output / 'prediction_path_changes.csv', predictions)
    write_csv(output / 'prediction_error_by_endpoint_change.csv', report['aggregate']['prediction_error_by_endpoint_change'])
    write_csv(output / 'coverage.csv', coverage)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    methods = report['aggregate']['methods']
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    names = list(methods)
    axes[0].bar(names, [methods[name]['critical_po_changes_vs_noop'] for name in names], label='Critical PO set changes')
    axes[0].bar(names, [methods[name]['observed_pair_changes_with_same_po_set_vs_noop'] for name in names],
                bottom=[methods[name]['critical_po_changes_vs_noop'] for name in names], label='PI changes, same PO set')
    axes[0].set(ylabel='Successful non-anchor observations')
    if names:
        axes[0].legend(fontsize=7)
    for method in names:
        rows = [row for row in observations if row['method'] == method and row['is_saved_final_selection'] and
                row['vs_noop_delay_improvement_pct'] is not None]
        if rows:
            axes[1].scatter([row['benchmark_index'] for row in rows],
                            [row['vs_noop_delay_improvement_pct'] for row in rows], s=14, alpha=.5, label=method)
    axes[1].set(xlabel='Benchmark index', ylabel='Saved final delay improvement vs no-op (%)')
    axes[1].axhline(0, color='gray', linewidth=.5)
    if axes[1].collections:
        axes[1].legend(fontsize=6)
    axes[0].tick_params(axis='x', rotation=70)
    fig.tight_layout()
    for extension in ['png', 'pdf']:
        fig.savefig(output / ('path_changes.' + extension), dpi=160)
    plt.close(fig)
    lines = ['# 已保存的关键端点与arrival链变化', '',
        f'冻结清单：{cohort["recorded_inputs"]}/{cohort["expected_inputs"]}有终态记录；路径分析{len(cases)}个成功设计。',
        f'覆盖状态：{report["coverage_counts"]}；源码/observer分层见summary.json。', '',
        '| 方法 | 成功非锚点观察 | PO集合变化 | PO不变但观测PI变化 | 跨失败请求的成功转移 |',
        '|---|---:|---:|---:|---:|']
    for name, value in methods.items():
        lines.append(f'| {name} | {value["successful_nonanchor_observations"]} | '
            f'{value["critical_po_changes_vs_noop"]} | {value["observed_pair_changes_with_same_po_set_vs_noop"]} | '
            f'{value["transitions_spanning_failed_requests"]} |')
    lines += ['', '只重读已有证据，不执行rewrite、CEC、mapping或新的数值预测。逐份复核原日志、CEC、AIG/netlist哈希、全部PO与逐arc路径重构。',
        'original→no-op序列化对照单列；搜索相对no-op比较。失败请求没有数值路径标签，转移明确跨越了多少失败请求。',
        '关键PO取2e-5 ps绝对容差内集合，近关键阈值为global arrival的95%。每个PO只保留一条代表最大arrival链，未枚举所有并列链。',
        'PI/PO位置索引可比较；mapped节点ID不跨映射比较。源/端点或相位变化不等于证明唯一逻辑路径身份切换，集合不变也不证明内部链不变。',
        '已保存的最终选点仅标记gradient与确定性基线；random有两种目标排序，不擅自选择一个作为统一final。',
        '完成的单例报告可来自仍在运行的审计；当前快照缺失不记作benchmark失败。其他重跑即使输入hash相同也不代替选定benchmark的路径证据。',
        '观察次数彼此相关，只作描述统计，不支持统计显著性或跨设计泛化推断。']
    lines += ['', '## 评估前预测误差与端点变化', '',
        '| 目标 | 端点变化类别 | 设计数 | 预测样本 | 按设计平均归一化MAE |', '|---|---|---:|---:|---:|']
    for row in report['aggregate']['prediction_error_by_endpoint_change']:
        lines.append(f'| {row["mode"]} | {row["endpoint_change"]} | {row["design_count"]} | '
                     f'{row["sample_count"]} | {row["mean_per_design_normalized_mae"]} |')
    lines += ['', '预测先核验训练标签仅来自更早成功观察，再按本次搜索的method/index/input hash连接路径；失败与基线锚点不计入误差。',
        '各类别内先按设计平均，再跨设计等权汇总；零delay不归一化。类别可能包含不同设计与相同设计的相关观察，差异不证明因果关系。',
        'SAME_OBSERVED_ENDPOINT_PAIR_SET只表示代表PI/PO对相同，不表示内部路径或映射未变。']
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({key: report[key] for key in ['analyzed_inputs', 'coverage_counts', 'audit_failure_count']}))
    return int(report['audit_failure_count'] > 0 or bool(invalid))


if __name__ == '__main__':
    raise SystemExit(main())
