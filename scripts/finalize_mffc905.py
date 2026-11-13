#!/usr/bin/env python3
"""Complete the mandated delivery from audited files; never run EDA or models."""
import argparse
from collections import Counter
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--development', action='store_true', help='Validate supplements only on an already audited developer run.')
    parser.add_argument('--delivery', type=Path, help='Existing audited delivery; required for developer validation.')
    args = parser.parse_args()
    run, experiment = args.run.resolve(), args.experiment.resolve()
    if not os.environ.get('SLURM_JOB_ID') or not all(p.is_relative_to('/data/wushangge') for p in [run, experiment]):
        parser.error('Use a committed /data Slurm analysis run.')
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from global_timing.common import save_json
    from global_timing.mffc_delivery import installation_details, named_tradeoffs, shape_summary, tool_cost_summary
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    old = args.delivery.resolve() if args.delivery else experiment / 'output/benchmark905_delivery'
    if not old.is_relative_to('/data/wushangge'):
        raise ValueError('Audited delivery must be local data evidence.')
    completion = json.loads((old / 'analysis_completion.json').read_text())
    if completion['status'] != 'AUDITED_AND_REPORTED' or completion['formal'] == args.development:
        raise ValueError('Wrong audit terminal status or formal/developer scope.')
    prefix = 'DEVELOPMENT' if args.development else 'BENCHMARK905'
    for relative, expected in completion['artifacts_sha256'].items():
        path = (old / relative).resolve()
        if not path.is_relative_to(old) or sha(path) != expected:
            raise ValueError('Original audited delivery changed: ' + relative)
    rows = json.loads((old / 'audited_records.json').read_text())
    summary = json.loads((old / (prefix + '_SUMMARY.json')).read_text())
    preflight = json.loads((experiment / 'input/mffc905_preflight.json').read_text())
    before = copy.deepcopy(summary)
    expected_inputs = 3 if args.development else 905
    expected = {(r['index'], method) for r in rows for method in ['A', 'B', 'Ours']}
    actual = {(r['index'], r['method']) for r in rows}
    if len(rows) != expected_inputs * 3 or len(actual) != len(rows) or actual != expected:
        raise ValueError('Incomplete/duplicate A/B/Ours coverage for requested scope.')
    job = str(json.loads((experiment / 'run.json').read_text())['slurm_job_id'])
    account = subprocess.check_output(['sacct', '-j', job,
        '--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocTRES', '-n', '-P'], text=True)
    fields = next((line.split('|') for line in account.splitlines() if line.split('|')[0] == job), None)
    if fields is None or fields[1:3] != ['COMPLETED', '0:0']:
        raise RuntimeError('Original experiment is not a confirmed successful terminal job; do not rerun it automatically.')
    accounting = {'job_id': job, 'state': fields[1], 'exit_code': fields[2],
                  'elapsed_seconds': int(fields[3]), 'allocated_tres': fields[4], 'raw': account}
    output = run / 'output' / ('supplement_' + experiment.name if args.development else 'final_mffc905_delivery')
    output.mkdir(exist_ok=False)
    additional_hashes, details = {}, []
    def optional(path):
        if not path.exists():
            return None
        additional_hashes[str(path)] = sha(path)
        try:
            return json.loads(path.read_text())
        except (ValueError, OSError):
            return None
    for row in rows:
        if row['status'] == 'PASS':
            row['timing_model'] = 'ABC_SCL_LIBERTY'
        if row['method'] != 'Ours':
            continue
        work = Path(row['detail_path'])
        config_path = work / 'config.json'
        if config_path.exists():
            config = optional(config_path)
            if config is not None:
                for field, tool in [('input_sha256', None), ('gpuls_sha256', 'gpuls'),
                                    ('checkpoint_sha256', 'checkpoint'), ('native_library_sha256', 'native_library')]:
                    expected_hash = row['input_sha256'] if tool is None else preflight['tools'][tool]['sha256']
                    if config[field] != expected_hash:
                        raise ValueError('Actual per-method asset identity differs from frozen inputs/tools: ' + field)
            elif row['status'] == 'PASS':
                raise ValueError('Successful Ours method has no readable configuration.')
        requests = []
        for directory in sorted(list((work / 'replay_noop').glob('eval_*')) + list((work / 'optimization').glob('eval_*'))):
            requests.append({'observation': optional(directory / 'observation.json'),
                             'replay': optional(directory / 'rewrite/replay.json')})
        data = installation_details(requests)
        details.append({'index': row['index'], **data})
        row.update(complete_installation_pass=data['installation_states'].get('PASS', 0),
                   complete_installation_fail=data['installation_states'].get('FAIL', 0),
                   complete_installation_unknown=data['installation_states'].get('UNKNOWN', 0),
                   nonempty_complete_installations=data['nonempty_complete_installations'],
                   rejection_statuses=data['rejection_statuses'],
                   accepted_rewrites_failed_partial=data['accepted_rewrites_in_failed_partial_installations'],
                   failed_request_inclusive_runtime_s=data['failed_request_inclusive_s']['sum'],
                   repair_request_inclusive_runtime_s=data['repair_request_inclusive_s']['sum'])
    supplement = {'installation_totals': {status: sum(d['installation_states'].get(status, 0) for d in details)
                                         for status in ['PASS', 'FAIL', 'UNKNOWN']},
                  'nonempty_complete_installations': sum(d['nonempty_complete_installations'] for d in details),
                  'accepted_rewrites_in_failed_partial_installations': sum(d['accepted_rewrites_in_failed_partial_installations'] for d in details),
                  'rejection_statuses': dict(sum((Counter(d['rejection_statuses']) for d in details), Counter())),
                  'failed_request_inclusive_s': sum(d['failed_request_inclusive_s']['sum'] or 0 for d in details),
                  'failed_request_clock_samples': sum(d['failed_request_inclusive_s']['n'] for d in details),
                  'repair_request_inclusive_s': sum(d['repair_request_inclusive_s']['sum'] or 0 for d in details),
                  'repair_request_clock_samples': sum(d['repair_request_inclusive_s']['n'] for d in details),
                  'missing_terminal_observations': sum(d['missing_terminal_observations'] for d in details),
                  'scope': 'Recorded complete/partial installation work and inclusive request costs; never new timing labels or exclusive fractions.'}
    shape = shape_summary(rows)
    tradeoffs = {baseline: {'common_feasible_n': summary['comparisons'][baseline]['common_three_valid_feasible_pairs'],
                           **named_tradeoffs(summary['comparisons'][baseline]['common_feasible_relation_grid'])}
                for baseline in ['A', 'B']}
    tool_costs = tool_cost_summary(rows)
    summary.update(delivery_supplement={'installations': supplement, 'gpu_shape': shape, 'named_tradeoffs': tradeoffs,
        'tool_costs': tool_costs,
        'experiment_accounting': accounting, 'analysis_source_commit': json.loads((run / 'run.json').read_text())['source_commit'],
        'original_delivery': str(old), 'scope': 'Read-only supplementary accounting; original comparisons/tolerances and original evidence remain unchanged.'})
    if not args.development:
        audit = json.loads((experiment / 'src/provenance/mffc_gpu_freeze_audit.json').read_text())
        for item in audit['identical_runtime_source_files']:
            if sha(experiment / 'src' / item['path']) != item['sha256']:
                raise ValueError('Immutable experiment source changed after freeze: ' + item['path'])
        summary['delivery_supplement']['immutable_source_files_verified'] = len(audit['identical_runtime_source_files'])
    # No recalculation or outcome-dependent selection changes primary statistics.
    if any(summary[k] != before[k] for k in before):
        raise ValueError('Delivery enrichment changed an original result statistic.')
    columns = sorted({k for r in rows for k in r})
    with (output / (prefix + '_RESULTS.csv')).open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in r.items()} for r in rows)
    save_json(output / (prefix + '_SUMMARY.json'), summary)
    save_json(output / 'installation_details.json', details)
    save_json(output / 'supplementary_evidence_hashes.json', additional_hashes)
    save_json(output / 'original_analysis_completion.json', completion)
    shutil.copytree(old / 'figures', output / 'figures')
    text = (old / (prefix + '_RESULTS.md')).read_text()
    supplement_text = ['', '## 完整安装、修复和实际DAG计算补充', '',
        f"完整安装PASS/FAIL/UNKNOWN为{supplement['installation_totals']}；其中非空完整组合{supplement['nonempty_complete_installations']}个。失败部分安装中确实接受{supplement['accepted_rewrites_in_failed_partial_installations']}个rewrite，仅计实际工作量，不给这些部分电路成功timing标签。拒绝状态：{supplement['rejection_statuses']}。", '',
        f"失败请求已有{supplement['failed_request_clock_samples']}个时钟，inclusive合计{supplement['failed_request_inclusive_s']:.4f}s；repair已有{supplement['repair_request_clock_samples']}个，inclusive合计{supplement['repair_request_inclusive_s']:.4f}s。缺终态观察{supplement['missing_terminal_observations']}个。共享replay/并发验证会重叠，这些不是独占wall占比。", '',
        '| 比较 | 共同可行分母 | delay更好且不更慢 | 更快且delay不差 | 更快但质量下降 | 更慢但质量改善 |',
        '|---|---:|---:|---:|---:|---:|']
    for baseline, values in tradeoffs.items():
        supplement_text.append(f"| {baseline} | {values['common_feasible_n']} | {values['delay_better_and_not_slower']} | {values['faster_and_delay_not_worse']} | {values['faster_but_quality_worse']} | {values['slower_but_quality_better']} |")
    supplement_text += ['', '前两列可能重叠；原九格关系保持不变。', '',
        f"原始正式作业{job}为{accounting['state']}/{accounting['exit_code']}，Slurm elapsed={accounting['elapsed_seconds']}s，资源{accounting['allocated_tres']}；它含套件准备/分析，不是单方法latency，也不等于所有并发阶段之和。", '',
        f"有实际CUDA原生传播的Ours方法点{shape['observed_cuda_method_points']}个。下表仅描述该子集的结构与事件成本；不构成同算法CPU/GPU因果对照。", '',
        '| 结构字段 / 区间 | 方法点 | MFFC中位数 | DAG层数中位 | 最大同层宽度中位 | 原生GPU内核合计中位ms | 完整E2E中位s |',
        '|---|---:|---:|---:|---:|---:|---:|']
    def number(value):
        return 'N/A' if value is None else f'{value:.4f}'
    for field, groups in shape['cuda_shape_bands'].items():
        for label, values in groups.items():
            stats = values['recorded']
            supplement_text.append(f"| {field}/{label} | {values['method_points']} | {number(stats['regions']['median'])} | {number(stats['region_dag_layers']['median'])} | {number(stats['max_layer_width']['median'])} | {number(stats['native_kernel_ms']['median'])} | {number(stats['end_to_end_runtime_s']['median'])} |")
    supplement_text += ['', '原生事件不包括其他Torch聚合、上下文初始化、CPU特征/Student和EDA；不同调用的batch可不同。结构分组为描述性补充，来源/AND主分组与所有胜负/时间容差仍使用冻结协议。完整逐点安装明细见`installation_details.json`，完整设备/结构分组见SUMMARY的`delivery_supplement`。', '']
    supplement_text += ['', '## CEC、Mapping与Replay的实际记录成本', '',
        '包含成功、失败及超时方法点内已保存的命令。每点时间为该类命令的inclusive和；有并发验证时会重叠，不能相加当独占wall占比。时间n只统计有记录的点，未出现该工具或时钟缺失不补零。调用数为已保存命令数，损坏命令文件另列。', '',
        '| 方法 / 工具 | 已记录调用数 | 有时间方法点 | 每点inclusive中位s | P90 s | 总inclusive s | 有调用但无时钟点 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for method, data in tool_costs['methods'].items():
        for category, cost in data['tools'].items():
            clock, count = cost['per_point_inclusive_s'], cost['recorded_command_counts']
            supplement_text.append(f"| {method}/{category} | {number(count['sum'])} | {clock['n']} | {number(clock['median'])} | {number(clock['p90'])} | {number(clock['sum'])} | {cost['points_with_recorded_commands_but_no_clock']} |")
    supplement_text += ['', '未读/损坏的命令文件数：' + str({method: data['unreadable_command_files'] for method, data in tool_costs['methods'].items()}) + '。完整冷wall使用主表，不以这些重叠分项代替。', '']
    text += '\n'.join(supplement_text)
    (output / (prefix + '_RESULTS.md')).write_text(text)
    save_json(output / 'delivery_completion.json', {'status': 'COMPLETE_AUDITED_DELIVERY', 'formal': not args.development,
        'designs': expected_inputs, 'method_points': len(rows), 'experiment_source_commit': summary['source_commit'],
        'experiment_accounting': accounting, 'original_statistics_unchanged': True,
        'artifacts_sha256': {str(p.relative_to(output)): sha(p) for p in output.rglob('*') if p.is_file()}})
    print(json.dumps({'status': 'COMPLETE_AUDITED_DELIVERY', 'formal': not args.development,
                      'method_points': len(rows), 'output': str(output)}))


if __name__ == '__main__':
    main()
