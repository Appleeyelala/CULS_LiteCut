#!/usr/bin/env python3
"""Supplementary four-design channel/repair controls after formal timing."""
import argparse
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def read(path):
    return json.loads(Path(path).read_text())


def signature(search):
    return [(o['choices'], o['status'],
             {k: o['metrics'][k] for k in ('delay_ps', 'area', 'po_arrivals_ps')} if o['status'] == 'PASS' else None)
            for o in search['observations']]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--smoke-run', type=Path, required=True)
    parser.add_argument('--reference-run', type=Path, required=True)
    parser.add_argument('--frozen-source', type=Path, required=True)
    parser.add_argument('--tool-run', type=Path, required=True)
    args = parser.parse_args()
    run, smoke, reference = [p.resolve() for p in (args.run, args.smoke_run, args.reference_run)]
    if not os.environ.get('SLURM_JOB_ID') or not all(p.is_relative_to('/data/wushangge') for p in (run, smoke, reference, args.frozen_source.resolve())):
        parser.error('Requires fresh committed /data snapshots and a Slurm GPU allocation.')
    from global_timing.common import digest, save_json
    from global_timing.profile_results import add_runtime_metrics, audit_profiles
    from global_timing.joint_results import matrix, paired
    protocol_path = run / 'src/config/gpu_joint_ablation_protocol.json'
    protocol = read(protocol_path)
    panel_path = run / 'src' / protocol['panel']
    if digest(panel_path) != protocol['panel_sha256']:
        raise ValueError('Frozen panel changed.')
    panel = read(panel_path)
    first = {}
    for row in panel['rows']:
        first.setdefault(tuple(row['and_bin']), row['index'])
    if list(first.values()) != protocol['indices']:
        raise ValueError('Supplementary selection is not the first row of each metadata bin.')
    selected_panel = dict(panel, rows=[r for r in panel['rows'] if r['index'] in protocol['indices']])
    inventory = Path(panel['inventory'])
    if digest(inventory) != panel['inventory_sha256']:
        raise ValueError('Frozen inventory changed.')
    save_json(run / 'input/ablation_preflight.json', {'protocol': protocol, 'protocol_sha256': digest(protocol_path),
        'panel_sha256': digest(panel_path), 'selected': selected_panel['rows'],
        'frozen_source': str(args.frozen_source.resolve()), 'frozen_source_commit': read(args.frozen_source.parent / 'run.json')['source_commit'],
        'validation_smoke_run': str(smoke), 'validation_reference_run': str(reference),
        'scope': 'Protocol written during formal execution; no supplementary results existed at selection. Main sixteen-design data/parameters unchanged.'})
    commands = [[sys.executable, '-B', str(run / 'src/scripts/run_tests.py'), '--output', str(run / 'output/ablation_tests.json')]]
    for name in ('test_ablations', 'test_pipeline', 'test_joint_optimization', 'test_surrogate', 'test_arrival'):
        commands[0] += ['--test', name]
    for command in commands:
        subprocess.run(command, cwd=run, check=True)

    # Same allocation, same real tools, both source versions: actual default
    # compatibility is checked rather than inferred from a changed source hash.
    refmeta = read(reference / 'run.json')
    if refmeta['state'] != 'PREPARED_NOT_SUBMITTED':
        raise ValueError('Reference validation workspace was already used.')
    item = next(r for r in read(inventory)['rows'] if r['index'] == 47)
    if digest(reference / 'input/design.aig') != item['sha256']:
        raise ValueError('Reference validation input is not the frozen c1355.')
    save_json(reference / 'input/profile_cases.json', [dict(item, staged_path=str(reference / 'input/design.aig'))])
    reference_output = reference / 'output/compatibility'
    reference_output.mkdir()
    refcommand = [sys.executable, '-B', str(run / 'src/scripts/profile_single_method.py'),
        '--run', str(reference), '--tool-run', str(args.tool_run), '--case-index', '47', '--seed', '20261112',
        '--evaluations', '4', '--variant', 'adaptive_batch_auto', '--source-root', str(args.frozen_source),
        '--output-base', str(reference_output), '--validation-workers', '2']
    refmeta.update(state='RUNNING', command=refcommand, slurm_job_id=os.environ['SLURM_JOB_ID'], experiment_executed=True,
                   actual_imported_source=str(args.frozen_source))
    (reference / 'run.json').write_text(json.dumps(refmeta, indent=2) + '\n')
    subprocess.run(refcommand, cwd=reference, check=True)
    refmeta.update(state='COMPLETED', exit_code=0)
    (reference / 'run.json').write_text(json.dumps(refmeta, indent=2) + '\n')

    common = ['--tool-run', str(args.tool_run), '--manifest', str(inventory), '--validation-workers', '2']
    smoke_command = [sys.executable, '-B', str(run / 'src/scripts/profile_single_method.py'), '--run', str(smoke),
                     *common, '--indices', '47', '--seed', '20261112', '--evaluations', '4',
                     '--variants', ','.join(protocol['variants'])]
    subprocess.run(smoke_command, cwd=smoke, check=True)
    smoke_parent = read(smoke / 'output/profile_summary.json')
    if smoke_parent['status'] != 'PASS' or len(smoke_parent['records']) != 4:
        raise ValueError('Four actual c1355 controls did not finish successfully.')
    actual_default = read(smoke / 'output/adaptive_batch_auto/case_0047/gradient_area_constraint/summary.json')
    frozen_default = read(reference_output / 'case_0047/gradient_area_constraint/summary.json')
    if signature(actual_default) != signature(frozen_default):
        raise ValueError('Default primary request vectors/status/actual metrics changed from frozen source.')
    save_json(run / 'output/default_compatibility.json', {'status': 'PASS',
        'frozen_source': str(args.frozen_source), 'current_source_commit': read(run / 'run.json')['source_commit'],
        'actual_default_summary_sha256': digest(smoke / 'output/adaptive_batch_auto/case_0047/gradient_area_constraint/summary.json'),
        'frozen_default_summary_sha256': digest(reference_output / 'case_0047/gradient_area_constraint/summary.json'),
        'scope': 'Actual same-allocation eight total logical requests across two independent source-version validations; these labels are not supplied to the study.'})
    smoke_rows, smoke_hashes = audit_profiles([smoke])
    save_json(run / 'output/smoke_audit.json', {'rows': smoke_rows, 'evidence_hashes': smoke_hashes})

    command = [sys.executable, '-B', str(run / 'src/scripts/profile_single_method.py'), '--run', str(run), *common,
        '--indices', ','.join(map(str, protocol['indices'])), '--seeds', ','.join(map(str, protocol['seeds'])),
        '--variants', ','.join(protocol['variants']), '--evaluations', '8', '--proposal-steps', '40',
        '--trajectories', '32', '--max-changes', '32', '--tool-timeout', '900', '--case-timeout', '1800']
    save_json(run / 'input/study_command.json', command)
    subprocess.run(command, cwd=run, check=True)
    parent = read(run / 'output/profile_summary.json')
    expected = set(itertools.product(protocol['indices'], protocol['seeds'], protocol['variants']))
    actual = [(r['index'], r['seed'], r['variant']) for r in parent['records']]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('Supplementary study coverage differs from the frozen protocol.')
    rows, hashes = audit_profiles([run])
    rows = add_runtime_metrics(rows)
    lookup = matrix(rows, selected_panel, protocol)
    results = {v: paired(lookup, selected_panel, protocol, protocol['primary_variant'], v)
               for v in protocol['variants'] if v != protocol['primary_variant']}
    output = run / 'output/ablation_analysis'
    output.mkdir()
    save_json(output / 'records.json', rows)
    save_json(output / 'paired_results.json', results)
    save_json(output / 'evidence_hashes.json', hashes)
    body = ['# 四设计补充消融', '',
        '本研究在正式面板运行期间按已有规模元数据选定四设计，先在同分配验证默认版本与冻结源码逐请求真实结果一致，再执行独立12个设计/seed组。'
        '每方法8请求，包含no-op、失败、修复。主16设计数据不回写；本子集不代替主面板，不建立未见泛化。', '',
        '| 移除项 | 完整可比seed对 | 主方法胜/平/负 | 主方法/移除后 delay几何均值 | 移除后/主方法中位E2E比 |',
        '|---|---:|---|---:|---:|']
    for variant, result in results.items():
        outcome = result['outcomes']
        body.append(f'| {variant} | {result["measured_area_feasible_seed_pairs"]}/{result["expected_seed_pairs"]} | '
                    f'{outcome.get("win",0)}/{outcome.get("tie",0)}/{outcome.get("loss",0)} | '
                    f'{result["geomean_delay_ratio_seed_pairs"]} | {result["median_shared_allocation_speed"]} |')
    body += ['', 'no-score保留真实Student准备成本，只移除局部分数delta；no-pairs同时禁用全代理显式pair系数和反馈pair描述量，DAG仍耦合端点；'
             'no-repair保留失败并把后续预算用于其他请求。差异是受控配置的真实结果，不等于精确梯度或独立电路总体上的因果证明。', '',
             '逐点完整安装、CEC、Liberty、实际时间与来源在records.json；所有原始hash在evidence_hashes.json。'
             '本研究未额外测native基线，内部time-to-quality参考只有no-op，不与主面板native参考的达标时间混合。', '']
    (output / 'REPORT.md').write_text('\n'.join(body))
    save_json(output / 'completion.json', {'status': 'EXECUTED_AND_AUDITED', 'expected_methods': len(expected),
        'reported_methods': len(rows), 'failed_methods': sum(r['status'] != 'PASS' for r in rows),
        'protocol_sha256': digest(protocol_path), 'scope': 'Small supplementary channel/repair controls; no claim that the default is superior.'})


if __name__ == '__main__':
    main()
