#!/usr/bin/env python3
"""Assemble reviewed, terminal study evidence; no EDA, inference or fitting."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def read(path):
    return json.loads(Path(path).read_text())


def accounting(path):
    metadata = read(path / 'run.json')
    job = str(metadata['slurm_job_id'])
    raw = subprocess.check_output(['sacct', '-j', job, '--format=JobIDRaw,State,ExitCode,Elapsed,MaxRSS', '-n', '-P'], text=True)
    parent = next(line.split('|') for line in raw.splitlines() if line.split('|')[0] == job)
    if parent[1:3] != ['COMPLETED', '0:0']:
        raise ValueError('Required evidence job is not successfully terminal: ' + job)
    return {'job_id': job, 'source_commit': metadata['source_commit'], 'raw': raw}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--formal-run', type=Path, required=True)
    parser.add_argument('--formal-analysis', type=Path, required=True)
    parser.add_argument('--ablation-run', type=Path, required=True)
    args = parser.parse_args()
    run, formal, analysis, ablation = [p.resolve() for p in
        (args.run, args.formal_run, args.formal_analysis, args.ablation_run)]
    if not os.environ.get('SLURM_JOB_ID') or not all(p.is_relative_to('/data/wushangge') for p in (run, formal, analysis, ablation)):
        parser.error('Requires new committed-source /data analysis workspace and Slurm.')
    from global_timing.common import digest, save_json
    from global_timing.joint_results import matrix, paired
    from global_timing.joint_plots import make_figures
    output = run / 'output/delivery'
    output.mkdir(exist_ok=False)
    accounts = {key: accounting(path) for key, path in
                [('formal', formal), ('analysis', analysis), ('ablations', ablation)]}
    source = analysis / 'output/final_synthesis'
    completion = read(source / 'analysis_completion.json')
    if completion['status'] != 'ANALYSIS_COMPLETE':
        raise ValueError('Formal analysis completion absent.')
    # Reuse its completed raw audit, verifying the actual artifacts it signed.
    # Do not repeat tens of GB of raw trace scans or alter the original output.
    for relative, checksum in completion['artifacts_sha256'].items():
        if digest(source / relative) != checksum:
            raise ValueError('Completed analysis artifact changed: ' + relative)
    study = ablation / 'output/ablation_analysis'
    if read(study / 'completion.json')['status'] != 'EXECUTED_AND_AUDITED':
        raise ValueError('Supplementary raw audit incomplete.')
    if read(ablation / 'output/default_compatibility.json')['status'] != 'PASS':
        raise ValueError('Default compatibility not proven.')
    tests = read(ablation / 'output/ablation_tests.json')
    if not tests['successful']:
        raise ValueError('Channel/repair tests did not pass.')
    rows, summary = read(source / 'audited_records.json'), read(source / 'statistics.json')
    original_provenance = read(source / 'provenance.json')
    if (Path(original_provenance['experiment_run']).resolve() != formal or
            original_provenance['experiment_source_commit'] != read(formal / 'run.json')['source_commit']):
        raise ValueError('Completed analysis belongs to a different formal experiment.')
    protocol = read(formal / 'src/config/gpu_joint_final_protocol.json')
    panel = read(formal / 'src' / protocol['panel'])
    lookup = matrix(rows, panel, protocol)
    supplements = read(study / 'records.json')
    supplemental_protocol = read(ablation / 'src/config/gpu_joint_ablation_protocol.json')
    supplemental_panel = dict(panel, rows=[r for r in panel['rows'] if r['index'] in supplemental_protocol['indices']])
    matrix(supplements, supplemental_panel, supplemental_protocol)
    groups = {}
    for dataset in sorted({r['relative_path'].split('/')[0] for r in panel['rows']}):
        subset = dict(panel, rows=[r for r in panel['rows'] if r['relative_path'].split('/')[0] == dataset])
        groups[dataset] = {control: paired(lookup, subset, protocol, protocol['primary_variant'], control)
                           for control in ('legacy_cpu', 'random_batch', 'adaptive_no_inner_batch_auto')}
    save_json(output / 'source_groups.json', groups)
    stages = {}
    for variant in protocol['variants']:
        selected = [r for r in rows if r['variant'] == variant and r['status'] == 'PASS']
        metrics = {'preparation_s': [r['preparation_runtime_s'] for r in selected if r.get('preparation_runtime_s') is not None]}
        for key in ('gradient_proposals', 'surrogate_calibration', 'cec_and_mapping'):
            metrics[key + '_inclusive_s'] = [r['stages'][key]['inclusive_s'] for r in selected if key in r.get('stages', {})]
        stages[variant] = {key: {'observed_method_points': len(values), 'median_s': statistics.median(values) if values else None}
                           for key, values in metrics.items()}
    save_json(output / 'stage_summary.json', stages)
    figures = make_figures(rows, panel, protocol, summary, output / 'figures')
    body = ['# 最终证据补充：来源分组与贡献边界', '',
        '本次只消费已终态、已原始审计的数据，并检查独立正式分析签名的文件hash。没有新增训练、推理、rewrite、CEC或mapping。', '',
        '## 数据来源分组', '',
        '下列为主方法/对照的真实delay比；seed对只作重复运行，不作为独立电路。所有面积/缺项口径沿用原协议，不产生新通过阈值。', '',
        '| 来源 | 对照 | 可比seed对/预期 | 主方法胜/平/负 | delay比几何均值 | 对照/主方法中位wall比 |',
        '|---|---|---:|---|---:|---:|']
    for dataset, controls in groups.items():
        for control, result in controls.items():
            o = result['outcomes']
            body.append(f'| {dataset} | {control} | {result["measured_area_feasible_seed_pairs"]}/{result["expected_seed_pairs"]} | '
                f'{o.get("win",0)}/{o.get("tie",0)}/{o.get("loss",0)} | '
                f'{result["geomean_delay_ratio_seed_pairs"]:.6f} | {result["median_shared_allocation_speed"]:.6f} |')
    body += ['', '## 成本与作业证据', '',
        '`stage_summary.json`保存各方法准备/提案/校准/CEC-mapping的阶段中位数与实际分母；阶段inclusive时间有嵌套，不能相加。'
        '缺失阶段为未记录/不适用，不补成零。完整wall和真实工具调用使用原正式报告。', '',
        '`accounting.json`保存三个原始作业的真实终态、退出码、elapsed及batch MaxRSS；MaxRSS不是GPU显存，显存峰值未专门记录。', '',
        '## 补充控制和图表', '',
        '四设计补充控制的原始报告路径：`' + str(study / 'REPORT.md') + '`。只移除LiteCut标量delta、显式邻接pair或修复；'
        'no-score仍付Student成本，no-pairs仍有DAG耦合，no-repair会改变预算请求序列。详见原控制协议，不推及四设计之外。', '',
        '重新导出的四组PNG/PDF只修正图例遮挡与y轴加性offset显示，数值来自同一正式audited_records/statistics。'
        '原独立分析目录的图表和hash保持原样。', '',
        '等时间曲线是在固定最多八请求运行上取30/120/600秒已完成点；不是在每个期限持续增加搜索预算的新实验。'
        '原生/贪心按各自停止规则结束后，保存其已测best。没有为未完成点补造no-op或预测标签。', '']
    (output / 'REPORT.md').write_text('\n'.join(body))
    save_json(output / 'accounting.json', accounts)
    references = [source / 'analysis_completion.json', source / 'statistics.json', source / 'audited_records.json',
        source / 'RESULTS.md', study / 'records.json', study / 'paired_results.json', study / 'evidence_hashes.json',
        study / 'completion.json', ablation / 'output/default_compatibility.json', ablation / 'output/ablation_tests.json',
        formal / 'output/final_panel_completion.json']
    save_json(output / 'provenance.json', {'source_commit': read(run / 'run.json')['source_commit'],
        'references_sha256': {str(p): digest(p) for p in references}, 'figure_hashes': figures,
        'scope': 'Supplementary evidence assembly only. Raw experiment and original analysis files preserved; no automatic claim of algorithm superiority or goal completion.'})
    save_json(output / 'completion.json', {'status': 'ASSEMBLED',
        'formal_methods': len(rows), 'supplementary_methods': len(supplements),
        'artifacts_sha256': {str(p.relative_to(output)): digest(p) for p in output.rglob('*') if p.is_file()}})
    print(json.dumps({'status': 'ASSEMBLED', 'output': str(output)}))


if __name__ == '__main__':
    main()
