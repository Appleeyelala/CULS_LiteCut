#!/usr/bin/env python3
"""Re-audit the terminal frozen experiment and write final statistics/figures."""
import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def read(path):
    return json.loads(Path(path).read_text())


def number(value, digits=4):
    return '缺失/未定义' if value is None else f'{value:.{digits}f}'


def write_report(output, profile, experiment, panel, protocol, summary, rows, accounting):
    methods = summary['methods']
    primary = protocol['primary_variant']
    old = summary['comparisons']['legacy_cpu']
    direct = methods[primary]
    assessments = summary['protocol_assessment']
    names = {'MET_ON_THIS_PANEL': '在该面板达到', 'NOT_MET_ON_THIS_PANEL': '在该面板未达到', 'INCOMPLETE': '证据不完整'}
    job = experiment['slurm_job_id']
    lines = ['# GPU 梯度联合优化：冻结面板结果', '',
        '本文汇总已有固定实验的原始证据复核与统计，不运行新的 rewrite、推理、训练或 mapping。', '',
        f'实验路径：`{profile}`；Slurm **{job}**：`{accounting["state"]}` / `{accounting["exit_code"]}`。',
        f'实验快照：`{experiment["source_commit"]}`。本次分析使用独立 committed-source 快照；分析清单同时保留两者身份。', '',
        f'固定 {len(panel["rows"])} 个输入 × {len(protocol["seeds"])} 种子 × {len(protocol["variants"])} 方法，共 **{len(rows)}** 个终态方法点。'
        '这些输入属于此前已分析的905集合，不是未见测试集；三个seed是同一设计的重复运行。', '',
        '## 预先口径与实际结论', '',
        f'- 相对旧版的QoR非劣门槛：**{names[assessments["qor_noninferiority_vs_legacy"]]}**。'
        f'有效配对 {old["measured_area_feasible_seed_pairs"]}/{old["expected_seed_pairs"]}；'
        f'delay比几何均值 {number(old["geomean_delay_ratio_seed_pairs"],6)}，P90 {number(old["p90_delay_ratio_seed_pairs"],6)}。'
        '原阈值分别≤1.01、≤1.05；不完整数据不能按成功子集宣称通过。',
        f'- 端到端中位加速≥1.5门槛：**{names[assessments["median_speed_at_least_1p5_vs_legacy"]]}**；'
        f'同分配旧版/主方法中位时间比 {number(old["median_shared_allocation_speed"],4)}。'
        '同时间质量的替代条件没有预先固定总体二元阈值，因此只展示完整胜负与可用性，不事后挑一个截点宣布通过。',
        '- 最终点只从完整安装、CEC证明、统一Liberty测量且面积≤1.05 no-op的点选择。Native/original若超面积仍保留原始值，不能按同约束给其判胜。', '',
        '## 实现、基线与完整成本', '',
        '主方法按固定规模规则保留小图全代理Softmax/STE/Adam，或在当前已测incumbent上计算一次DAG adjoint，'
        '保留相邻MFFC双线性项，批量更新32条float64 Softmax轨迹40步，随后做全局合法邻域投影。'
        '冻结LiteCut的19维局部预测只作为常量特征，不反传进入Student，也不把局部分数当真实path/global delay。', '',
        '可选八维标量残差只使用此前成功真实观察；每个组合独立从原图replay，全部指定选择必须ACCEPTED，再各自CEC与ABC mapping/timing。'
        'batch只复用原图枚举及并行执行独立CPU验证，不把MFFC不重叠当成时序或安装独立。', '',
        '旧版从冻结旧Python快照和旧gpuls导入；新版使用已审计的新gpuls默认正增益政策。'
        '原图/native控制没有无用候选/模型准备；随机没有Student/代理拟合；greedy使用真实冻结Student。'
        '所有方法的必要导入、初始化、特征、校准、传输、失败/修复、CEC、mapping和保存都在单电路冷wall内，另计一次输入staging。'
        '源码归档/排队不在该wall内，系统页缓存未清，阶段inclusive时间存在嵌套，不能相加。', '',
        '原生ABC与CULS分别是一遍原流程，不伪称八次搜索；greedy在确定点与必要修复后停止。'
        '随机/梯度至多8逻辑请求，包含no-op、warmup、失败、修复及探索，不是8次成功改写。', '',
        '| 方法 | 成功/全部 | 面积可行final | 中位E2E s（成功进程） | 达标/全部 | 达标者中位时间 s | 失败请求 | replay / CEC / map调用 |',
        '|---|---:|---:|---:|---:|---:|---:|---|']
    for variant in protocol['variants']:
        m = methods[variant]
        lines.append(f'| {variant} | {m["successful_points"]}/{m["expected_method_points"]} | {m["area_feasible_final_points"]} | '
            f'{number(m["median_end_to_end_s_successful_methods"])} | {m["quality_target_reached"]}/{m["expected_method_points"]} | '
            f'{number(m["median_time_to_quality_among_hits_s"])} | {m["failed_requests_in_successful_methods"]} | '
            f'{m["native_replay_commands"]} / {m["cec_commands"]} / {m["mapping_commands"]} |')
    lines += ['', '达标阈值固定为面积可行no-op/native ABC较低delay的1.01倍；native缺失/超面积时只用no-op。'
        '“达标者中位时间”是条件统计，必须连同未达标分母阅读。调用总数覆盖成功方法进程；失败进程的缺失记账另列，不当成零成本成功。', '',
        '## 梯度与GPU独立分析', '',
        '| 对照（主方法/对照） | 完整可比seed对 | 胜/平/负 | delay比几何均值 | 完整seed的设计数 |',
        '|---|---:|---|---:|---:|']
    for control, comparison in summary['comparisons'].items():
        outcomes = comparison['outcomes']
        lines.append(f'| {control} | {comparison["measured_area_feasible_seed_pairs"]}/{comparison["expected_seed_pairs"]} | '
            f'{outcomes.get("win",0)}/{outcomes.get("tie",0)}/{outcomes.get("loss",0)} | '
            f'{number(comparison["geomean_delay_ratio_seed_pairs"],6)} | {comparison["complete_designs"]} |')
    cpu = summary['cpu_gradient_vs_random']
    gpu = summary['gpu']
    lines += ['', f'去掉GPU策略后的 `adaptive_batch_cpu` 对自然成本random：有效{cpu["measured_area_feasible_seed_pairs"]}'
        f'/{cpu["expected_seed_pairs"]}对，delay比几何均值{number(cpu["geomean_delay_ratio_seed_pairs"],6)}。'
        '它比较的是含特征/代理/更新/修复的完整梯度包，并未单独隔离每一个局部特征。', '',
        '`adaptive_no_inner_batch_auto`只去掉内层backward/Adam，保留代理校准、大图adjoint、反馈、初始化和最终投影。'
        '它不是完全无梯度方法；若预算先被warmup/repair占满，不能把该点称为实际执行了内层消融提案。', '',
        f'主方法直接梯度请求成功 {direct["successful_direct_requests"]}/{direct["direct_requests"]}；'
        f'其中面积可行且真实delay低于no-op的成功直接点 {direct["area_feasible_direct_points_better_than_noop"]}。'
        f'最终来源计数：`{json.dumps(direct["final_sources"],ensure_ascii=False,sort_keys=True)}`。'
        '修复、warmup、no-op收益不全部归入直接梯度贡献。', '',
        f'实际执行CUDA提案的配对 {gpu["actual_cuda_seed_pairs"]}，覆盖{gpu["actual_cuda_designs"]}设计；'
        f'其中同分配、代码/模型/域/工作量相同、完整请求序列和EDA调用数一致的配对 '
        f'{gpu["identical_sequence_shared_allocation_seed_pairs"]}，覆盖{gpu["identical_sequence_shared_allocation_designs"]}设计，'
        f'端到端CPU/auto中位时间比 {number(gpu["median_end_to_end_speed_identical_cuda_pairs"],4)}。'
        f'另有{gpu["cpu_or_no_proposal_seed_pairs"]}个CPU或未执行提案的配对，不计为GPU收益。'
        '所有变化请求、缺失和更慢的配对均保存在CSV/JSON中。', '',
        '开发集ACE/div另有一次完整同算法CPU/CUDA配对327.205694/169.601841s（约1.929倍），'
        '八个请求完全一致但最终均no-op；这个额外开发样例不并入正式16设计统计，也不意味着更早找到改善点。', '',
        '## 相同时间的真实质量', '',
        '| 对照 | 截点 s | 双方可用时胜/平/负 | 仅主方法可用 / 仅对照可用 / 双方无点 | 双方可用delay比几何均值 |',
        '|---|---:|---|---|---:|']
    for control, entries in summary['equal_time'].items():
        for entry in entries:
            counts = entry['outcomes']
            lines.append(f'| {control} | {entry["cutoff_s"]} | {counts.get("win",0)}/{counts.get("tie",0)}/{counts.get("loss",0)} | '
                f'{counts.get("proposed_only",0)}/{counts.get("control_only",0)}/{counts.get("neither_available",0)} | '
                f'{number(entry["ratio_geomean_when_both_available"],6)} |')
    lines += ['', '这些点来自真实monotonic完成时间，超出截点的结果不倒填；尚无已验证点的情况不补成no-op。'
        '曲线按设计展示全部seed已具有点后的几何均值，三次重复不当作三个独立电路。', '',
        '## 逐设计结果', '',
        '下表为三个seed的实际final delay中位数；括号中为可比面积可行seed数。缺失/超面积不填造结果。完整每seed delay/area、预算、source、时间与调用数在CSV。', '',
        '| 输入 | AND | 主方法 ps | 旧版 ps | random ps | native ABC ps | native CULS ps | 主方法E2E中位 s |',
        '|---|---:|---|---|---|---|---|---:|']
    for design in panel['rows']:
        fields = []
        for variant in (primary, 'legacy_cpu', 'random_batch', 'native_abc', 'native_culs'):
            selected = [r for r in rows if r['index'] == design['index'] and r['variant'] == variant
                        and r['status'] == 'PASS' and r.get('area_feasible_1p05')]
            fields.append((number(statistics.median([r['delay_ps'] for r in selected])) + f' ({len(selected)}/{len(protocol["seeds"])})') if selected else '缺失/不可行 (0)')
        runtime = [r['end_to_end_with_input_staging_s'] for r in rows if r['index'] == design['index'] and r['variant'] == primary and r['status'] == 'PASS']
        lines.append(f'| {design["index"]:04d} {design["relative_path"]} | {design["and_count"]} | ' + ' | '.join(fields) +
                     ' | ' + number(statistics.median(runtime) if runtime else None) + ' |')
    lines += ['', '## 证据、复现与边界', '',
        '- `audited_records.json`：全部方法点及每个成功观察的真实完成时间；`evidence_hashes.json`：原始AIG、trace、CEC、Liberty、网表、命令、工具与来源hash。',
        '- `statistics.json`：完整/缺失分母、预先门槛、seed对与设计层级统计；`methods.csv`、`pairs.csv`、`method_points.csv`提供可复用表格。',
        '- `figures/`：quality-runtime散点、逐设计validated曲线、GPU贡献与梯度控制图，均保存PNG/PDF和hash。',
        '- 正式源码、实际命令、输入/模型/库/工具及Slurm配置由原run的`src/`、`input/final_panel_preflight.json`、`job.slurm`、`run.json`与本分析的`provenance.json`共同定位。',
        '- `source_commit`是外层运行快照；`imported_source_commit`明确区分冻结旧算法实际导入来源。编译优化记录见仓库`docs/global_timing/PERFORMANCE_AUDIT.md`，不把旧ABC说成未优化构建。', '',
        '当前证据不建立精确local→global时序预测、AGD论文严格复现、唯一逻辑critical-path身份或物理signoff结果。'
        '独立control保留了原图序列化差异和候选安装失败；面积单位采用ASAP7库单位。'
        '正式结果是固定小面板的描述性比较，没有以相关设计/重复seed提供统计显著性或未见设计泛化结论。', '',
        '尚可研究的问题包括有效方向与真实mapper决策的一致性、可执行组合/修复成本及CPU校准/特征准备瓶颈；'
        '本报告没有把这些问题当成已经完成的新算法。无论门槛是否达到，均保留上述五项交付的可复现实现、基线、独立贡献、质量—时间结果与限制。', '']
    with (output / 'RESULTS.md').open('x') as stream:
        stream.write('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--final-profile', type=Path, required=True)
    args = parser.parse_args()
    run, profile = args.run.resolve(), args.final_profile.resolve()
    if not os.environ.get('SLURM_JOB_ID') or not all(p.is_relative_to('/data/wushangge') for p in (run, profile)):
        parser.error('Requires a new committed-source /data analysis run and Slurm allocation.')
    from global_timing.common import digest, save_json
    from global_timing.profile_results import add_runtime_metrics, audit_profiles
    from global_timing.joint_results import summarize
    from global_timing.joint_plots import make_figures
    output = run / 'output/final_synthesis'
    output.mkdir(exist_ok=False)
    experiment = read(profile / 'run.json')
    job = str(experiment['slurm_job_id'])
    account = subprocess.check_output(['sacct', '-j', job, '--format=JobIDRaw,State,ExitCode', '-n', '-P'], text=True)
    fields = next(line.split('|') for line in account.splitlines() if line.split('|')[0] == job)
    accounting = {'job_id': job, 'state': fields[1], 'exit_code': fields[2], 'raw': account}
    save_json(output / 'experiment_accounting.json', accounting)
    if fields[1] != 'COMPLETED' or fields[2] != '0:0':
        raise RuntimeError('Original frozen job is not a verified successful terminal run; preserve and inspect its logs.')
    completion = read(profile / 'output/final_panel_completion.json')
    protocol_path = profile / 'src/config/gpu_joint_final_protocol.json'
    protocol = read(protocol_path)
    panel_path = profile / 'src' / protocol['panel']
    panel = read(panel_path)
    preflight = read(profile / 'input/final_panel_preflight.json')
    if (completion['status'] != 'EXECUTED_AND_AUDITED' or digest(protocol_path) != preflight['protocol_sha256']
            or digest(panel_path) != protocol['panel_sha256']
            or digest(profile / 'output/profile_summary.json') != completion['profile_summary_sha256']):
        raise ValueError('Frozen protocol/panel/experiment completion evidence changed.')
    rows, hashes = audit_profiles([profile])
    rows = add_runtime_metrics(rows, protocol['cutoffs_s'])
    summary = summarize(rows, panel, protocol)
    save_json(output / 'audited_records.json', rows)
    save_json(output / 'evidence_hashes.json', hashes)
    save_json(output / 'statistics.json', summary)
    figures = make_figures(rows, panel, protocol, summary, output / 'figures')
    for filename, records in [('methods.csv', [{'variant': k, **v} for k, v in summary['methods'].items()]),
                              ('pairs.csv', [p for v in summary['comparisons'].values() for p in v['pairs']]),
                              ('method_points.csv', [{k: v for k, v in r.items() if k not in ('points', 'baseline', 'stages', 'algorithm_sha256')} for r in rows])]:
        names = sorted({key for record in records for key in record})
        with (output / filename).open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=names)
            writer.writeheader()
            writer.writerows(records)
    write_report(output, profile, experiment, panel, protocol, summary, rows, accounting)
    provenance = {'experiment_run': str(profile), 'experiment_source_commit': experiment['source_commit'],
        'analysis_run': str(run), 'analysis_source_commit': read(run / 'run.json')['source_commit'],
        'protocol_sha256': digest(protocol_path), 'panel_sha256': digest(panel_path),
        'completion_sha256': digest(profile / 'output/final_panel_completion.json'),
        'analysis_code_sha256': {name: digest(run / 'src' / name) for name in
            ['global_timing/profile_results.py', 'global_timing/joint_results.py', 'global_timing/joint_plots.py', 'scripts/summarize_final_joint.py']},
        'figure_artifacts_sha256': figures,
        'scope': 'New postprocessing only; original experiment files/snapshot were not modified and no EDA/model work was performed.'}
    save_json(output / 'provenance.json', provenance)
    save_json(output / 'analysis_completion.json', {'status': 'ANALYSIS_COMPLETE',
        'coverage': summary['coverage'], 'protocol_assessment': summary['protocol_assessment'],
        'artifacts_sha256': {str(p.relative_to(output)): digest(p) for p in output.rglob('*') if p.is_file()},
        'scope': 'Final report/figures completed; this status is not a claim of algorithm superiority or paper readiness.'})
    print(json.dumps({'status': 'ANALYSIS_COMPLETE', 'output': str(output), 'protocol_assessment': summary['protocol_assessment']}))


if __name__ == '__main__':
    main()
