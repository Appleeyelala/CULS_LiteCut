"""Tables and exportable figures from audited measured A/B/Ours records only."""
from collections import defaultdict
import json


def number(value, digits=4):
    return 'N/A' if value is None else f'{value:.{digits}f}'


def write_report(output, prefix, records, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures = output / 'figures'
    figures.mkdir()
    comparisons = summary['comparisons']
    for label, key, title, logarithmic in [
        ('delay_relative_A_B', 'delay_ratio', 'Mapped delay ratio: Ours / baseline', False),
        ('runtime_relative_A_B', 'runtime_ratio_baseline_over_ours', 'Cold runtime ratio: baseline / Ours', True),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4.4))
        for baseline in ['A', 'B']:
            values = sorted(p[key] for p in comparisons[baseline]['points'] if p[key] is not None and (not logarithmic or p[key] > 0))
            if values:
                ax.plot(values, [(i + 1) / len(values) for i in range(len(values))], label=f'{baseline}: n={len(values)}')
        ax.axvline(1, color='gray', linestyle='--', linewidth=1)
        if logarithmic:
            ax.set_xscale('log')
        else:
            ax.ticklabel_format(axis='x', style='plain', useOffset=False)
        ax.set(xlabel=title, ylabel='Fraction of defined successful pairs', ylim=(0, 1.02))
        ax.legend()
        ax.grid(alpha=.25)
        fig.tight_layout()
        for ext in ['png', 'pdf']:
            fig.savefig(figures / f'{label}.{ext}', dpi=170)
        plt.close(fig)
    grouped = defaultdict(dict)
    for r in records:
        grouped[r['index']][r['method']] = r
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, baseline in zip(axes, ['A', 'B']):
        points = [p for p in comparisons[baseline]['points'] if p['delay_ratio'] is not None and p['runtime_ratio_baseline_over_ours'] is not None
                  and all(r['status'] == 'PASS' and r.get('area_feasible') is True for r in grouped[p['index']].values())]
        for low, high in [(0, 4096), (4096, 16384), (16384, 65536), (65536, None)]:
            data = [p for p in points if grouped[p['index']]['Ours']['and_count'] >= low
                    and (high is None or grouped[p['index']]['Ours']['and_count'] < high)]
            if data:
                ax.scatter([1 / p['runtime_ratio_baseline_over_ours'] for p in data], [p['delay_ratio'] for p in data],
                           s=15, alpha=.6, label=f'{low}..{high if high else "max"} AND ({len(data)})')
        ax.axhline(1, color='gray', linestyle='--', linewidth=1)
        ax.axvline(1, color='gray', linestyle='--', linewidth=1)
        ax.set(xscale='log', xlabel=f'Runtime Ours / {baseline}', ylabel=f'Delay Ours / {baseline}',
               title='All three valid and area feasible')
        ax.ticklabel_format(axis='y', style='plain', useOffset=False)
        ax.grid(alpha=.25)
        ax.legend(fontsize=7)
    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(figures / f'quality_runtime_by_size.{ext}', dpi=170)
    plt.close(fig)
    lines = ['# MFFC GPU 三方法评估：' + ('完整905结果' if summary['formal'] else '开发验证，非正式905结果'), '',
             f"证据范围：{summary['designs']}输入、{summary['method_points']}终态方法点；原始测量审计{summary['audit']['actual_evaluations']}份，成功replay观察{summary['audit']['successful_replay_observations']}份。", '',
             '| 方法 | 尝试/成功/失败/超时 | 中位冷E2E s | P90 s | 全部终态总记录时间 s | 面积可行成功点 |',
             '|---|---|---:|---:|---:|---:|']
    for method, data in summary['methods'].items():
        states = data['states']
        runtime = data['runtime_all_terminal']
        lines.append(f"| {method} | {data['attempted']}/{states.get('PASS',0)}/{states.get('FAIL',0)}/{states.get('TIMEOUT',0)} | {number(runtime['median'])} | {number(runtime['p90'])} | {number(runtime['sum'])} | {data['area_feasible']} |")
    lines += ['', '| 比较 | 成功配对 | Delay胜/平/负 | 改善中位 % | Delay ratio几何均值（分母） | 配对baseline/Ours时间比中位 |', '|---|---:|---|---:|---|---:|']
    for baseline, data in comparisons.items():
        w = data['delay_win_tie_loss']
        gm = data['delay_ratio_geometric']
        lines.append(f"| 相对{baseline} | {data['successful_pairs']} | {w.get('better',0)}/{w.get('tie',0)}/{w.get('worse',0)} | {number(data['improvement_percent']['median'])} | {number(gm['value'],6)} ({gm['n_positive_defined']}) | {number(data['paired_runtime_ratio']['median'])} |")
    lines += ['', 'Delay ratio=Ours/baseline，越低越好；时间比=baseline/Ours，越高越好。全部单方法runtime含失败/超时；配对比只含成功方法。面积单位为固定Liberty的库单位。CEC通过仅指成功测量；失败不给替代delay。', '',
              '缺少完整冷时钟的方法点数：' + str({m: d['runtime_unavailable'] for m, d in summary['methods'].items()}) + '；这类中断保持N/A，不补零，总记录时间仅求和可用时钟。', '',
              'GPU并行的是Candidate→Decision→MFFC聚合与保留依赖的完整Region DAG前/反传播，最终真实改图仍顺序受控。仅A/B对照不能独立证明GPU或梯度的因果收益；阶段与设备是实现证据。候选预检仍非完整可执行性预测。', '',
              '原始目录：`' + summary['experiment'] + '`；算法源码`' + summary['source_commit'] + '`。所有预测、Student分数与真实ABC_SCL_LIBERTY分列。无候选设计数：' + str(summary['known_no_candidate_designs']) + '，未知候选数设计：' + str(summary['unknown_candidate_count_designs']) + '。', '',
              '## 完整来源与规模分组', '',
              '| 分组 | 输入数 | Ours成功/尝试 | 对A Delay比 | 对B Delay比 | A/Ours时间比 | B/Ours时间比 |', '|---|---:|---|---:|---:|---:|---:|']
    for family in ['by_source', 'by_and_size']:
        for name, group in summary[family].items():
            ours = group['methods']['Ours']
            a, b = group['comparisons']['A'], group['comparisons']['B']
            lines.append(f"| {family}/{name} | {group['designs']} | {ours['states'].get('PASS',0)}/{ours['attempted']} | {number(a['delay_ratio_geometric']['value'],6)} | {number(b['delay_ratio_geometric']['value'],6)} | {number(a['paired_runtime_ratio']['median'])} | {number(b['paired_runtime_ratio']['median'])} |")
    lines += ['', '## 面积条件下的质量—时间关系', '',
              '仅统计A/B/Ours共同成功且均满足本方法付费no-op面积阈值的输入。better/tie/worse分别按冻结delay/runtime容差判断；下表九格关系互斥，未按结果挑子集。', '']
    for baseline in ['A', 'B']:
        data = comparisons[baseline]
        lines += [f"对{baseline}，共同有效可行分母{data['common_three_valid_feasible_pairs']}：", '',
                  '```json', json.dumps(data['common_feasible_relation_grid'], indent=2, ensure_ascii=False), '```', '']
    lines += ['## 实际设备、调用与主要成本', '', '| 方法 | CEC调用 | Mapping调用 | 原生replay命令 | 失败候选请求 | repair请求 | 梯度设备/最终来源 |', '|---|---:|---:|---:|---:|---:|---|']
    for name, data in summary['methods'].items():
        lines.append(f"| {name} | {data['cec_commands']} | {data['mapping_commands']} | {data['replay_commands']} | {data['failed_requests']} | {data['repair_requests']} | {data['gradient_devices']} / {data['final_sources']} |")
    lines += ['', 'Ours各inclusive阶段：', '', '| 阶段 | 有记录方法点 | 中位 s | P90 s | 总inclusive s |', '|---|---:|---:|---:|---:|']
    for name, data in summary['methods']['Ours']['inclusive_stages_s'].items():
        lines.append(f"| {name} | {data['n']} | {number(data['median'])} | {number(data['p90'])} | {number(data['sum'])} |")
    lines += ['', '阶段和工具调用可嵌套/并发，不能相加当完整wall或直接换算独占百分比。整批本次driver elapsed为' + number(summary['suite_completion']['session_elapsed_s']) + 's；与单电路cold latency分开。断点多session的总elapsed需结合各Slurm记录。', '',
              '已中断且无完整观察的内部请求：' + str({m: d['incomplete_internal_requests'] for m, d in summary['methods'].items()}) + '；损坏/不完整的失败命令文件：' + str({m: d['unreadable_partial_command_files'] for m, d in summary['methods'].items()}) + '。这些保持显式缺失，不补造成成功或零成本。', '',
              '## 图与完整数据', '']
    for name in ['delay_relative_A_B', 'runtime_relative_A_B', 'quality_runtime_by_size']:
        lines += [f'![{name}](figures/{name}.png)', '', f'[导出PDF](figures/{name}.pdf)', '']
    lines += [f'逐电路长表：`{prefix}_RESULTS.csv`；完整统计与分母：`{prefix}_SUMMARY.json`；逐点核验索引：`audited_records.json`；原始证据hash：`evidence_hashes.json`。', '',
              '## 解释边界', '',
              '本轮是固定小预算、一个预定seed和既有905集合的真实结果，不是未见集或最优性证明。可微梯度只属于固定原图代理，不经过Student或真实mapper；保留final best也可能来自repair/no-op/付费original。A/B与Ours的成本不同，不能把整体相对速度当作纯GPU加速。没有布局布线后signoff，也没有在本轮加入random/greedy/旧版或joint扩展。', '']
    (output / (prefix + '_RESULTS.md')).write_text('\n'.join(lines), encoding='utf-8')
