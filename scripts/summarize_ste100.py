#!/usr/bin/env python3
"""Summarize actual cold-method records, never substitute proxy predictions."""
from collections import Counter
import csv
import json
import math
from pathlib import Path
import statistics
import sys


def summarize(run):
    run = Path(run)
    output = run / 'output'
    sample = json.loads((output / 'STE100_SAMPLE_MANIFEST.json').read_text())['rows']
    preflight = json.loads((run / 'input/ste100_preflight.json').read_text())
    records = [json.loads(p.read_text()) for p in sorted((output / 'records').glob('*.json'))]
    by_key = {(r['index'], r['method']): r for r in records}
    if len(by_key) != len(records):
        raise ValueError('Duplicate terminal method records.')
    expected = {(r['index'], m) for r in sample for m in ('A', 'B', 'Ours')}
    if not set(by_key) <= expected:
        raise ValueError('Non-sample method record.')
    fields = ['circuit_id', 'input', 'A_delay_ps', 'A_runtime_s', 'B_delay_ps', 'B_runtime_s',
              'Ours_delay_ps', 'Ours_runtime_s']
    with (output / 'STE100_RESULTS.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in sample:
            item = {'circuit_id': row['index'], 'input': row['relative_path']}
            for method in ('A', 'B', 'Ours'):
                record = by_key.get((row['index'], method), {})
                item[method + '_delay_ps'] = record['metrics']['delay_ps'] if record.get('status') == 'PASS' else None
                item[method + '_runtime_s'] = record.get('end_to_end_runtime_s')
            writer.writerow(item)
    med = lambda xs: statistics.median(xs) if xs else None
    fmt = lambda x: 'N/A' if x is None else f'{x:.6f}'
    lines = ['# STE + Adam 固定随机 100 电路结果', '',
             f"正式终态：{len(records)}/300；电路清单：100/905；源码 `{preflight['source_commit']}`。",
             '采样种子 20261116；优化种子 20261117；40 Adam steps、1 restart、最多16编辑。',
             'Delay 为同一 ABC/ASAP7 Liberty 的真实 global timing，单位 ps；Runtime 为独立冷进程完整端到端秒数。', '',
             '| 方法 | 成功/100 | 真实 Delay 中位数 ps | Runtime 中位数 s（所有已完成尝试） |',
             '|---|---:|---:|---:|']
    method_stats = {}
    for method in ('A', 'B', 'Ours'):
        rows = [r for r in records if r['method'] == method]
        delays = [r['metrics']['delay_ps'] for r in rows if r['status'] == 'PASS']
        times = [r['end_to_end_runtime_s'] for r in rows if r.get('end_to_end_runtime_s') is not None]
        method_stats[method] = dict(success=len(delays), attempts=len(rows), delay_median_ps=med(delays),
                                    runtime_median_s=med(times), runtime_sum_s=sum(times))
        lines.append(f'| {method} | {len(delays)} | {fmt(med(delays))} | {fmt(med(times))} |')
    lines += ['', '不同电路的原始 Delay 中位数不能作为配对性能收益；性能比较以下表为准。', '',
              '| 配对 | 成功配对数 | Delay 比几何均值（正值分母数） | Ours 胜/平/负 | Runtime 比中位数 |',
              '|---|---:|---:|---:|---:|']
    paired = {}
    for method in ('A', 'B'):
        pairs = [(by_key.get((r['index'], 'Ours')), by_key.get((r['index'], method))) for r in sample]
        pairs = [(a,b) for a,b in pairs if a and b and a['status'] == b['status'] == 'PASS']
        ratios, runtimes, counts = [], [], Counter()
        for a,b in pairs:
            x,y = a['metrics']['delay_ps'], b['metrics']['delay_ps']
            if min(x,y) > 0:
                ratios.append(x/y)
            tol = max(2e-5, 2e-6 * max(abs(x), abs(y)))
            counts['tie' if abs(x-y) <= tol else 'win' if x < y else 'loss'] += 1
            if b.get('end_to_end_runtime_s', 0) and a.get('end_to_end_runtime_s') is not None:
                runtimes.append(a['end_to_end_runtime_s']/b['end_to_end_runtime_s'])
        gmean = math.exp(statistics.mean(map(math.log, ratios))) if ratios else None
        paired[method] = dict(pairs=len(pairs), delay_geomean_ratio=gmean, positive_delay_pairs=len(ratios),
                              counts=dict(counts), runtime_ratio_median=med(runtimes), runtime_pairs=len(runtimes))
        lines.append(f"| Ours/{method} | {len(pairs)} | {fmt(gmean)} ({len(ratios)}) | {counts['win']}/{counts['tie']}/{counts['loss']} | {fmt(med(runtimes))} |")
    points, ours = [], []
    for r in records:
        work = Path(r['detail_path']) / 'work'
        nmap = len(list(work.rglob('mapping.command.json')))
        point = {k:r.get(k) for k in ('index','method','status','error','end_to_end_runtime_s','final_source','detail_path')}
        point['mapping_calls'] = nmap
        if r['method'] == 'Ours':
            if nmap > 2:
                raise ValueError('Ours exceeded Mapping budget.')
            path = work / 'summary.json'
            s = json.loads(path.read_text()) if path.exists() else {}
            final = s.get('final_candidate') or {}
            point.update(decisions=s.get('decisions'), backend=s.get('backend'), adam_steps=s.get('adam_steps'),
                         logits_total_change_norm=s.get('logits_total_change_norm'),
                         distinct_hard_combinations=s.get('distinct_hard_combinations_including_noop'),
                         final_candidate_status=final.get('status', 'NOT_REQUESTED'),
                         final_candidate_error=final.get('error'),
                         final_candidate_actual_delay_ps=final.get('metrics', {}).get('delay_ps'),
                         selected_edits=s.get('selected', {}).get('applied_rewrites'))
            ours.append(point)
        points.append(point)
    maps = Counter(r['mapping_calls'] for r in ours)
    sources = Counter(r['final_source'] for r in ours)
    validations = Counter(r['final_candidate_status'] for r in ours)
    lines += ['', f'方法成功/失败均保留；Ours Mapping 调用数分布：{dict(maps)}；最终来源：{dict(sources)}。',
              f'最终候选验证状态：{dict(validations)}。失败/面积不合格/无真实改善时返回 Ours 自己已付费测量的原图。',
              'Ours 原图 Mapping 成本计入自身 Runtime；内循环无 ABC/CEC/Mapping、无在线校准或修复重试。',
              '有效候选电路执行40步，零候选电路不加载代理；每步真实梯度与 Adam 更新范数见各 work/optimization/trajectory.json。',
              '未用代理预测替代真实结果；本对照不能单独证明 GPU 相对同算法 CPU 的加速。',
              '胜/平/负容差：max(2e-5 ps, 2e-6 × 两者较大 Delay)；所有配对排除失败，几何平均另排除零 Delay。',
              '', '路径：STE100_RESULTS.csv、STE100_SAMPLE_MANIFEST.json、STE100_STATUS.json；逐点原始结果见 case_XXXX_{A,B,Ours}/。']
    status = dict(all_300_terminal=set(by_key) == expected, expected=300, terminal=len(records),
                  missing=[{'index':i, 'method':m} for i,m in sorted(expected-set(by_key))],
                  source_commit=preflight['source_commit'], tools=preflight['tools'], protocol=preflight['protocol'],
                  method_stats=method_stats, paired=paired, points=points,
                  scope='Synthetic unit fixture and three smoke circuits excluded; raw failed candidates retained.')
    with (output / 'STE100_STATUS.json').open('x') as stream:
        json.dump(status, stream, indent=2, allow_nan=False)
        stream.write('\n')
    with (output / 'STE100_SUMMARY.md').open('x') as stream:
        stream.write('\n'.join(lines)+'\n')


if __name__ == '__main__':
    summarize(Path(sys.argv[1]))
