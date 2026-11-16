#!/usr/bin/env python3
"""Read saved formal-905 clocks; reconcile nested stages without running tools."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics


FROZEN = 'ecc85e6ac82571caf9f1a6004098dbe976536256'
STAGES = ['original_cec_mapping', 'mffc_partition', 'candidate_enumeration',
          'boolean_validation', 'candidate_reuse_preflight', 'litecut_preparation_inference',
          'global_model_preparation', 'global_gradient_and_joint_blocks',
          'gpu_or_cpu_proxy_calibration', 'noop_replay_validation', 'search_replay_validation',
          'unattributed_wall']
LABELS = ['原图 CEC+Mapping', 'MFFC 分区', '候选枚举及目录准备', 'Boolean/上下文证明',
          '候选复用预检', 'LiteCut 准备、特征、推理及缓存写入', '全局代理构建及设备初始化',
          '联合梯度与候选块评分', '代理反馈校准', '付费 no-op Replay+CEC+Mapping',
          '搜索提案 Replay+CEC+Mapping（外层）', '未单独计时部分']


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def stats(values):
    values = sorted(float(v) for v in values if v is not None)
    if not values:
        return {'n': 0, 'sum': 0.0, 'mean': None, 'median': None, 'p90': None}
    position = (len(values) - 1) * .9
    low, high = math.floor(position), math.ceil(position)
    return {'n': len(values), 'sum': math.fsum(values), 'mean': statistics.mean(values),
            'median': statistics.median(values),
            'p90': values[low] * (high - position) + values[high] * (position - low)
                   if low != high else values[low], 'min': values[0], 'max': values[-1]}


def ranks(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        for k in range(i, j):
            result[order[k]] = (i + j - 1) / 2
        i = j
    return result


def spearman(xs, ys):
    x, y = ranks(xs), ranks(ys)
    mx, my = statistics.mean(x), statistics.mean(y)
    denominator = math.sqrt(sum((a-mx)**2 for a in x) * sum((b-my)**2 for b in y))
    return sum((a-mx)*(b-my) for a, b in zip(x, y)) / denominator if denominator else None


def save_json(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def write_csv(path, rows):
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--bundle', type=Path, required=True)
    args = ap.parse_args()
    run, bundle = args.run.resolve(), args.bundle.resolve()
    assert os.environ.get('SLURM_JOB_ID')
    assert run.is_relative_to('/data/wushangge/CULS_LiteCut')
    output = run / 'output'
    coverage = json.loads((bundle / 'audit/coverage.json').read_text())
    assert coverage['source_commit'] == FROZEN and coverage['counts']['designs'] == 905
    versions = json.loads((bundle / 'config/versions.json').read_text())
    formal = Path(versions['experiment'])
    code_paths = ['global_timing/mffc_experiment.py', 'global_timing/execution.py',
                  'global_timing/litecut.py', 'scripts/run_mffc905.py',
                  'projects/culs/tools/gate3e_run.py']
    for name in code_paths:
        assert sha(run / 'src' / name) == sha(formal / 'src' / name), name
    inputs = [bundle / name for name in ['runtime/method_runtime.csv', 'runtime/stages.csv',
              'runtime/native_gpu_profiles.jsonl', 'audit/coverage.json', 'benchmark/BENCHMARK905_RESULTS.csv']]
    with inputs[0].open() as stream:
        methods = list(csv.DictReader(stream))
    assert len(methods) == 2715
    by_id = defaultdict(dict)
    for row in methods:
        row['circuit_id'] = int(row['circuit_id'])
        row['cold_wall_s'] = float(row['cold_wall_s'])
        row['tools'] = json.loads(row['tool_seconds_inclusive_json']) or {}
        row['stage_sums'] = json.loads(row['stage_seconds_json']) or {}
        assert row['method'] not in by_id[row['circuit_id']]
        by_id[row['circuit_id']][row['method']] = row
    assert set(by_id) == set(range(905)) and all(set(v) == {'A','B','Ours'} for v in by_id.values())
    benchmark = {}
    with inputs[-1].open() as stream:
        for row in csv.DictReader(stream):
            if row['method'] == 'Ours':
                benchmark[int(row['index'])] = row
    native = {}
    with inputs[2].open() as stream:
        for line in stream:
            value = json.loads(line)
            native[value['circuit_id']] = value
    per_circuit, cache_states = [], Counter()
    for cid, group in sorted(by_id.items()):
        ours = group['Ours']
        path = bundle / f'optimization/cases/{cid:04d}/work/summary.json'
        inputs.append(path)
        summary = json.loads(path.read_text())
        assert ours['status'] == summary['status'] == 'PASS'
        stages = summary['stages']
        component = {name: 0.0 for name in STAGES}
        real = [i for i, s in enumerate(stages) if s['stage'] == 'real_replay_cec_mapping']
        assert real and stages[real[0]-1]['stage'] == 'boolean_validation'
        component['noop_replay_validation'] = stages[real[0]]['elapsed_s']
        nested = 0.0
        for i in real[1:]:
            assert stages[i+1]['stage'] == 'real_batch_replay_cec_mapping', (cid, i)
            nested += stages[i]['elapsed_s']
        raw = Counter()
        for s in stages:
            name, elapsed = s['stage'], s['elapsed_s']
            raw[name] += elapsed
            if name == 'real_batch_replay_cec_mapping':
                component['search_replay_validation'] += elapsed
            elif name != 'real_replay_cec_mapping':
                assert name in component, name
                component[name] += elapsed
        for name, elapsed in ours['stage_sums'].items():
            assert math.isclose(raw[name], elapsed, abs_tol=1e-8)
        accounted = math.fsum(component.values())
        assert math.isclose(accounted, math.fsum(raw.values()) - nested, abs_tol=1e-8)
        component['unattributed_wall'] = ours['cold_wall_s'] - accounted
        assert component['unattributed_wall'] >= 0, cid
        assert math.isclose(sum(component.values()), ours['cold_wall_s'], abs_tol=1e-8)
        for method in ['A','B']:
            m = group[method]
            m['serial_tool_s'] = sum(m['tools'].get(k,0) for k in ['cec','mapping','native_rewrite'])
            m['residual_s'] = m['cold_wall_s'] - m['serial_tool_s']
            assert m['residual_s'] >= 0, (cid, method)
        kernel = sum(sum(p['kernel_ms_since_collect'].values()) for p in native[cid]['native_profiles'])
        if ours['native_kernel_ms']:
            assert math.isclose(kernel,float(ours['native_kernel_ms']),abs_tol=1e-6)
        if int(ours['decisions']) > 0:
            cache_path = formal / f'output/case_{cid:04d}_Ours/work/litecut.json'
            inputs.append(cache_path)
            cache_states[str(json.loads(cache_path.read_text())['cache']['cache_hit'])] += 1
        record = {'circuit_id': cid, 'input': benchmark[cid]['input'],
                  'and_count': int(benchmark[cid]['and_count']), 'device': ours['device'],
                  'decisions': int(ours['decisions']), 'candidates': int(ours['candidates']),
                  'regions': int(ours['regions']), 'layers': int(ours['region_dag_layers']),
                  'A_status': group['A']['status'], 'B_status': group['B']['status'],
                  'A_wall_s': group['A']['cold_wall_s'], 'B_wall_s': group['B']['cold_wall_s'],
                  'Ours_wall_s': ours['cold_wall_s'], 'Ours_over_A': ours['cold_wall_s']/group['A']['cold_wall_s'],
                  'Ours_over_B': ours['cold_wall_s']/group['B']['cold_wall_s'] if group['B']['status']=='PASS' else None,
                  'nested_single_count':len(real)-1,'nested_seconds_removed':nested,
                  **component, 'native_kernel_s': kernel/1000 if ours['device']=='cuda' else None,
                  'native_kernel_wall_fraction':kernel/1000/ours['cold_wall_s'] if ours['device']=='cuda' else None,
                  'enumeration_tool_s':ours['tools'].get('enumeration',0),
                  'enumeration_other_s':component['candidate_enumeration']-ours['tools'].get('enumeration',0),
                  'mapping_commands':int(ours['mapping_commands']), 'cec_commands':int(ours['cec_commands']),
                  'replay_commands':int(ours['replay_commands']),
                  'final_source':benchmark[cid]['final_source']}
        assert record['enumeration_other_s'] >= 0
        per_circuit.append(record)
    total = sum(r['Ours_wall_s'] for r in per_circuit)
    phase_stats = {name: {**stats(r[name] for r in per_circuit),
                   'recorded_nonzero':sum(r[name]>0 for r in per_circuit),
                   'recorded_nonzero_stats':stats(r[name] for r in per_circuit if r[name]>0),
                   'share_of_total_wall':sum(r[name] for r in per_circuit)/total} for name in STAGES}
    methods_summary, tools = {}, {}
    for method in ['A','B','Ours']:
        records = [g[method] for g in by_id.values()]
        methods_summary[method] = {'states':dict(Counter(r['status'] for r in records)),
                                   'wall_s':stats(r['cold_wall_s'] for r in records)}
        tools[method] = {}
        for tool, count_name in [('mapping','mapping_commands'),('cec','cec_commands'),
                                 ('rewrite','replay_commands'),('native_rewrite','abc_native_rewrite_commands'),
                                 ('enumeration',None)]:
            values = [r['tools'].get(tool) for r in records]
            calls = sum(int(r[count_name] or 0) for r in records) if count_name else sum(tool in r['tools'] for r in records)
            tools[method][tool] = {'calls':calls,'per_point_inclusive_s':stats(values),
                                  'aggregate_seconds_per_call':sum(v for v in values if v is not None)/calls if calls else None}
    pairs = {}
    for baseline in ['A','B']:
        selected = [r for r in per_circuit if r[baseline+'_status']=='PASS']
        base_rows = [by_id[r['circuit_id']][baseline] for r in selected]
        gap = sum(r['Ours_wall_s']-r[baseline+'_wall_s'] for r in selected)
        decomposition = {
            'real_evaluation_windows_difference':sum(r['original_cec_mapping']+r['noop_replay_validation']+
                r['search_replay_validation']-b['tools'].get('mapping',0)-b['tools'].get('cec',0)
                for r,b in zip(selected,base_rows)),
            'mffc_candidate_boolean_preflight':sum(sum(r[k] for k in STAGES[1:5]) for r in selected),
            'litecut':sum(r['litecut_preparation_inference'] for r in selected),
            'global_model':sum(r['global_model_preparation'] for r in selected),
            'gradient_and_calibration':sum(r['global_gradient_and_joint_blocks']+r['gpu_or_cpu_proxy_calibration'] for r in selected),
            'residual_difference':sum(r['unattributed_wall']-b['residual_s'] for r,b in zip(selected,base_rows)),
            'baseline_native_rewrite_subtraction':-sum(b['tools'].get('native_rewrite',0) for b in base_rows)}
        assert math.isclose(sum(decomposition.values()),gap,abs_tol=1e-7)
        ratios=[r['Ours_wall_s']/r[baseline+'_wall_s'] for r in selected]
        pairs[baseline]={'n':len(selected),'Ours_over_baseline':stats(ratios),
            'wall_difference_s':stats(r['Ours_wall_s']-r[baseline+'_wall_s'] for r in selected),
            'gap_sum_s':gap,'gap_decomposition_sum_s':decomposition,
            'gap_decomposition_mean_s':{k:v/len(selected) for k,v in decomposition.items()},
            'gap_decomposition_fraction':{k:v/gap for k,v in decomposition.items()},
            'faster_beyond_protocol_tolerance':sum(r['Ours_wall_s'] < r[baseline+'_wall_s']-max(.01,.01*r[baseline+'_wall_s']) for r in selected)}
    groups={}
    for name, selected in [('all',per_circuit)]+[(d,[r for r in per_circuit if r['device']==d]) for d in ['cuda','cpu','none']]:
        groups[name]={'n':len(selected),'wall_s':stats(r['Ours_wall_s'] for r in selected),
            'Ours_over_A':stats(r['Ours_over_A'] for r in selected),
            'Ours_over_B_success':stats(r['Ours_over_B'] for r in selected),
            'stage_mean_s':{k:stats(r[k] for r in selected)['mean'] for k in STAGES}}
    cuda=[r for r in per_circuit if r['device']=='cuda']
    scored=[r for r in per_circuit if r['decisions']>0]
    summary={'scope':'Saved ecc85e6 formal905 only. Postprocessing, no EDA/ML/optimizer/performance tests.',
        'created_at':datetime.now(timezone.utc).isoformat(),'algorithm_source_commit':FROZEN,
        'analysis_source_commit':json.loads((run/'run.json').read_text())['source_commit'],
        'bundle':str(bundle),'output':str(output),'slurm_job_id':os.environ['SLURM_JOB_ID'],
        'coverage':coverage,'methods':methods_summary,'pairs':pairs,'ours_disjoint_top_level_stages':phase_stats,
        'tools_inclusive_not_additive':tools,'groups':groups,'cache_hit_counts':dict(cache_states),
        'nested_single_replay_windows_removed':{'count':sum(r['nested_single_count'] for r in per_circuit),
            'seconds':sum(r['nested_seconds_removed'] for r in per_circuit)},
        'native_cuda_only':{'n':len(cuda),'kernel_s':stats(r['native_kernel_s'] for r in cuda),
            'kernel_wall_fraction':stats(r['native_kernel_wall_fraction'] for r in cuda),
            'kernel_fraction_of_cuda_total_wall':sum(r['native_kernel_s'] for r in cuda)/sum(r['Ours_wall_s'] for r in cuda),
            'scope':'Native DAG kernels only, excludes Torch aggregation, CPU selection, initialization and EDA.'},
        'candidate_enumeration':{'tool_s':stats(r['enumeration_tool_s'] for r in per_circuit),
            'other_s':stats(r['enumeration_other_s'] for r in per_circuit)},
        'descriptive_spearman':{'candidates_vs_litecut_stage':spearman([r['candidates'] for r in scored],[r['litecut_preparation_inference'] for r in scored]),
            'layers_vs_native_kernel':spearman([r['layers'] for r in cuda],[r['native_kernel_s'] for r in cuda])},
        'top10_slowest_ours':sorted(per_circuit,key=lambda r:r['Ours_wall_s'],reverse=True)[:10],
        'reconciliation':{'all_905_have_nonnegative_unattributed_wall':True,
            'minimum_unattributed_wall_s':min(r['unattributed_wall'] for r in per_circuit),
            'all_per_circuit_components_sum_to_cold_wall':True,
            'all_paired_gap_components_sum_to_actual_gap':True,
            'instrumented_algorithm_files_equal_frozen_source':True}}
    write_csv(output/'runtime_per_circuit.csv',per_circuit)
    write_csv(output/'runtime_stage_summary.csv',[{'stage':k,**{f:v for f,v in v.items() if f!='recorded_nonzero_stats'}} for k,v in phase_stats.items()])
    save_json(output/'runtime_summary.json',summary)
    with (output/'INPUT_CHECKSUMS.sha256').open('x') as stream:
        for p in sorted(set(inputs)):
            stream.write(sha(p)+'  '+str(p)+'\n')
    make_report(output,summary,per_circuit)
    print(json.dumps({'status':'VERIFIED_SAVED_DATA_ANALYSIS','output':str(output),
        'methods':methods_summary,'gap_mean_s':{b:pairs[b]['gap_decomposition_mean_s'] for b in pairs},
        'stage_shares':{k:v['share_of_total_wall'] for k,v in phase_stats.items()},
        'native_cuda':summary['native_cuda_only'],'reconciliation':summary['reconciliation']},ensure_ascii=False))


def make_report(output,s,rows):
    lines=['# 最新 905 实验 Runtime 分析','',
        '正式算法冻结 `'+FROZEN+'`。仅分析已存 CSV/JSON，没有执行新 CULS、ABC、CEC、Mapping、LiteCut 或优化器。',
        'A=原图 CEC+Mapping；B=一次 CPU ABC rewrite 后 CEC+Mapping；Ours=候选全量准备、局部打分、GPU 全局梯度及多次真实验证。',
        '','## 端到端结果','','| 方法 | PASS/FAIL | 中位 s | 平均 s | P90 s | 合计 s |','|---|---:|---:|---:|---:|---:|']
    for method,d in s['methods'].items():
        w=d['wall_s']; lines.append(f"| {method} | {d['states'].get('PASS',0)}/{d['states'].get('FAIL',0)} | {w['median']:.4f} | {w['mean']:.4f} | {w['p90']:.4f} | {w['sum']:.4f} |")
    for b,p in s['pairs'].items():
        lines.append(f"\n相对 {b}：{p['n']} 个成功配对，逐电路 Ours/{b} 耗时比中位 {p['Ours_over_baseline']['median']:.4f}；耗时差中位 {p['wall_difference_s']['median']:.4f}s。超出协议容差的更快电路 {p['faster_beyond_protocol_tolerance']} 个。")
    lines += ['','## 去重后的 Ours 顶层计时区间','','| 阶段 | 有执行电路 | 执行者中位 s | 全905平均 s | 总 wall 占比 |','|---|---:|---:|---:|---:|']
    for k,label in zip(STAGES,LABELS):
        d=s['ours_disjoint_top_level_stages'][k]; lines.append(f"| {label} | {d['recorded_nonzero']} | {d['recorded_nonzero_stats']['median']:.4f} | {d['mean']:.4f} | {100*d['share_of_total_wall']:.2f}% |")
    nested=s['nested_single_replay_windows_removed']
    lines += ['',f"原始阶段表中 {nested['count']} 个单请求 replay 窗口被外层 batch 包含，合计重复 {nested['seconds']:.4f}s。只保留初始化的 no-op `real_replay_cec_mapping` 和所有 `real_batch_replay_cec_mapping`；其余嵌套单请求不再累加。源码保证这些保留的顶层区间顺序执行，内部两路验证的并发只计外层 wall。", 
        '其余顶层阶段按源码顺序保留；剩余量=cold E2E−这些互不重叠阶段。所有905电路剩余量非负、逐项相加严格回到E2E。该剩余量包含未单独计时的导入/启动、哈希、图/目录保存、日志扫描、序列化、同步间隙等；没有独立测量，不能全部归因给 Torch 或传输。',
        '阶段占比按总秒数/总E2E计算；中位数不可直接相加。原始工具/request inclusive 时间仍不可相加作wall占比。',
        '','## 对 A/B 耗时差的可加分解','','| 差值来源 | 相对A平均增量 s（905对） | 相对B平均增量 s（901对） |','|---|---:|---:|']
    gap_labels={'real_evaluation_windows_difference':'真实评估相关区间增量（含 Ours replay）',
        'mffc_candidate_boolean_preflight':'分区、枚举、Boolean证明与预检', 'litecut':'LiteCut管线',
        'global_model':'全局模型构建', 'gradient_and_calibration':'梯度+校准',
        'residual_difference':'未单独计时部分的差值','baseline_native_rewrite_subtraction':'扣除基线原生ABC rewrite'}
    for key,label in gap_labels.items():
        a=s['pairs']['A']['gap_decomposition_mean_s'][key]; b=s['pairs']['B']['gap_decomposition_mean_s'][key]
        lines.append(f'| {label} | {a:.4f} | {b:.4f} |')
    lines += ['','A/B 的验证按顺序工具命令窗口统计，Ours 按包含结果保存等工作的顶层评估窗口统计；粒度差进入剩余量。此分解用于完整核算，不声称独立测出了 Ours mapping/CEC 各自的独占wall。B的4个失败保留在全方法表中，成功配对分解排除它们。',
        '','## 工具调用与 inclusive 成本','','| 方法/工具 | 调用数 | 每电路有调用者中位 s | 调用总 inclusive s | 总秒数/调用数 s |','|---|---:|---:|---:|---:|']
    for method,tools in s['tools_inclusive_not_additive'].items():
        for name,d in tools.items():
            if d['calls']:
                st=d['per_point_inclusive_s']; lines.append(f"| {method}/{name} | {d['calls']} | {st['median']:.4f} | {st['sum']:.4f} | {d['aggregate_seconds_per_call']:.4f} |")
    lines += ['','Mapping 计时覆盖 `read_lib; read_aiger; map; topo; stime -p; write_verilog` 整个ABC命令，包含进程、库读入及结果写出；不是单独mapper内核时间。Ours两路验证可能并发，工具总inclusive秒数不能解释为独占wall或直接相加。',
        '','## GPU为什么没有使整体更快','',
        '1. A没有候选搜索，B只有一次原生rewrite；三种方法做的总工作量不同。Ours额外付费测原图和serialized no-op，再评估提案；不能用这两种基线测出同一算法的GPU加速率。',
        '2. `global_timing/litecut.py:score_catalogue`逐decision/choice构造子图、验证truth、提取特征、生成JSON；`projects/culs/tools/gate3e_run.py:LiteCutPredictor.__init__`明确以CPU加载Student。128批次推理不意味着使用GPU。没有独立CPU特征/Student/缓存写入子时钟，无法进一步精确拆分本阶段。',
        f"3. 整体对全部 no-op和候选共 {s['coverage']['counts']['litecut_score_rows']:,} 行打分，物理候选 {s['coverage']['counts']['replacement_candidates']:,}，decision {s['coverage']['counts']['decisions']:,}。随后每电路总请求预算仅4；前端全候选准备工作量远大于最终提交组合数。",
        '4. `scripts/run_mffc905.py:child/main`每个电路/方法启动独立子进程；Ours在判定有无候选之前就导入Torch，CUDA全局模型逐电路构建。初始化与导入成本不能跨电路摊销。811个有候选设计的局部分数缓存命中状态：'+str(s['cache_hit_counts'])+'。',
        '5. 实际设备分布：CUDA806、CPU5、无候选94。GPU负责候选聚合与全Region DAG前反传播；最终改图顺序受控，CEC与ABC mapping不由这个GPU代理加速。']
    gpu=s['native_cuda_only']; lines += ['',f"806个CUDA设计原生DAG内核合计中位 {gpu['kernel_s']['median']*1000:.4f}ms；逐电路kernel/E2E比例中位 {100*gpu['kernel_wall_fraction']['median']:.4f}%，按该子集总秒数加权为 {100*gpu['kernel_fraction_of_cuda_total_wall']:.4f}%。这只是原生内核事件，不能代表全部GPU/Torch耗时。即使把这部分记录时间完全消掉，账面收益仍有限；这不是CPU/GPU因果对照。", 
        '','## 设备分组（描述性，不能比较成CPU/GPU加速率）','','| 设备 | 电路数 | E2E中位s | E2E平均s | Ours/A中位 |','|---|---:|---:|---:|---:|']
    for name in ['cuda','cpu','none']:
        d=s['groups'][name]; lines.append(f"| {name} | {d['n']} | {d['wall_s']['median']:.4f} | {d['wall_s']['mean']:.4f} | {d['Ours_over_A']['median']:.4f} |")
    lines += ['','## 耗时最长的10个电路','','| ID/输入 | Ours s | LiteCut s | 原图+no-op+搜索真实区间 s | GPU原生内核 s |','|---|---:|---:|---:|---:|']
    for r in s['top10_slowest_ours']:
        real=r['original_cec_mapping']+r['noop_replay_validation']+r['search_replay_validation']; kernel=r['native_kernel_s']
        lines.append(f"| {r['circuit_id']}: {r['input']} | {r['Ours_wall_s']:.4f} | {r['litecut_preparation_inference']:.4f} | {real:.4f} | {kernel if kernel is not None else 'N/A'} |")
    lines += ['','## 结论与证据边界','',
        '现有数据支持：整体慢主要由额外真实验证和CPU候选/LiteCut准备造成，GPU原生梯度传播本身已是小成本。优先检查真实验证次数/复用，以及全候选的CPU特征与数据准备；单纯继续优化原生GPU内核不能解释或弥补现有端到端差距。这里只定位既有成本，没有修改算法。',
        '未验证：同一905算法、同一请求组合和预算下CPU/GPU的因果加速；Student GPU推理收益；独立H2D/D2H/特征/推理/JSON写入耗时；省略验证或跨电路共享初始化后的速度和质量。现有记录不支持给这些项编造精确收益。',
        '原生GPU核函数及Torch聚合的阶段界限不同；阶段计时同步只给出已有执行的wall，不是设备利用率。固定方法顺序、未清系统页缓存、单seed、小预算保持原协议。',
        '','## 文件与复现','',f"原数据：`{s['bundle']}`；输出：`{output}`；分析Slurm作业：`{s['slurm_job_id']}`。", 
        '`runtime_summary.json`保存全部统计和分母；`runtime_per_circuit.csv`保存905个电路的去重分解与A/B对照；`runtime_stage_summary.csv`为阶段汇总；`INPUT_CHECKSUMS.sha256`记录输入哈希。',
        '`scripts/analyze_mffc905_runtime.py --run <prepared-run> --bundle <existing-bundle>`只读已存结果。所有解释依赖的5个计时源码文件已逐字节核对与冻结正式src一致。',
        '','![Mean cold wall and measured stages](runtime_breakdown.png)','']
    (output/'MFFC905_RUNTIME_ANALYSIS.md').open('x').write('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    series={m:[] for m in ['A','B','Ours']}
    labels=['Real evaluation windows','ABC native rewrite','MFFC/candidates/proofs/preflight',
            'LiteCut preparation/inference','Global model initialization','Gradient + calibration',
            'Unattributed wall']
    for m in ['A','B']:
        tools=s['tools_inclusive_not_additive'][m]
        real=sum(tools[k]['per_point_inclusive_s']['sum'] for k in ['cec','mapping'])/905
        rewrite=tools['native_rewrite']['per_point_inclusive_s']['sum']/905
        series[m]=[real,rewrite,0,0,0,0,s['methods'][m]['wall_s']['mean']-real-rewrite]
    stages=s['ours_disjoint_top_level_stages']
    series['Ours']=[sum(stages[k]['mean'] for k in ['original_cec_mapping','noop_replay_validation','search_replay_validation']),0,
        sum(stages[k]['mean'] for k in STAGES[1:5]),stages['litecut_preparation_inference']['mean'],
        stages['global_model_preparation']['mean'],sum(stages[k]['mean'] for k in STAGES[7:9]),stages['unattributed_wall']['mean']]
    fig,ax=plt.subplots(figsize=(10,4.8)); bottom=[0.,0.,0.]; names=list(series)
    for i,label in enumerate(labels):
        heights=[series[m][i] for m in names]; ax.bar(names,heights,bottom=bottom,label=label)
        bottom=[b+h for b,h in zip(bottom,heights)]
    for x,y in zip(names,bottom): ax.text(x,y+.3,f'{y:.2f}s',ha='center')
    ax.set(ylabel='Mean cold wall per attempted circuit (s)',title='Frozen ecc85e6: 905 attempts per method; B includes 4 failures')
    ax.set_ylim(0,max(bottom)*1.14); ax.legend(loc='upper left',bbox_to_anchor=(1,1),fontsize=8)
    ax.grid(axis='y',alpha=.2); fig.tight_layout()
    for ext in ['png','pdf']: fig.savefig(output/f'runtime_breakdown.{ext}',dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
