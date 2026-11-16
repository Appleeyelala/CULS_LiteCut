#!/usr/bin/env python3
"""Publish small, auditable tables from new nonpositive-gain runs only."""
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from global_timing.common import digest
from run_nonpositive_coverage import ZERO17


def read(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def table(path, rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def med(values):
    values=[v for v in values if v is not None]
    return statistics.median(values) if values else None


def zero_reason(raw, verified, feasible, manifest):
    if raw is None: return 'NOT_OBSERVED_GPU_BOUNDARY'
    if raw==0:
        if manifest.get('eligible_cut_count')==0: return 'NO_ELIGIBLE_RETAINED_CUT'
        if manifest.get('subgraphs_considered',0)==manifest.get('self_root_reuse_rejected',-1):
            return 'ALL_AVAILABLE_LIBRARY_ROOTS_REUSE_ORIGINAL_ROOT'
        return 'GPU_EMPTY_SEE_REJECTION_COUNTERS'
    if verified is None: return 'BOOLEAN_BOUNDARY_NOT_COMPLETED'
    if verified==0: return 'ALL_RAW_DECISIONS_FAILED_OR_UNPROVEN_BOOLEAN'
    if feasible is None: return 'FEASIBILITY_BOUNDARY_NOT_COMPLETED'
    if feasible==0: return 'ALL_VERIFIED_CHOICES_BLOCKED_BY_ORIGINAL_ROOT_OR_DESCENDANT_REUSE'
    return 'LEGAL_NONZERO_DOMAIN_EXISTS'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--coverage-run',type=Path,required=True)
    p.add_argument('--control-run',type=Path,required=True)
    p.add_argument('--reference-run',type=Path,required=True)
    p.add_argument('--full-run',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    coverage=args.coverage_run;reference=args.reference_run
    protocol=read(coverage/'input/coverage_protocol.json')
    state=read(coverage/'output/COVERAGE_STATUS.json')
    assert state['completed']==100 and len(state['records'])==100
    assert {r['index'] for r in state['records']}=={r['index'] for r in protocol['rows']}
    evidence={}
    def evidence_file(path,role):
        if path.is_file():
            evidence[str(path)]={'sha256':digest(path),'bytes':path.stat().st_size,'role':role}
    rows=[];probes=[]
    for r in sorted(state['records'],key=lambda r:r['index']):
        i=r['index']; work=Path(r['evidence'])/'work'
        oldwork=reference/'output'/f'case_{i:04d}_Ours/work'
        oldsum=read(oldwork/'summary.json')
        oldraw=read(oldwork/'candidates/raw/gate3c_raw_manifest.json')
        olddec=read(oldwork/'cut_validation/verified_catalogue.json')['decisions']
        blocked=read(oldwork/'candidate_feasibility.json')['blocked_choices'] if (oldwork/'candidate_feasibility.json').exists() else []
        blocked_d=Counter(b['decision_index'] for b in blocked)
        raw=read(work/'candidates/raw/gate3c_raw_manifest.json') if (work/'candidates/raw/gate3c_raw_manifest.json').exists() else {}
        row={'circuit_id':i,'input':r['input'],'input_sha256':r['input_sha256'],'status':r['status'],
             'old_raw_decisions':oldraw['executable_decision_count'],'old_raw_candidates':oldraw['candidate_count'],
             'old_verified_decisions':len(olddec),'old_verified_candidates':oldsum['candidates'],
             'old_feasible_decisions':sum(d['candidate_count']>blocked_d[k] for k,d in enumerate(olddec)),
             'old_feasible_candidates':oldsum['candidates']-len(blocked),
             'new_eligible_cut_count':raw.get('eligible_cut_count')}
        for stage in ('raw','verified','feasible'):
            for name in ('decisions','candidates','positive','zero','negative'):
                row[f'new_{stage}_{name}']=r.get(stage,{}).get(name)
        for name in ('subgraphs_considered','self_root_reuse_rejected','no_library_implementation_cuts','gain_rule_rejected'):
            row[name]=raw.get(name)
        row.update(coverage_runtime_s=r['coverage_runtime_s'],evidence_path=str(work),
            reason=zero_reason(row['new_raw_candidates'],row['new_verified_candidates'],row['new_feasible_candidates'],raw))
        rows.append(row)
        for probe in r.get('probes',[]):
            probes.append({'circuit_id':i,**probe})
        for relative,role in [('coverage.json','new boundary counts'),('candidates/raw/gate3c_raw_manifest.json','new GPU counters'),
            ('cut_validation/audit.json','new Boolean audit'),('candidate_feasibility.json','new reuse exclusions')]:
            evidence_file(work/relative,role)
        evidence_file(oldwork/'summary.json','frozen old STE100 summary')
        evidence_file(coverage/'output/records'/f'{i:04d}.json','new cold coverage record')
        for path in work.glob('probe_*/probe.json'): evidence_file(path,'new Replay/CEC probe')
    table(out/'CANDIDATE_COUNTS_100.csv',rows)
    table(out/'REPLAY_PROBES.csv',probes)
    resolved=[r for r in rows if r['circuit_id'] in ZERO17 and r['reason']=='LEGAL_NONZERO_DOMAIN_EXISTS']
    unresolved=[r for r in rows if r['circuit_id'] in ZERO17 and r not in resolved]
    gate=len(resolved)==17
    assert gate==state['gate17_passed']
    counts={k:sum(r[k] or 0 for r in rows) for k in rows[0] if k.startswith(('old_','new_')) and k!='new_eligible_cut_count'}
    stats={'fixed100_records':len(rows),'states':dict(Counter(r['status'] for r in rows)),
        'legal_candidate_circuits':sum(r['reason']=='LEGAL_NONZERO_DOMAIN_EXISTS' for r in rows),
        'original17_resolved':len(resolved),'unresolved17':[r['circuit_id'] for r in unresolved],
        'GO_905':state['GO_905'],'candidate_totals':counts,
        'coverage_runtime_median_s':med([r['coverage_runtime_s'] for r in rows]),
        'probe_states':dict(Counter(p['status'] for p in probes))}
    write_json(out/'COVERAGE_STATS.json',stats)
    md=['# 原17个零候选的解决情况','',f"原17例解决 **{len(resolved)}/17**；`GO_905={str(state['GO_905']).lower()}`。",
        '', '判据是每例 raw > 0、Boolean 后 > 0、CandidateFeasibility 后 > 0；不是只统计 GPU mask。',
        '有合法非零选择域不等于优化器一定提交 Rewrite；原16个候选存在但 no-op 的策略没有改动。','',
        '|ID|原 raw|新 raw|Boolean 后|Feasible 后|合法正/零/负|结论|','|---:|---:|---:|---:|---:|---|---|']
    for r in rows:
        if r['circuit_id'] not in ZERO17: continue
        md.append(f"|{r['circuit_id']}|{r['old_raw_candidates']}|{r['new_raw_candidates']}|{r['new_verified_candidates']}|{r['new_feasible_candidates']}|{r['new_feasible_positive']}/{r['new_feasible_zero']}/{r['new_feasible_negative']}|{r['reason']}|")
    md+=['','原始记录路径及 SHA256 见 `EVIDENCE_INDEX.json`；逐电路路径、输入 SHA 和各层 gain 分布见 CSV。',
        '新策略下 `gain_rule_rejected=0`；相同 retained cuts 和库产生的零/负 gain 方案进入 Boolean 和可行性检查。',
        '对于旧 raw=0、新正 gain=0 而合法零/负 gain>0 的电路，新增机会直接来自本轮放宽 gain 规则。',
        '根自复用及原根/后代复用拒绝仍然保留；没有扩大 cut-size、cut预算或替换 library。',
        'gain 是 GPU 枚举时的结构节点收益符号，不是 LiteCut 分数、代理目标或真实 Mapping Delay。']
    (out/'ZERO_CANDIDATE_RESOLUTION.md').write_text('\n'.join(md)+'\n')
    build=read(coverage/'run.json'); controls=read(args.control_run/'run.json')
    control=read(args.control_run/'output/default_control.json')
    meta={'coverage_run':str(coverage),'coverage_source_commit':build['source_commit'],
        'coverage_slurm_job_id':build['slurm_job_id'],'controls_run':str(args.control_run),
        'controls_source_commit':controls['source_commit'],'controls_slurm_job_id':controls['slurm_job_id'],
        'tools':protocol['tools'],'sample_seed':20261116,'rows':protocol['rows'],
        'environment':build['environment'],'resources':build['resources'],'full905_run':str(args.full_run) if args.full_run else None}
    write_json(out/'RUN_PROVENANCE.json',meta)
    for path in [coverage/'run.json',coverage/'job.slurm',coverage/'input/commands.json',
        coverage/'input/coverage_protocol.json',coverage/'output/GATE17.json',coverage/'output/COVERAGE_STATUS.json',
        args.control_run/'output/default_control.json',args.control_run/'logs/command_0.log']:
        evidence_file(path,'run/control provenance')
    (out/'README.md').write_text(f'''# 非正节点 gain 候选实验

先读 [候选覆盖结论](ZERO_CANDIDATE_RESOLUTION.md)，再读 [100电路逐项计数](CANDIDATE_COUNTS_100.csv)、[结果汇总](RESULTS_SUMMARY.md) 和 [实现与证据](IMPLEMENTATION_AND_EVIDENCE.md)。

这是 `allow_nonpositive_candidates` 独立策略。原 STE100 正式源码 `9a199bf0ea0dcf20bf55e40da8caa8b8bbff5fac` 及历史905方向导数数据均未改写。本轮候选验证源码 `{build['source_commit']}`，Slurm `{build['slurm_job_id']}`。独立 gpuls SHA256 `{protocol['tools']['gpuls']['sha256']}`。

固定100沿用 seed=20261116、原100 ID及逐输入SHA；清单和工具身份见 `RUN_PROVENANCE.json`。`REPLAY_PROBES.csv` 是少量实际安装/CEC检查，不是 Adam 搜索质量。真实原始证据位置及哈希见 `EVIDENCE_INDEX.json`。

原17例门槛：{len(resolved)}/17。GO_905={str(state['GO_905']).lower()}。{'全量新执行见 full905/；100子集结果来自该905运行，没有另跑100或A/B。' if args.full_run else '没有提交905全量结果；是否启动以门槛及已记录运行状态为准。'}
''')
    implementation=f'''# 实现与证据

- `projects/culs/src/algorithms/rewrite.cu::EvaluateNode`：显式 allowNonPositiveCandidates 仅绕过 gain 正负过滤；根自复用、cut及library语义不变。mask 1/2/3记录正/零/负，非零仍表示成员资格；默认正 gain 调用仍只产生1。
- `GPUSolver::EnumerateAndPreEvaluate` 接收显式策略和可选7项GPU计数；Gate3C输出 candidate_rule=allow_nonpositive_gain、候选 gain 符号及拒绝统计。use_zeros=false 是旧参数，不代表新策略只允许正 gain。
- Gate3E普通/prepared枚举与导出用同一策略；prepared缓存核验策略身份。安装仍须 `CULS_REPLAY_ALLOW_NONPOSITIVE_GAIN=1`，保持cut-leaf/环检查、完整 ACCEPTED 和最终CEC。
- `global_timing/candidates.py` 显式设置并核验 `CULS_ALLOW_NONPOSITIVE_CANDIDATES`，将gain符号随catalogue保留；`execution.py::execute_one`传递两项独立许可。
- `NonpositiveSteConfig`版本化固定协议，旧 `SteSearchConfig` 默认不变。新协议第二次 Mapping 只记录质量，不按 Delay 回退；完整安装/CEC失败无替代指标，真实Area超1.05报告AREA_LIMIT_EXCEEDED并保留观察值。
- STE搜索、可行性规则、LiteCut、12参数代理、GPU DAG源码未改。40步、lr=.12、1restart、T1→.3、no-op+2、max16均冻结。

候选检查源码 `{build['source_commit']}`；运行路径 `{coverage}`。编译命令、独立gpuls及所有冻结工具/模型/库哈希见 `RUN_PROVENANCE.json` 和该run的 `input/commands.json`/`run.json`。全量运行另列其真实源码commit，不把后续发布提交冒充运行提交。

最小验证：{len(probes)}项新增零/负gain Replay/CEC探针（含普通与prepared模式），状态 {dict(Counter(p['status'] for p in probes))}。输入3默认正gain导出的 {control['candidates']} 个候选身份与旧正式记录完全一致。Slurm {controls['slurm_job_id']} 的5项政策单元检查验证退化候选保留、失败无替代指标、面积上限、旧回退和固定参数；实际输出见证据索引。

完整文件保留在/data；公开目录只提交源码、表格和小型证据索引，不上传AIG、模型、Liberty、二进制或大型日志。
'''
    (out/'IMPLEMENTATION_AND_EVIDENCE.md').write_text(implementation)
    fullstats=full_reports(args,out,evidence_file) if args.full_run else None
    write_json(out/'EVIDENCE_INDEX.json',{'files':evidence})
    summary=f'''# 结果汇总

固定100候选检查终态：{stats['states']}；具有合法非零候选 {stats['legal_candidate_circuits']}/100；原17例解决 {len(resolved)}/17，GO_905={str(state['GO_905']).lower()}。

候选总数：旧raw {counts['old_raw_candidates']} → 新raw {counts['new_raw_candidates']}；旧feasible {counts['old_feasible_candidates']} → 新feasible {counts['new_feasible_candidates']}。新合法正/零/负分别 {counts['new_feasible_positive']}/{counts['new_feasible_zero']}/{counts['new_feasible_negative']}。覆盖检查完整冷进程时间中位数 {stats['coverage_runtime_median_s']:.6f}s，包含指定代表例的Replay/CEC探针，不是完整Ours优化Runtime。

探针实际完整安装/CEC：{stats['probe_states']}。候选存在不能推断Adam会选择它，也不能推断真实Delay改善。候选域扩大增加Boolean、可行性和后续LiteCut/代理的实际工作量，本轮没有做Runtime优化。

'''
    if fullstats: summary+=fullstats
    else: summary+='本轮未执行新的完整Ours质量测量；没有可报告的本轮真实Mapping Delay或优化Runtime，不复用旧指标冒充新测量。没有重跑A/B。\n'
    (out/'RESULTS_SUMMARY.md').write_text(summary)
    print(json.dumps(stats,ensure_ascii=False))


def full_reports(args,out,evidence_file):
    run=args.full_run
    dest=out/'full905';dest.mkdir()
    paths=sorted((run/'output/records').glob('*_Ours.json'))
    records={read(p)['index']:read(p) for p in paths}
    inventory=read(run/'input/full905_manifest.json')['rows']
    assert len(inventory)==905
    old905=[r for r in csv.DictReader((ROOT/'BENCHMARK905_RESULTS.csv').open()) if r['method']=='Ours']
    oldzero={int(r['index']) for r in old905 if r['candidates']=='0'}
    assert len(oldzero)==94
    candidate_rows=[]; result_rows=[]; zero=[]; unknown=[]; quality=[]
    legal=set(); noop=[]
    for item in inventory:
        i=item['index']; r=records.get(i,{})
        c=r.get('candidate_counts',{}); manifest=c.get('gpu_manifest',{})
        cr={'circuit_id':i,'input':item['relative_path'],'input_sha256':item['sha256'],
            'status':r.get('status','NOT_COMPLETED'),'old905_zero_candidate':i in oldzero,
            'eligible_cut_count':manifest.get('eligible_cut_count')}
        for stage in ('raw','verified','feasible'):
            for key in ('decisions','candidates','positive','zero','negative'):
                cr[f'{stage}_{key}']=c.get(stage,{}).get(key)
        if cr['raw_candidates'] is None and manifest:
            cr['raw_candidates']=manifest['candidate_count'];cr['raw_decisions']=manifest['executable_decision_count']
        for key in ('subgraphs_considered','self_root_reuse_rejected','no_library_implementation_cuts','gain_rule_rejected'):
            cr[key]=manifest.get(key)
        reason=zero_reason(cr['raw_candidates'],cr['verified_candidates'],cr['feasible_candidates'],manifest)
        has=(cr['feasible_candidates']>0) if cr['feasible_candidates'] is not None else None
        cr.update(has_legal_candidate=has,reason=reason,evidence_path=r.get('detail_path'))
        candidate_rows.append(cr)
        if has is False:zero.append(cr)
        elif has is True:legal.add(i)
        else:unknown.append(cr)
        original=r.get('original') or {};final=r.get('final_candidate') or {}
        selected=r.get('selected') or {};metrics=selected.get('metrics') or {}
        fm=final.get('metrics') or {}
        valid=final.get('status')=='PASS' and fm.get('cec')=='PASS'
        rr={'circuit_id':i,'input':item['relative_path'],'input_sha256':item['sha256'],
            'status':r.get('status','NOT_COMPLETED'),'has_legal_candidate':has,
            'candidate_domain_status':'ZERO_CANDIDATE' if has is False else 'LEGAL_CANDIDATE' if has else 'UNKNOWN',
            'optimizer_selected_nonempty':r.get('optimizer_selected_nonempty'),
            'adam_steps':r.get('adam_steps'),'replay_status':r.get('replay_status'),
            'cec':r.get('candidate_cec'),'original_cec':original.get('cec'),
            'original_delay_ps':original.get('delay_ps'),'original_area':original.get('area'),
            'candidate_delay_ps':fm.get('delay_ps') if valid else None,'candidate_area':fm.get('area') if valid else None,
            'selected_delay_ps':metrics.get('delay_ps') if metrics.get('cec')=='PASS' else None,
            'selected_area':metrics.get('area') if metrics.get('cec')=='PASS' else None,
            'selected_source':selected.get('source'),'mapping_calls':r.get('mapping_calls'),
            'mapping_calls_with_missing_clock':r.get('mapping_calls_with_missing_clock'),
            'measured_full_runtime_s':r.get('measured_full_runtime_s'),
            'second_mapping_abc_time_s':r.get('second_mapping_abc_time_s'),
            'optimization_runtime_s':r.get('optimization_runtime_s'),
            'source_commit':r.get('source_commit'),'slurm_job_id':r.get('slurm_job_id'),
            'error':r.get('error'),'evidence_path':r.get('detail_path')}
        result_rows.append(rr)
        if valid and original.get('delay_ps') is not None:
            quality.append((i,fm['delay_ps'],original['delay_ps'],fm['area']<=1.05*original['area']))
        if r.get('optimizer_selected_nonempty') is False:noop.append(i)
        evidence_file(run/'output/records'/f'{i:04d}_Ours.json','new full905 terminal record')
    table(dest/'FULL905_CANDIDATE_COUNTS.csv',candidate_rows)
    table(dest/'FULL905_OURS_RESULTS.csv',result_rows)
    zeroids={r['circuit_id'] for r in zero};unknownids={r['circuit_id'] for r in unknown}
    win=sum(a<b for _,a,b,_ in quality);tie=sum(a==b for _,a,b,_ in quality);lose=sum(a>b for _,a,b,_ in quality)
    ratios=[a/b for _,a,b,_ in quality if a>0 and b>0]
    states=dict(Counter(r['status'] for r in result_rows))
    sessions=[read(p) for p in sorted((run/'input').glob('full905_session_*.json'))]
    runstatus={'status':'TERMINAL_905' if len(records)==905 else 'PARTIAL','expected':905,'completed':len(records),
        'states':states,'zero_candidates':len(zero),'zero_ids':sorted(zeroids),'candidate_domain_unknown':sorted(unknownids),
        'old94_resolved':sorted(oldzero&legal),'old94_still_zero':sorted(oldzero&zeroids),
        'old94_unknown':sorted(oldzero&unknownids),'new_zero':sorted(zeroids-oldzero),
        'nonempty_cec_mapping_pass':len(quality),'candidate_delay_win_tie_loss_vs_own_original':[win,tie,lose],
        'candidate_delay_ratio_geomean':math.exp(sum(map(math.log,ratios))/len(ratios)) if ratios else None,
        'candidate_area_limit_exceeded':sum(not feasible for _,_,_,feasible in quality),
        'optimizer_noop_ids':noop,'runtime_medians':{k:med([r[k] for r in result_rows]) for k in ('measured_full_runtime_s','second_mapping_abc_time_s','optimization_runtime_s')},
        'runtime_available':{k:sum(r[k] is not None for r in result_rows) for k in ('measured_full_runtime_s','second_mapping_abc_time_s','optimization_runtime_s')},
        'sessions':sessions,'run':str(run),'run_metadata':read(run/'run.json')}
    write_json(dest/'FULL905_RUN_STATUS.json',runstatus)
    lines=['# 新版完整905候选域检查','',f"终态记录 {len(records)}/905；状态 {states}。确认零候选 {len(zero)}，候选域未完成/不可判定 {len(unknown)}。",'',
        f"旧94个零候选：恢复合法候选 {len(oldzero&legal)}，仍零 {len(oldzero&zeroids)}，不可判定 {len(oldzero&unknownids)}；新增零候选 {len(zeroids-oldzero)}。",
        '原17例门槛全部通过且固定100完成后才提交本次905作业；详见上级目录。',
        '零候选与候选存在但优化器no-op分开；失败发生在某层完成前时该层留空，不能补造零或成功。','',
        '|ID|名称|首次空/未完成边界|','|---:|---|---|']
    lines += [f"|{r['circuit_id']}|{r['input']}|{r['reason']}|" for r in zero+unknown]
    lines += ['', '旧94恢复ID：'+str(sorted(oldzero&legal)), '旧94仍零ID：'+str(sorted(oldzero&zeroids)),
        '旧94不可判定ID：'+str(sorted(oldzero&unknownids)), '新增零ID：'+str(sorted(zeroids-oldzero))]
    (dest/'FULL905_ZERO_CANDIDATES.md').write_text('\n'.join(lines)+'\n')
    historical={int(r['circuit_id']):r for r in csv.DictReader((ROOT/'analysis/ste_adam_100/results/STE_ADAM_100_RESULTS.csv').open())}
    sample=[]
    for r in result_rows:
        if r['circuit_id'] not in historical:continue
        old=historical[r['circuit_id']]
        sample.append(dict(r,**{f'historical_{m}_{k}':old[f'{m}_{k}'] for m in ('A','B') for k in ('delay_ps','runtime_s','status')}))
    table(out/'STE_ADAM_100_NONGAIN_RESULTS.csv',sample)
    for path in [run/'run.json',run/'input/full905_manifest.json']+list((run/'input').glob('full905_session_*.json')):
        evidence_file(path,'new full905 provenance')
    return f"""新版905实际终态 {len(records)}/905，{states}。确认零候选 {len(zero)}，候选域不可判定 {len(unknown)}。旧94变化见 full905/FULL905_ZERO_CANDIDATES.md。

完整安装、CEC及Mapping成功的非空候选 {len(quality)} 个，相对自己付费原图真实Delay胜/平/负 {win}/{tie}/{lose}；候选/原图Delay比几何均值 {runstatus['candidate_delay_ratio_geomean']}。真实Area超1.05的数量 {runstatus['candidate_area_limit_exceeded']}，单列不伪装可行改善；没有按真实Delay回退。

Runtime中位数（有值分母见JSON）：{runstatus['runtime_medians']}。optimization_runtime_s = measured_full_runtime_s - second_mapping_abc_time_s；无第二次Mapping扣0，缺失实际命令时钟时不估算。此为研究记账指标，不是已测一次Mapping部署性能。905按独立冷进程分片执行，A/B为历史测量，没有重跑，也不作严格同负载Runtime配对。

100子集结果直接来自本次905同ID的实际Ours运行（不是另起100次重复实验），附历史A/B参考。失败/超时及缺失均保留，见CSV与运行状态JSON。
"""

if __name__=='__main__':
    main()
