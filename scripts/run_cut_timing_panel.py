#!/usr/bin/env python3
"""Frozen 96-pair evaluation, with a prediction barrier before all native Replay.

Prediction inputs contain NO candidate measurements. The sole timing label read
by predict() is the new normalized original anchor. No search or fitting runs.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from global_timing.common import digest, save_json
from global_timing.structure import parse_aig
from global_timing.mapper_input import normalize, aiger_bytes, canonical_graph


def read(path):return json.loads(Path(path).read_text())


def stage(args):
    manifest=read(ROOT/'analysis/joint_gradient_v3/learning_manifest.json')
    rows=[r for r in manifest['selected'] if r['role']=='validation']
    assert len(rows)==12
    args.output.mkdir(parents=True,exist_ok=False)
    index=[]
    for r in rows:
        cid=r['circuit_id'];old=args.data/f'case_{cid:04d}';frozen=read(old/'frozen.json');source=Path(frozen['source'])
        out=args.output/f'case_{cid:04d}';out.mkdir()
        for name in ['frozen.json','scores.json','catalogue.json']:
            shutil.copyfile(old/name,out/name)
        for name,src in [('input.aig',source/'input.aig'),('owners.csv',source/'work/partition/owners.csv'),('context.json',source/'context.json')]:
            shutil.copyfile(src,out/name)
        assert digest(out/'input.aig')==r['input_sha256']
        combos=[c for c in frozen['combinations'] if c['index']>0]
        assert len(combos)==8 and all(any(c['choices']) for c in combos)
        index.append({'circuit_id':cid,'input':r['input'],'folder':str(out),
                      'files':{p.name:digest(p) for p in out.iterdir()},'combinations':combos})
    save_json(args.output/'panel.json',{'base_commit':'9da83466686fb981c7fcbc6347720cdff9e76baa',
              'source_commit':os.environ['CUT_SOURCE_COMMIT'],'data_source':str(args.data),'circuits':index,
              'protocol':{'max_cut_inputs':4,'cuts_per_node':8,'matches_per_truth':4,'fixed_slew_ps':20.,
              'litecut_blend_exponent':.5,'scale_bounds':[.5,2.],'preflight_ids':[1,44,75],
              'mapping_command':'read_lib; read_aiger; map; topo; stime -p; write_verilog',
              'anchor':'per-PO rise/fall: normalized original + (model(combo)-model(noop)), clamp at zero',
              'new_neural_networks':False,'optimizer_changes':False}})


def predict(args):
    import torch
    from global_timing.evaluation import Evaluator
    from global_timing.cut_library import CutLibrary
    from global_timing.cut_timing import CutTimingProxy
    from global_timing.boolean_candidates import parse_library
    from global_timing.candidates import Catalogue
    from global_timing.feasibility import CandidateFeasibility
    from diagnose_joint_gradient_structure_v3 import literal_composition
    manifest=read(args.output/'panel.json');rewrite=parse_library(ROOT/'projects/culs/src/algorithms/rewrite_library.inc')
    lib=None;allrows=[]
    for r in manifest['circuits']:
        folder=Path(r['folder']);context=read(folder/'context.json');tools=context['tools'];aig=parse_aig(folder/'input.aig')
        for key in ['abc','library','checkpoint']:
            if digest(Path(tools[key]['path']))!=tools[key]['sha256']:raise ValueError('Changed tool '+key)
        normalized=folder/'normalized_original.aig';normalize(aig,normalized)
        evaluator=Evaluator(Path(tools['abc']['path']),Path(tools['library']['path']),folder/'input.aig',900)
        baseline=evaluator.evaluate(normalized,folder/'anchor')
        # Small causal control: same bytes twice, plus saved zero-edit Replay.
        if r['circuit_id'] in manifest['protocol']['preflight_ids']:
            repeat=evaluator.evaluate(normalized,folder/'repeat_identical')
            old=args.data/f'case_{r["circuit_id"]:04d}'/'measurements/combo_00/rewrite/raw/aigs/choice.aig'
            noop=folder/'normalized_saved_noop.aig';normalize(parse_aig(old),noop)
            if digest(noop)!=digest(normalized):raise ValueError('Normalized original/no-op bytes differ')
            control=evaluator.evaluate(noop,folder/'repeat_noop')
            assert repeat['po_rise_fall']==baseline['po_rise_fall']==control['po_rise_fall']
            save_json(folder/'input_control.json',{'status':'PASS','original_sha256':digest(normalized),'noop_sha256':digest(noop),
                        'same_input_delay_ps':[baseline['delay_ps'],repeat['delay_ps'],control['delay_ps']]})
        if lib is None:
            lib=CutLibrary(Path(tools['library']['path']));save_json(args.output/'library_model.json',lib.metadata())
        raw=read(folder/'catalogue.json');cat=Catalogue(raw['decisions'],raw['conflicts'],{int(k):v for k,v in raw['region_decisions'].items()})
        rule=CandidateFeasibility(aig,cat,rewrite)
        combos=r['combinations']
        for c in combos:rule.validate(c['choices'])
        started=time.monotonic()
        proxy=CutTimingProxy(aig,cat,read(folder/'scores.json')['scores'],lib,rewrite).cuda()
        save_json(folder/'port_model.json',{'arcs':proxy.audit,'device':torch.cuda.get_device_name(),'construction_s':time.monotonic()-started})
        with torch.no_grad():
            zero=proxy(proxy.one_hot([0]*len(cat.decisions)))[0]
            anchor=torch.tensor([[v['rise_ps'],v['fall_ps']] for v in baseline['po_rise_fall']],device='cuda',dtype=torch.double)
            predictions=[]
            for c in combos:
                raw_po=proxy(proxy.one_hot(c['choices']))[0]
                po=(anchor+(raw_po-zero)).clamp_min(0.)
                # Explicit symbolic composition is solely an installation audit,
                # not the numeric timing predictor; no actual Replay has run yet.
                composed,_=literal_composition(aig,cat.decisions,c['choices'],rewrite,simplify=True)
                # Native replay removes unreachable gates. Filter the diagnostic
                # expansion before the same nonoptimizing serialization.
                needed=set();stack=[l>>1 for l in composed.po_literals]
                while stack:
                    n=stack.pop()
                    if n not in composed.fanin_literals or n in needed:continue
                    needed.add(n);stack.extend(l>>1 for l in composed.fanin_literals[n])
                composed.fanin_literals={n:p for n,p in composed.fanin_literals.items() if n in needed}
                import hashlib
                row={**c,'circuit':f'ID{r["circuit_id"]}/{r["input"]}/combination_{c["index"]:02d}',
                     'predicted_mapping_delay_ps':float(po.max()),'predicted_po_rise_fall_ps':po.cpu().tolist(),
                     'model_noop_po_ps':zero.cpu().tolist(),'model_combo_po_ps':raw_po.cpu().tolist(),
                     'symbolic_normalized_sha256':hashlib.sha256(aiger_bytes(canonical_graph(composed))).hexdigest(),
                     'saved_unix_ns':time.time_ns()}
                predictions.append(row)
        save_json(folder/'predictions.json',{'rows':predictions,'anchor_sha256':digest(folder/'anchor/result.json'),
                  'normalized_original_sha256':digest(normalized),'frozen_sha256':digest(folder/'frozen.json')})
        allrows.append({'circuit_id':r['circuit_id'],'predictions':str(folder/'predictions.json'),'sha256':digest(folder/'predictions.json')})
        print(json.dumps({'phase':'predicted','circuit':r['circuit_id'],'rows':len(predictions)}),flush=True)
        del proxy;torch.cuda.empty_cache()
    save_json(args.output/'PREDICTIONS_FROZEN.json',{'created_unix_ns':time.time_ns(),'rows':allrows,'total':96,
              'source_commit':os.environ['CUT_SOURCE_COMMIT'],'panel_sha256':digest(args.output/'panel.json')})


def measure(args):
    from global_timing.candidates import Catalogue
    from global_timing.execution import replay_many
    from global_timing.evaluation import Evaluator
    barrier=read(args.output/'PREDICTIONS_FROZEN.json');manifest=read(args.output/'panel.json')
    assert len(barrier['rows'])==12 and barrier['total']==96
    for r in barrier['rows']:
        if digest(Path(r['predictions']))!=r['sha256']:raise ValueError('Prediction freeze changed')
    for r in manifest['circuits']:
        folder=Path(r['folder']);pred=read(folder/'predictions.json')['rows'];context=read(folder/'context.json')
        raw=read(folder/'catalogue.json');cat=Catalogue(raw['decisions'],raw['conflicts'],{int(k):v for k,v in raw['region_decisions'].items()})
        targets=[folder/'measurements'/f'combo_{c["index"]:02d}'/'rewrite' for c in pred]
        for p in targets:p.parent.mkdir(parents=True,exist_ok=False)
        replay_start=time.time_ns();assert replay_start>barrier['created_unix_ns']
        executions=replay_many(args.gpuls,folder/'input.aig',folder/'owners.csv',cat,[c['choices'] for c in pred],
                folder/'batch_replay',targets,timeout=900,prepared=True,allow_nonpositive_gain=True)
        evaluator=Evaluator(Path(context['tools']['abc']['path']),Path(context['tools']['library']['path']),folder/'input.aig',900)
        for c,execution,target in zip(pred,executions,targets):
            row={'circuit':c['circuit'],'index':c['index'],'choices':c['choices'],
                 'predicted_mapping_delay_ps':c['predicted_mapping_delay_ps'],'prediction_file_sha256':digest(folder/'predictions.json'),
                 'replay_start_unix_ns':replay_start,'execution':execution,'status':execution['status']}
            try:
                if execution['status']!='PASS':raise RuntimeError(execution.get('error'))
                candidate=target.parent/'normalized_candidate.aig';normalize(parse_aig(Path(execution['aig'])),candidate)
                row['input_sha256']=digest(candidate)
                row['symbolic_structure_matches']=digest(candidate)==c['symbolic_normalized_sha256']
                if not row['symbolic_structure_matches']:raise ValueError('Symbolic joint graph differs from actual Replay')
                metrics=evaluator.evaluate(candidate,target.parent/'evaluation')
                row.update(status='PASS',metrics=metrics,actual_abc_mapping_delay_ps=metrics['delay_ps'],
                           area_feasible=metrics['area']<=read(folder/'anchor/result.json')['area']*1.05)
            except Exception as exc:row.update(status='FAIL',error=f'{type(exc).__name__}: {exc}')
            save_json(target.parent/'observation.json',row)
            print(json.dumps({'phase':'measured','circuit':r['circuit_id'],'combination':c['index'],'status':row['status']}),flush=True)
    save_json(args.output/'measurement_complete.json',{'finished_unix_ns':time.time_ns(),'gpuls_sha256':digest(args.gpuls),
              'source_commit':os.environ['CUT_SOURCE_COMMIT'],'predictions_freeze_sha256':digest(args.output/'PREDICTIONS_FROZEN.json')})


def report(args):
    manifest=read(args.output/'panel.json');rows=[];exceptions=[];evidence=[]
    for r in manifest['circuits']:
        folder=Path(r['folder']);predictions=read(folder/'predictions.json')['rows']
        for c in predictions:
            path=folder/'measurements'/f'combo_{c["index"]:02d}'/'observation.json'
            record=read(path) if path.exists() else {'status':'MISSING'}
            rows.append([c['circuit'],c['predicted_mapping_delay_ps'],record.get('actual_abc_mapping_delay_ps','N/A')])
            if record['status']!='PASS' or not record.get('area_feasible',False):
                exceptions.append({'circuit':c['circuit'],'status':record['status'],'area_feasible':record.get('area_feasible'),'error':record.get('error')})
            evidence.append({'circuit':c['circuit'],'path':str(path),'sha256':digest(path) if path.exists() else None})
    assert len(rows)==96 and len({r[0] for r in rows})==96
    args.report.mkdir(parents=True,exist_ok=True)
    header=['circuit','predicted_mapping_delay_ps','actual_abc_mapping_delay_ps']
    with (args.report/'PREDICTED_VS_ABC_DELAY.csv').open('w') as f:
        w=csv.writer(f);w.writerow(header);w.writerows(rows)
    text='| 电路/组合编号 | 模型预测 Mapping Delay（ps） | 真实 ABC Mapping Delay（ps） |\n|---|---:|---:|\n'
    for circuit,pred,actual in rows:text+=f'| {circuit} | {pred:.6f} | '+(f'{actual:.6f}' if isinstance(actual,(float,int)) else actual)+' |\n'
    text+='\n本轮采用统一 AIG 输入、Liberty Cut/引脚传播及冻结 LiteCut 局部尺度；全部预测先冻结，再真实 Replay/Mapping，无实测回退或改选。'
    text+=f' 固定 96 行，异常/面积违规 {len(exceptions)} 项；逐项说明与证据见 IMPLEMENTATION.md。数值是代理估计与真实 ABC_SCL_LIBERTY 对照，不代表预测准确性已成立。\n'
    (args.report/'RESULTS.md').write_text(text)
    save_json(args.output/'report_audit.json',{'rows':96,'exceptions':exceptions,'evidence':evidence,
                'mapping_calls':len(list(args.output.glob('case_*/**/mapping.command.json'))),
                'prediction_freeze_sha256':digest(args.output/'PREDICTIONS_FROZEN.json')})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['stage','predict','measure','report'])
    p.add_argument('--data',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--gpuls',type=Path)
    p.add_argument('--report',type=Path);a=p.parse_args()
    if a.phase!='report' and (not os.environ.get('SLURM_JOB_ID') or not ROOT.is_relative_to('/data/wushangge')):
        p.error('Run immutable /data source under Slurm')
    globals()[a.phase](a)


if __name__=='__main__':main()
