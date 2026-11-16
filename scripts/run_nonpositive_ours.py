#!/usr/bin/env python3
"""Gated full905 Ours only, fixed STE Adam, independent cold processes."""
import argparse
from collections import Counter
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from global_timing.common import digest, run_tool, save_json
from run_nonpositive_coverage import ZERO17, counts


def candidate_counts(work):
    """Read completed boundaries only; never equate unavailable data with zero."""
    from global_timing.candidates import Catalogue
    result = {}
    path = work / 'candidates/raw/gate3c_raw_manifest.json'
    if path.exists():
        result['gpu_manifest'] = json.loads(path.read_text())
    for key, name in [('raw', 'candidates/catalogue.json'), ('verified', 'cut_validation/verified_catalogue.json')]:
        path = work / name
        if path.exists():
            data = json.loads(path.read_text())
            result[key] = counts(Catalogue(data['decisions'], [], {}))
            if key == 'verified' and (work / 'candidate_feasibility.json').exists():
                blocked = json.loads((work / 'candidate_feasibility.json').read_text())['blocked_choices']
                denied = {(v['decision_index'], v['choice_index']) for v in blocked}
                class Allowed:
                    pass
                f = Allowed()
                f.blocked = [[None] + ['blocked' if (i,c) in denied else None
                    for c in range(1,d['candidate_count']+1)] for i,d in enumerate(data['decisions'])]
                result['feasible'] = counts(Catalogue(data['decisions'],[],{}),f)
                result['blocked_reasons'] = dict(Counter(v['reason'] for v in blocked))
    return result


def child(context):
    import torch
    from global_timing.mffc_ste_experiment import SteExperiment, NonpositiveSteConfig
    torch.set_num_threads(1)
    data = json.loads(context.read_text())
    tools, outer = data['tools'], context.parent
    assert digest(outer/'input.aig') == data['input']['sha256']
    experiment = SteExperiment(outer/'input.aig', tools['gpuls']['path'], tools['abc']['path'],
        tools['library']['path'], tools['checkpoint']['path'], tools['native_library']['path'],
        outer/'work', NonpositiveSteConfig(), validation_workers=2)
    result = experiment.run()
    result['torch'] = {'version':torch.__version__, 'threads':torch.get_num_threads(),
        'cuda_device':torch.cuda.get_device_name() if torch.cuda.is_initialized() else None}
    save_json(outer/'child_result.json', result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--coverage-run', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--child', action='store_true')
    parser.add_argument('--context', type=Path)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not ROOT.is_relative_to('/data/wushangge'):
        parser.error('Requires committed /data source and Slurm.')
    if args.child:
        return child(args.context)
    from global_timing.mffc_ste_experiment import NonpositiveSteConfig
    NonpositiveSteConfig().validate()
    run = args.run.resolve()
    lock = (run/f'shard_{args.shard}.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    gate_path = args.coverage_run/'output/COVERAGE_STATUS.json'
    gate = json.loads(gate_path.read_text())
    old = json.loads((args.coverage_run/'input/coverage_protocol.json').read_text())
    sample = {r['index'] for r in old['rows']}
    records = {r['index']:r for r in gate['records']}
    if (not gate['GO_905'] or len(sample)!=100 or set(records)!=sample
        or any(r['status']!='PASS' for r in records.values())
        or any(any(records[i][k]['candidates']<=0 for k in ('raw','verified','feasible')) for i in ZERO17)):
        raise ValueError('Original17 and complete fixed100 gate not satisfied; refusing full905.')
    inventory = json.loads(args.manifest.read_text())
    rows = inventory['rows']
    expected = json.loads((ROOT/'analysis/ste_adam_100/config/sampled_100_manifest.json').read_text())['manifest_sha256']
    if (digest(args.manifest)!=expected or len(rows)!=905 or sorted(r['index'] for r in rows)!=list(range(905))
        or len({r['relative_path'] for r in rows})!=905):
        raise ValueError('Must use the entire frozen original905 manifest.')
    assert 0<=args.shard<args.shards<=8
    meta = json.loads((run/'run.json').read_text())
    tools = old['tools']
    for item in tools.values():
        assert digest(Path(item['path'])) == item['sha256']
    # The independent binary is from this exact native implementation.
    for name in ['projects/culs/src/algorithms/rewrite.cu','projects/culs/src/algorithms/rewrite.h',
                 'global_timing/candidates.py','global_timing/execution.py','global_timing/mffc_ste_experiment.py',
                 'global_timing/mffc_ste_search.py','global_timing/mffc_model.py','global_timing/feasibility.py']:
        assert digest(ROOT/name)==digest(args.coverage_run/'src'/name)
    session = {'source_commit':meta['source_commit'], 'tools':tools, 'search':asdict(NonpositiveSteConfig()),
        'slurm_job_id':os.environ['SLURM_JOB_ID'], 'node':os.environ.get('SLURMD_NODENAME'),
        'manifest':str(args.manifest), 'manifest_sha256':digest(args.manifest),
        'gate':str(gate_path), 'gate_sha256':digest(gate_path), 'shard':args.shard,'shards':args.shards,
        'case_timeout_s':1800,'validation_workers':2,'methods':['Ours']}
    save_json(run/'input'/f'full905_session_{args.shard}.json',session)
    records = []
    for row in rows:
        if row['index']%args.shards!=args.shard:
            continue
        started=time.perf_counter()
        outer=run/'output'/f'case_{row["index"]:04d}_Ours'
        outer.mkdir(exist_ok=False)
        assert digest(Path(row['staged_path']))==row['sha256']
        shutil.copyfile(row['staged_path'],outer/'input.aig')
        context=outer/'context.json'
        save_json(context,{'input':{k:row[k] for k in ('index','relative_path','sha256','staged_path')},
            'tools':tools,'source_commit':meta['source_commit'],'search':session['search']})
        cache=run/'cache'/str(row['index']);cache.mkdir()
        environment=dict(os.environ,PYTHONPATH=str(ROOT),CUDA_CACHE_PATH=str(cache),
            XDG_CACHE_HOME=str(cache),TORCH_EXTENSIONS_DIR=str(cache/'torch'),PYTHONDONTWRITEBYTECODE='1')
        save_json(outer/'pre_launch.json',{'elapsed_s':time.perf_counter()-started})
        result,error=None,None
        try:
            run_tool([sys.executable,'-B',__file__,'--child','--context',str(context)],outer,'cold_method',1800,environment)
            result=json.loads((outer/'child_result.json').read_text())
            status=result['status']
        except Exception as exc:
            error=f'{type(exc).__name__}: {exc}'
            command=json.loads((outer/'cold_method.command.json').read_text())
            status='TIMEOUT' if command.get('timed_out') else 'FAIL'
        full=time.perf_counter()-started
        work=outer/'work'
        second_path=work/'final/eval_0000/evaluation/mapping.command.json'
        if second_path.exists():
            second=json.loads(second_path.read_text())['elapsed_s']
        elif (second_path.parent/'mapping.log').exists():
            second=None  # interrupted before the nested command clock was saved
        else:
            second=0.0
        record={'index':row['index'],'input':row['relative_path'],'input_sha256':row['sha256'],
            'method':'Ours','status':status,'error':error,'source_commit':meta['source_commit'],
            'slurm_job_id':os.environ['SLURM_JOB_ID'],'detail_path':str(outer),
            'measured_full_runtime_s':full,'second_mapping_abc_time_s':second,
            'optimization_runtime_s':full-second if second is not None else None,
            'mapping_calls':len(list(work.rglob('mapping.command.json'))),
            'mapping_calls_with_missing_clock':sum(not p.with_name('mapping.command.json').exists() for p in work.rglob('mapping.log')),
            'candidate_counts':candidate_counts(work)}
        if result:
            record.update(original=result['original'],selected=result['selected'],final_candidate=result['final_candidate'],
                optimizer_selected_nonempty=bool(result['proxy_best'] and any(result['proxy_best']['choices'])),
                adam_steps=result['adam_steps'],backend=result['backend'])
        save_json(run/'output/records'/f'{row["index"]:04d}_Ours.json',record)
        records.append(record)
        print(json.dumps({k:record[k] for k in ('index','status','measured_full_runtime_s','optimization_runtime_s')}),flush=True)
    save_json(run/'output'/f'completion_shard_{args.shard}.json',{'count':len(records),
        'states':dict(Counter(r['status'] for r in records)),'source_commit':meta['source_commit']})

if __name__=='__main__':
    main()
