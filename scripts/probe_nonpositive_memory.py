#!/usr/bin/env python3
"""Isolated saved-input reproduction; no enumeration, Replay, CEC or Mapping."""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-run', type=Path, required=True)
    parser.add_argument('--circuit', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not ROOT.is_relative_to('/data/wushangge'):
        parser.error('Run committed /data source through Slurm.')
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    from global_timing.candidates import Catalogue
    from global_timing.feasibility import CandidateFeasibility
    from global_timing.mffc_model import MffcGpuSurrogate
    from global_timing.mffc_ste_search import ste_adam_search
    from global_timing.structure import make_partition, parse_aig

    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    def emit(event, **details):
        row = dict(event=event, epoch_s=time.time(), **details)
        if torch.cuda.is_initialized():
            row.update(allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
                       peak_allocated=torch.cuda.max_memory_allocated())
        with (args.output/'events.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')
        print(json.dumps(row), flush=True)

    class Allocations(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            before = torch.cuda.memory_allocated() if torch.cuda.is_initialized() else 0
            try:
                result = func(*args, **(kwargs or {}))
            except Exception:
                emit('operator_exception', operator=str(func), tensors=[
                    {'shape':list(a.shape), 'layout':str(a.layout), 'dtype':str(a.dtype)}
                    for a in args if isinstance(a, torch.Tensor)])
                raise
            after = torch.cuda.memory_allocated() if torch.cuda.is_initialized() else 0
            if after-before > 128*1024**2:
                emit('large_allocation', operator=str(func), delta=after-before, tensors=[
                    {'shape':list(a.shape), 'layout':str(a.layout), 'dtype':str(a.dtype)}
                    for a in args if isinstance(a, torch.Tensor)])
            return result

    metadata = {k:os.environ.get(k) for k in ['SLURM_JOB_ID','SLURMD_NODENAME',
        'SLURM_JOB_GPUS','SLURM_STEP_GPUS','CUDA_VISIBLE_DEVICES','CUDA_DEVICE_ORDER']}
    metadata.update(torch=torch.__version__, pid=os.getpid(), circuit=args.circuit,
        gpu_inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,pci.bus_id,memory.total,memory.used',
                                               '--format=csv'], text=True),
        gpu_processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory',
                                               '--format=csv'], text=True))
    emit('start', **metadata)
    outer=args.reference_run/'output'/f'case_{args.circuit:04d}_Ours'
    work=outer/'work'
    model=None
    try:
        partition=make_partition(parse_aig(outer/'input.aig'))
        emit('partition_ready')
        data=json.loads((work/'cut_validation/verified_catalogue.json').read_text())
        catalogue=Catalogue(data['decisions'], data['conflicts'],
                            {int(k):v for k,v in data['region_decisions'].items()})
        del data
        emit('catalogue_ready', decisions=len(catalogue.decisions), conflicts=len(catalogue.conflicts))
        feasibility=CandidateFeasibility(partition.aig,catalogue,
                                         ROOT/'projects/culs/src/algorithms/rewrite_library.inc')
        emit('feasibility_ready')
        cache=json.loads((work/'litecut.json').read_text())['scores_path']
        scores=json.loads(Path(cache).read_text())['scores']
        emit('scores_loaded', rows=len(scores))
        baseline=json.loads((work/'original/result.json').read_text())
        context=json.loads((outer/'context.json').read_text())
        original_to=torch.nn.Module.to
        def inspected_to(module, *a, **kw):
            buffers=[]
            for name, value in module.named_buffers():
                if value.layout==torch.sparse_coo:
                    size=value._indices().numel()*8+value._values().numel()*value.element_size()
                    nnz=value._nnz()
                else:
                    size=value.numel()*value.element_size(); nnz=None
                buffers.append(dict(name=name,shape=list(value.shape),bytes=size,nnz=nnz))
            emit('before_model_to', buffers=buffers)
            return original_to(module,*a,**kw)
        torch.nn.Module.to=inspected_to
        with Allocations():
            model=MffcGpuSurrogate(partition,catalogue,scores,baseline,
                                   context['tools']['native_library']['path'])
        torch.nn.Module.to=original_to
        del scores
        gc.collect()
        emit('model_ready')
        with Allocations():
            best, trace=ste_adam_search(model,feasibility)
        emit('ste40_pass', steps=trace['steps'], edits=best['edits'],
             logits_change=trace['logits_total_change_norm'])
        (args.output/'trajectory.json').write_text(json.dumps(trace,indent=2)+'\n')
        compact={k:v for k,v in best.items() if k!='choices'}
        (args.output/'proxy_best.json').write_text(json.dumps(compact,indent=2)+'\n')
        result={'status':'PASS','mapping_calls':0,'steps':trace['steps']}
    except Exception as exc:
        emit('failure', error=str(exc), traceback=traceback.format_exc())
        result={'status':'FAIL','error':str(exc),'mapping_calls':0}
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__':
    raise SystemExit(main())
