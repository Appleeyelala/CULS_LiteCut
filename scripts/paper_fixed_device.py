"""Compare synchronized CPU and CUDA forward/backward at identical candidate weights."""
from pathlib import Path
import sys
import time
import torch

from paper_joint_evidence import FrozenCase, ROOT, read, save_json
from global_timing.boolean_candidates import parse_library
from global_timing.feasibility import CandidateFeasibility
from global_timing.liberty_cover import Liberty
from global_timing.load_bank import LoadBank
from global_timing.structure import parse_aig


def measure(case, device, rules, rule, aig):
    bank = LoadBank(aig, case.cat, rule, read(case.prep/'scores.json')['scores'],rules,
                    Liberty(case.old/'original/tables',device),case.tools['abc'],case.tools['liberty'],
                    case.out/device,.5)
    def sync():
        if device=='cuda':
            torch.cuda.synchronize()
    sync()
    hard=[]
    with torch.no_grad():
        for i,vector in enumerate((case.zero,case.selected)):
            sync()
            begin=time.perf_counter()
            hard.append(float(bank.forward(bank.hard(vector))[0]))
            save_json(case.out/device/f'hard_{i}.json',dict(delay_ps=hard[-1],seconds=time.perf_counter()-begin))
    tensors=[]
    times=[]
    # Identical initial values on both devices, with and without exact logit ties.
    generator=torch.Generator().manual_seed(20261117)
    for values in (torch.zeros(bank.states),torch.randn(bank.states,generator=generator)*.1):
        z=values.to(device).requires_grad_()
        sync()
        start=time.perf_counter()
        delay,_=bank.forward(bank.soft(z))
        sync()
        middle=time.perf_counter()
        (delay/hard[0]).backward()
        sync()
        end=time.perf_counter()
        times.append(dict(forward=middle-start,backward=end-middle,delay=float(delay)))
        tensors.append(z.grad.detach().cpu().double())
        assert torch.isfinite(tensors[-1]).all()
        save_json(case.out/device/f'soft_{len(times)-1}.json',dict(**times[-1],gradient_norm=float(tensors[-1].norm())))
    return dict(hard=hard,soft=times,peak_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None),tensors


def main(out,names):
    out.mkdir()
    rules=parse_library(ROOT/'projects/culs/src/algorithms/rewrite_library.inc')
    for name in names:
        case=FrozenCase(name,out/name)
        aig=parse_aig(case.source)
        rule=CandidateFeasibility(aig,case.cat,rules)
        cpu,left=measure(case,'cpu',rules,rule,aig)
        torch.cuda.reset_peak_memory_stats()
        gpu,right=measure(case,'cuda',rules,rule,aig)
        grads=[]
        for a,b in zip(left,right):
            scale=float(a.norm()*b.norm())
            grads.append(dict(norm_cpu=float(a.norm()),norm_gpu=float(b.norm()),max_abs_diff=float((a-b).abs().max()),
                              cosine=float(torch.dot(a,b))/scale if scale else None))
        save_json(case.out/'devices.json',dict(cpu=cpu,cuda=gpu,gradient_comparison=grads,
                                              cpu_threads=torch.get_num_threads(),gpu=torch.cuda.get_device_name()))
        print(name,grads,flush=True)


if __name__=='__main__':
    torch.set_num_threads(1)
    main(Path(sys.argv[1]),sys.argv[2:])
