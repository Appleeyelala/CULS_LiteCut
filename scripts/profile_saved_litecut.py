#!/usr/bin/env python3
"""Bounded function profile on evenly spaced saved decisions; no search/Mapping."""
import argparse
import cProfile
import json
import os
from pathlib import Path
import pstats
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-run', type=Path, required=True)
    p.add_argument('--circuit', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--decisions', type=int, default=64)
    args = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not ROOT.is_relative_to('/data/wushangge'):
        p.error('Requires committed /data source and Slurm.')
    import torch
    from global_timing.candidates import Catalogue
    from global_timing.litecut import score_catalogue
    from global_timing.structure import parse_aig
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.reference_run/'output'/f'case_{args.circuit:04d}_Ours'
    context = json.loads((source/'context.json').read_text())
    started = time.perf_counter()
    data = json.loads((source/'work/cut_validation/verified_catalogue.json').read_text())
    decisions = data['decisions']
    count = min(args.decisions, len(decisions))
    if count < 2:
        raise ValueError('At least two decisions required for the spaced probe.')
    indices = [i*(len(decisions)-1)//(count-1) for i in range(count)]
    catalogue = Catalogue([decisions[i] for i in indices], [], {})
    total = len(decisions)
    del data, decisions
    partition = SimpleNamespace(aig=parse_aig(source/'input.aig'))
    preparation = time.perf_counter()-started
    profiler = cProfile.Profile()
    started = time.perf_counter()
    with profiler:
        scores, cache = score_catalogue(source/'input.aig', partition, catalogue,
            Path(context['tools']['checkpoint']['path']), args.output/'cache')
    elapsed = time.perf_counter()-started
    profiler.dump_stats(str(args.output/'profile.pstats'))
    stats = pstats.Stats(profiler)
    rows = [dict(file=k[0], line=k[1], function=k[2], calls=v[1], self_s=v[2], cumulative_s=v[3])
            for k,v in stats.stats.items()]
    rows.sort(key=lambda r:r['cumulative_s'], reverse=True)
    result = dict(circuit=args.circuit, total_decisions=total, selected_decision_indices=indices,
        score_rows=len(scores), cache=cache, input_loading_s=preparation, profiled_litecut_s=elapsed,
        top_functions=rows[:100], mapping_calls=0, search_calls=0,
        scope='64 evenly spaced archived decisions only; instrumentation timing is not full-case runtime and must not be extrapolated as measured full-case time.')
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['top_functions','cache']}))


if __name__=='__main__':
    main()
