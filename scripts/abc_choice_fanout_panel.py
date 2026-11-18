"""Compare the native logarithmic fanout penalty on a shared conditional cut bank."""
import argparse
import csv
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from abc_choice_reference import read
from abc_mode0_panel import write_csv
from global_timing.abc_choices import choice_program
from global_timing.boolean_candidates import parse_library
from global_timing.common import digest, legacy, run_tool, save_json, tool_path
from global_timing.structure import parse_aig


def measure(args, folder, lib, reference):
    frozen = read(folder/'frozen.json')
    i = frozen['circuit_id']
    original = Path(frozen['source'])/'input.aig'
    if digest(original) != frozen['input_sha256']:
        raise ValueError('Input drift')
    aig = parse_aig(original)
    cat = read(folder/'catalogue.json')['decisions']
    scores = read(folder/'scores.json')['scores']
    vectors = [r['choices'] for r in frozen['combinations']]
    if [r['index'] for r in frozen['combinations']] != list(range(9)):
        raise ValueError('Invalid frozen vector order')
    rows = []
    for factor, alpha in ((0, 0.), (100, 0.), (100, .5)):
        d = args.output / f'{i}_{factor}_{alpha}'
        d.mkdir()
        lines, _ = choice_program(aig, cat, vectors, scores, lib, alpha)
        spec = d/'spec'
        spec.write_text('\n'.join(lines)+'\n')
        cmd = f'read_lib {tool_path(args.liberty)}; read_aiger {tool_path(original)}; map -F {factor}'
        log = run_tool([args.abc, '-c', cmd], d, 'hard', 900, dict(os.environ,
                       CULS_CHOICES=str(spec), CULS_HARD_VECTORS=str(len(vectors))))
        results = [line.split() for line in log.splitlines() if line.startswith('CULS_HARD ')]
        if [int(v[1]) for v in results] != list(range(9)) or any(int(v[4]) for v in results):
            raise ValueError('Incomplete hard sweep')
        for _, k, score, _, _ in results:
            k, score = int(k), float(score)
            if not factor and reference[i, k] != score:
                raise ValueError('Native conditional forward drift')
            rows.append(dict(circuit=i, index=k, edits=sum(bool(c) for c in vectors[k]),
                             factor=factor, alpha=alpha, score=score))
    return rows


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('panel', 'abc', 'liberty', 'reference', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    if not os.getenv('SLURM_JOB_ID') or not args.output.resolve().is_relative_to('/data/wushangge'):
        p.error('Use a /data Slurm snapshot')
    with args.reference.open() as f:
        ref = {(int(r['circuit']), int(r['index'])): float(r['hard']) for r in csv.DictReader(f)}
    ids = sorted({i for i, _ in ref})
    save_json(args.output/'protocol.json', dict(factors=[0,100,100], alphas=[0.,0.,.5], ids=ids,
              choice_rule='min(score, edits, index) over 1..8; no area filter',
              reason='100 adds one inverter worst delay per doubling of native fanout; no fitting',
              abc_sha256=digest(args.abc), liberty_sha256=digest(args.liberty), reference_sha256=digest(args.reference)))
    lib = parse_library(ROOT/'projects/culs/src/algorithms/rewrite_library.inc')
    legacy('gate3b_analyze')
    with ThreadPoolExecutor(max_workers=int(os.environ['SLURM_CPUS_PER_TASK'])) as pool:
        groups = list(pool.map(lambda i: measure(args, args.panel/f'case_{i:04d}', lib, ref), ids))
    write_csv(args.output/'predictions.csv', [r for g in groups for r in g])
