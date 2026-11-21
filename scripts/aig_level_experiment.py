"""Run the ordered structural AIG experiment from a committed Slurm snapshot."""
import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import re
import shutil
import time
import unittest

from global_timing.common import ROOT, digest, run_tool, save_json, tool_path
from global_timing.level_candidates import LibraryDepths, export_pool
from global_timing.level_weights import propagate_levels
from global_timing.structure import make_partition, parse_aig

PILOT = ('c1355', 'c5315', 'intra_pred_PE', 'cabac_ucontext')
PAPER_INPUTS = Path('/data/wushangge/CULS_LiteCut/runs/20261119T020347Z-litecut-ours14-af7ccd61/input/circuits')


def load(path):
    return json.loads(path.read_text())


def csv_rows(path, columns, rows):
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def pilot_inputs(run):
    with (ROOT / 'analysis/paper14_fullflow/circuits.csv').open() as manifest:
        paper = {r['circuit']: (PAPER_INPUTS / r['filename'], r['sha256']) for r in csv.DictReader(manifest)}
    paper['c1355'] = (run / 'input/design.aig', '2602b10f131a6ba5996521036ba130aab128a4aa15e3ab49ce44f2f02047f0b0')
    return {name: paper[name] for name in PILOT}


def stage_inputs(run):
    staged = {}
    inventory = []
    folder = run / 'input/circuits'
    folder.mkdir()
    for name, (path, sha) in pilot_inputs(run).items():
        target = folder / (name + '.aig')
        shutil.copyfile(path, target)
        if digest(target) != sha:
            raise ValueError(f'{name} input digest differs from the frozen cohort')
        inventory.append(dict(circuit=name, source=str(path), path=str(target), sha256=sha))
        staged[name] = target
    save_json(folder / 'inventory.json', inventory)
    return staged


def frozen_tools(run):
    paths = load(ROOT / 'config/load_rewrite_tools.json')
    hashes = load(ROOT / 'config/ste100_protocol.json')['frozen_tool_sha256']
    abc, lib = Path(paths['frozen_abc']), run / 'input/asap7'
    for name, path in [('abc', abc), ('library', lib)]:
        if digest(path) != hashes[name]:
            raise ValueError(f'{name} differs from STE100')
    return abc, lib


def abc_depth(abc, aig, output):
    log = run_tool([abc, '-c', f'read_aiger {tool_path(aig)}; print_stats'], output, 'abc_levels')
    value = re.findall(r'\blev\s*=\s*(\d+)', log)
    if len(value) != 1:
        raise ValueError(f'Missing or ambiguous ABC level in {output}')
    return int(value[0])


def original_levels(abc, name, path, output):
    start = time.monotonic()
    aig = parse_aig(path)
    levels = propagate_levels(aig)
    partition = make_partition(aig)
    partition.write(output / 'partition')
    rows = [dict(region=r['region_id'], root=r['root'], level=levels.level[r['root']],
                 req=levels.required[r['root']], slack=levels.slack(r['root'])) for r in partition.regions]
    csv_rows(output / 'roots.csv', ['region', 'root', 'level', 'req', 'slack'], rows)
    unreachable = [n for n, req in enumerate(levels.required) if req is None]
    csv_rows(output / 'unreachable.csv', ['node', 'kind', 'level'],
             (dict(node=n, kind='AND' if n > aig.pi_count else 'PI' if n else 'CONST',
                   level=levels.level[n]) for n in unreachable))
    hist = Counter(r['slack'] for r in rows if r['slack'] is not None)
    po_levels = Counter(levels.level[lit >> 1] for lit in aig.po_literals)
    result = dict(circuit=name, input_sha256=digest(path), ands=aig.and_count, D=levels.depth,
                  abc_D=abc_depth(abc, path, output), strict_roots=len(rows),
                  zero_slack_roots=hist[0], root_slack_histogram=dict(hist),
                  po_level_histogram=dict(po_levels), deepest_outputs=po_levels[levels.depth],
                  unreachable_nodes=len(unreachable), unreachable_roots=len(rows) - sum(hist.values()),
                  wall_s=time.monotonic() - start)
    save_json(output / 'level_check.json', result)
    if levels.depth != result['abc_D']:
        raise ValueError(f"{name} original D mismatch: Python {levels.depth}, ABC {result['abc_D']}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--cohort', choices=['pilot'], required=True,
                        help='The development and external cohorts follow all five pilot gates.')
    parser.add_argument('--through', type=int, choices=[1, 2], default=1)
    parser.add_argument('--native-build', type=Path)
    args = parser.parse_args()
    if args.through >= 2 and not args.native_build:
        parser.error('--native-build is required for native candidate stages')
    run = args.run.resolve()
    if not os.getenv('SLURM_JOB_ID') or not run.is_relative_to('/data/wushangge') or ROOT != run / 'src':
        parser.error('Use the committed snapshot inside its /data Slurm job')
    out = run / 'output'
    stage = 'unit_fixtures'
    case = None
    passed = []
    try:
        suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'), pattern='test_level_weights.py')
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
            raise ValueError('Hand-computed level fixtures failed')
        stage = 'original_levels'
        abc, lib = frozen_tools(run)
        save_json(out / 'tools.json', {'abc': str(abc), 'abc_sha256': digest(abc),
            'liberty': str(lib), 'liberty_sha256': digest(lib), 'model': None})
        staged = stage_inputs(run)
        for case, path in staged.items():
            result = original_levels(abc, case, path, out / case)
            print(case, 'D', result['D'], 'ABC', result['abc_D'], flush=True)
        passed.append(stage)
        if args.through >= 2:
            stage = 'candidate_dags'
            build = args.native_build.resolve()
            gpuls = build / 'build/culs/gpuls'
            reference = Path(load(ROOT / 'config/load_rewrite_tools.json')['gpuls'])
            if (load(build / 'run.json')['state'] != 'COMPLETED'
                    or digest(gpuls) != (build / 'output/checksums.sha256').read_text().split()[0]):
                raise ValueError('Native build has no matching successful binary digest')
            for filename in ['rewrite.cu', 'rewrite.h', 'rewrite_library.inc']:
                rel = Path('projects/culs/src/algorithms') / filename
                if digest(ROOT / rel) != digest(build / 'src' / rel):
                    raise ValueError(f'Native build source differs at {filename}')
            save_json(out / 'native_tools.json', dict(build=str(build), gpuls=str(gpuls),
                      gpuls_sha256=digest(gpuls), reference=str(reference), reference_sha256=digest(reference)))
            library = LibraryDepths()
            for case, path in staged.items():
                result = export_pool(abc, gpuls, reference, path, make_partition(parse_aig(path)),
                                     library, out / case / 'pool')
                print(case, 'verified decisions', result['noops'], 'physical', result['physical'], flush=True)
            passed.append(stage)
        save_json(out / 'gates.json', dict(status='PASS', gates_passed=passed))
    except Exception as exc:
        save_json(out / 'failure.json', dict(stage=stage, circuit=case, gates_passed=passed,
                  error=f'{type(exc).__name__}: {exc}'))
        raise
    finally:
        manifest = run / 'run.json'
        data = load(manifest)
        data['outputs'] = str(out)
        manifest.write_text(json.dumps(data, indent=2) + '\n')


if __name__ == '__main__':
    main()
