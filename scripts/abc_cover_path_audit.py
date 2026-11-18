"""Trace two real mapped covers, and score initial covers on an offline frozen panel."""
import argparse
import csv
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from global_timing.common import digest, run_tool, save_json, tool_path
from global_timing.evaluation import parse_timing
from global_timing.path_analysis import parse_paths
from abc_mode0_panel import write_csv

DETAIL = {75: (6, 8), 179: (1, 7), 44: (1, 7)}


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def measure(args, inputs, circuit):
    rows = []
    for k in range(1, 9):
        native = inputs[circuit, k, 0.]
        installed = Path(native['installed'])
        if digest(installed) != native['sha256']:
            raise ValueError('Frozen replay changed')
        with installed.open('rb') as f:
            npi = int(f.readline().split()[2])
        detail = k in DETAIL.get(circuit, ())
        scores = {}
        for stage, alpha in [('initial', 0.), ('initial', .5)] + ([('final', 0.)] if detail else []):
            r = inputs[circuit, k, alpha]
            tag = args.pins / f'case_{circuit:04d}/v{k:02d}/a0.5/pins.txt'
            if alpha and not int(r['tags']):
                scores[stage, alpha] = scores[stage, 0.]
                continue
            d = args.output / f'case_{circuit:04d}_{k}_{stage}_{alpha}'
            d.mkdir()
            trace = detail and alpha == 0
            env = dict(os.environ, CULS_GLOBAL_TIMING_REPORT='1', CULS_COVER_TRACE='1')
            if trace:
                env.update(CULS_GLOBAL_TIMING_PATH_REPORT='1', CULS_ARC_AUDIT='1')
            if alpha:
                env['CULS_CHOICES'] = str(tag)
            command = (f'read_lib {tool_path(args.liberty)}; read_aiger {tool_path(installed)}; '
                       + ('map -r -v' if stage == 'initial' else 'map -v') + '; topo; stime -p')
            if trace:
                command += '; write_genlib gates.genlib; write_verilog mapped.v; print_delay'
                if circuit == 179:
                    command += '; print_delay po0006; print_delay po1035; print_delay po0003'
                if circuit == 75:
                    command += '; print_delay po090; print_delay po092'
            log = run_tool([args.abc, '-c', command], d, 'abc', 900, env)
            timing = parse_timing(log)
            mode0, = [float(s.split()[1]) for s in log.splitlines() if s.startswith('CULS_MODE0 ')]
            if not math.isclose(mode0, float(r['prediction']), abs_tol=1e-4, rel_tol=0):
                raise ValueError((circuit, k, alpha, 'native mismatch', mode0))
            if stage == 'final' and timing['delay_ps'] != float(r['real_ps']):
                raise ValueError('Final cover delay drift')
            scores[stage, alpha] = timing['delay_ps']
            if trace:
                paths = parse_paths(log, timing, npi)['paths']
                audit = {(int(v[1]), int(v[2])): v[3:] for line in log.splitlines()
                         if (v := line.split()) and v[0] == 'GT_AUDIT'}
                for path in paths:
                    for arc, node in zip(path['arcs'], path['nodes']):
                        pin, fanout, nominal, load_only, slew_only, max_slew, max_load = audit[
                            path['po_index'], arc['sequence']]
                        arc.update(cell=node['cell'], pin_name=pin, fanout=int(fanout),
                                   nominal_ps=float(nominal), load_only_ps=float(load_only),
                                   slew_only_ps=float(slew_only), table_max_slew=float(max_slew),
                                   table_max_load=float(max_load))
                save_json(d/'trace.json', dict(paths=paths, timing=timing, sha256=digest(installed)))
        rows.append(dict(circuit=circuit, index=k, edits=native['edits'], native=float(native['native']),
                         initial_scl=scores['initial', 0.], assisted_scl=scores['initial', .5],
                         final_ps=native['real_ps'], area_ratio=native['area_ratio']))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('abc', 'liberty', 'pins', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--ids', nargs='+', type=int)
    args = p.parse_args()
    if not os.getenv('SLURM_JOB_ID') or not args.output.resolve().is_relative_to('/data/wushangge'):
        p.error('Use a /data Slurm snapshot')
    src = args.pins/'combinations.csv'
    inputs = {(int(r['circuit']), int(r['index']), float(r['alpha'])): r for r in read_csv(src)}
    ids = args.ids or sorted({key[0] for key in inputs})
    save_json(args.output/'protocol.json', dict(abc_sha256=digest(args.abc), liberty_sha256=digest(args.liberty),
              input_table_sha256=digest(src), ids=ids, selection='(score, edits, index)', area_filter=False,
              methods=['native', 'initial_scl', 'assisted_scl'], diagnostic_offline_mapping=True))
    with ThreadPoolExecutor(max_workers=int(os.environ['SLURM_CPUS_PER_TASK'])) as pool:
        groups = list(pool.map(lambda i: measure(args, inputs, i), ids))
    write_csv(args.output/'panel.csv', [r for group in groups for r in group])


if __name__ == '__main__':
    main()
