"""Independently explain a saved candidate-0 discrepancy on its original AIG."""
import argparse
import csv
import gzip
from itertools import product
import json
import os
from pathlib import Path

from global_timing.common import ROOT, digest, save_json
from global_timing.structure import parse_aig
from aig_level_experiment import abc_depth, frozen_tools


def paths_to_root(aig, node, boundary=()):
    """Enumerate node paths in this small failing cone, from frontier to root."""
    if node in boundary or node not in aig.fanin_literals:
        yield (node,)
        return
    for lit in aig.fanin_literals[node]:
        for path in paths_to_root(aig, lit >> 1, boundary):
            yield path + (node,)


def projected_output(path, root, output):
    # Preserve the AND section byte-for-byte, dropping obsolete PO symbols.
    with path.open('rb') as stream:
        header = stream.readline().split()
        for _ in range(int(header[4])):
            stream.readline()
        data = stream.read()
    left = 2 * int(header[5])
    end = 0
    while left:
        left -= int(data[end] < 128)
        end += 1
    header[4] = b'1'
    output.write_bytes(b' '.join(header) + b'\n' + str(2 * root).encode() + b'\n' + data[:end])


def check_boundary_truth(aig, root, leaves, frontier_paths, truth):
    external = sorted({p[0] for p in frontier_paths} - set(leaves) - {0})
    variables = leaves + external
    nodes = sorted({n for p in frontier_paths for n in p[1:]})
    checked = 0
    for bits in product((0, 1), repeat=len(variables)):
        values = {0: 0, **dict(zip(variables, bits))}
        for node in nodes:
            a, b = aig.fanin_literals[node]
            values[node] = (values[a >> 1] ^ (a & 1)) & (values[b >> 1] ^ (b & 1))
        index = sum(bits[i] << i for i in range(len(leaves)))
        if values[root] != (truth >> index) & 1:
            raise ValueError('Original cone with free boundary variables differs from native truth')
        checked += 1
    return external, checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--failed-run', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    if not os.getenv('SLURM_JOB_ID') or ROOT != run / 'src' or not run.is_relative_to('/data/wushangge'):
        parser.error('Run the diagnostic in its committed Slurm snapshot')
    failed = args.failed_run.resolve()
    out = run / 'output'
    failure = json.loads((failed / 'output/failure.json').read_text())
    case = failure['circuit']
    pool = failed / 'output' / case / 'pool'
    saved_path = pool / 'noop_discrepancy.json'
    witness = json.loads(saved_path.read_text())
    aig_path = failed / 'input/circuits' / (case + '.aig')
    if digest(aig_path) != witness['input_sha256']:
        raise ValueError('Failing input identity changed')
    aig = parse_aig(aig_path)
    root, leaves = witness['root'], witness['leaves']
    full_paths = list(paths_to_root(aig, root))
    cut_paths = list(paths_to_root(aig, root, leaves))
    lengths = [max((len(p) - 1 for p in cut_paths if p[0] == leaf), default=None) for leaf in leaves]
    arrivals = [max(map(len, paths_to_root(aig, leaf))) - 1 for leaf in leaves]
    prediction = max(a + d for a, d in zip(arrivals, lengths) if d is not None)
    true_level = max(map(len, full_paths)) - 1
    bypass = max((p for p in full_paths if set(leaves).isdisjoint(p)), key=len)
    if (prediction != witness['cut_prediction'] or true_level != witness['actual_root_level']
            or prediction >= true_level or lengths != witness['leaf_depths'] or arrivals != witness['leaf_levels']
            or len(bypass) - 1 != true_level):
        raise ValueError('Independent original path enumeration differs from the saved discrepancy')
    external, checked = check_boundary_truth(aig, root, leaves, cut_paths, witness['decision']['truth'])
    root_aig = out / 'root_as_po.aig'
    projected_output(aig_path, root, root_aig)
    copy = parse_aig(root_aig)
    if (copy.fanin_literals, copy.pi_count, copy.po_literals) != (aig.fanin_literals, aig.pi_count, (2 * root,)):
        raise ValueError('Output projection changed original fanins or PI order')
    abc, _ = frozen_tools(run)
    independent_abc = abc_depth(abc, root_aig, out)
    if independent_abc != true_level:
        raise ValueError('Frozen ABC disagrees on the isolated original root')
    table_counts = {}
    for table in sorted((failed / 'output').glob('*/pool/candidates.csv.gz')):
        with gzip.open(table, 'rt', newline='') as stream:
            counts = {'noops': 0, 'physical': 0}
            for row in csv.DictReader(stream):
                counts['noops' if row['choice_index'] == '0' else 'physical'] += 1
            table_counts[table.parents[1].name] = counts
    save_json(out / 'witness.json', dict(failed_run=str(failed), failure_sha256=digest(saved_path),
              input_sha256=digest(aig_path), original_root=root, cut_slot=witness['cut_slot'], leaves=leaves,
              leaf_levels=arrivals, leaf_depths=lengths, leaf_only_level=prediction, actual_level=true_level,
              abc_root_as_only_po=independent_abc, abc_sha256=digest(abc), projected_aig_sha256=digest(root_aig),
              frontier_paths=cut_paths, longest_bypass_path=bypass, outside_pis=external,
              exhaustive_boundary_assignments=checked, native_truth=witness['decision']['truth'],
              saved_candidate_rows=table_counts))
    manifest = run / 'run.json'
    data = json.loads(manifest.read_text())
    data['outputs'] = str(out)
    manifest.write_text(json.dumps(data, indent=2) + '\n')


if __name__ == '__main__':
    main()
