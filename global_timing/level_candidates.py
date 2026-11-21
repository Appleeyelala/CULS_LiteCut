"""Unit-AND leaf depths from the verified Gate3C pool, without a model runtime."""
import csv
import gzip
import json
import os
from collections import Counter
from itertools import zip_longest

from .boolean_candidates import parse_library
from .candidates import enumerate_candidates
from .common import ROOT, digest, run_tool, save_json, tool_path
from .cut_validation import validate_catalogue
from .level_weights import candidate_root_level, original_leaf_depths, propagate_levels
from .phase_ports import candidate_transfers

LIBRARY = ROOT / 'projects/culs/src/algorithms/rewrite_library.inc'


class LibraryDepths:
    """Cache materialized library DAG transfers in the original cut-slot order."""
    def __init__(self):
        self.library = parse_library(LIBRARY)
        self.cache = {}

    def get(self, decision, subgraph):
        truth, cls = decision['truth'], decision['truth_class']
        size = len(decision['cut_leaves'])
        key = (truth, cls, size, subgraph)
        if key not in self.cache:
            # Positive port IDs distinguish real cut slots from padded constant 0.
            ports = list(range(1, size + 1))
            template = dict(decision, cut_leaves=ports)
            paths, _ = candidate_transfers(template, subgraph, self.library, inversion_cost=0)
            depths = {}
            for (port, _), length in paths.items():
                depths[port] = max(depths.get(port, 0), int(length))
            self.cache[key] = tuple(depths.get(port) for port in ports)
        return self.cache[key]

    def scalar_level(self, decision, subgraph, arrivals):
        """Independent forward propagation through the replacement vertices."""
        lib = self.library
        perm = lib['perms4'][lib['p_perms'][decision['truth']]]
        inputs = list(arrivals) + [0] * (4 - len(arrivals))
        values = [inputs[i] for i in perm]
        cls = lib['classes'][decision['truth_class']]
        root = cls['roots'][subgraph]
        for a, b, _, _ in cls['fanins'][:max(root - 3, 0)]:
            values.append(1 + max(values[a], values[b]))
        return values[root]


def compare_raw(reference_gpuls, aig, native, work):
    """Repeat the uninstrumented native policy and compare every raw cut in order."""
    env = dict(os.environ, CULS_ALLOW_NONPOSITIVE_CANDIDATES='1')
    env.pop('CULS_EXPORT_LEVEL_DETAILS', None)
    raw = work / 'raw'
    run_tool([reference_gpuls, '-c', f'read {tool_path(aig)}; gate3c_rewrite_decisions_dump -o {tool_path(raw)}'],
             work, 'reference_enumeration', 900, env)
    summary = json.loads((raw / 'gate3c_raw_manifest.json').read_text())
    if summary['candidate_rule'] != 'allow_nonpositive_gain':
        raise ValueError('Reference must use the relaxed enumeration policy')
    cuts = candidates = 0
    filename = 'gpu_rewrite_cuts.jsonl'
    with (raw / filename).open() as before, (native / filename).open() as after:
        for left, right in zip_longest(before, after):
            if left is None or right is None:
                raise ValueError('Different raw cut counts on the same input')
            old, new = json.loads(left), json.loads(right)
            added = new.pop('candidate_nodes_added')
            if len(added) != new['candidate_count'] or new != old:
                raise ValueError(f"Native reference disagrees at original root {new['root']} cut {new['cut_slot']}")
            for n, sign in zip(added, new['candidate_gain_signs'], strict=True):
                gain = new['saved_count'] - n
                if (gain > 0) - (gain < 0) != sign:
                    raise ValueError('Device count and recorded gain sign disagree')
            cuts += 1
            candidates += len(added)
    if (cuts, candidates) != (summary['eligible_cut_count'], summary['candidate_count']):
        raise ValueError('Raw native manifest counts disagree')
    return dict(raw_cuts=cuts, raw_physical=candidates, reference_gpuls_sha256=digest(reference_gpuls))


def export_pool(abc, gpuls, reference, path, partition, library, work):
    identity = digest(path)
    levels = propagate_levels(partition.aig)
    catalog = enumerate_candidates(gpuls, path, partition, work / 'enumeration', 900, True, True)
    audit = compare_raw(reference, path, work / 'enumeration/raw', work / 'reference')
    catalog, validation = validate_catalogue(abc, partition, catalog, work / 'validation', 900)
    # An independent join from the reference enumeration through the saved
    # Boolean/owner filter checks the actual pool used for every scoring arm.
    legal = {(r['root'], r['cut_slot']) for r in validation['entries'] if r['status'] == 'PASS'}
    expected = {}
    with (work / 'reference/raw/gpu_rewrite_cuts.jsonl').open() as stream:
        for line in stream:
            r = json.loads(line)
            key = r['root'], r['cut_slot']
            if key in legal:
                if key in expected:
                    raise ValueError(f'Duplicate original decision {key}')
                expected[key] = (r['cut_leaves'], r['candidate_subgraph_indices'])
    audit.update(input_sha256=identity, verified_decisions=len(legal), excluded_decisions=validation['excluded_decisions'],
                 verified_physical=sum(len(item[1]) for item in expected.values()))
    save_json(work / 'identity_counts.json', audit)
    columns = ['input_sha256', 'root', 'cut_slot', 'choice_index', 'subgraph_index', 'cut_leaves',
               'leaf_depths', 'native_added', 'native_saved', 'and_delta', 'predicted_root_level',
               'original_root_level', 'outside_arrival']
    counts = Counter()
    fixture = None
    with gzip.open(work / 'candidates.csv.gz', 'xt', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(columns)
        for d in catalog.decisions:
            root, slot, leaves = d['root'], d['cut_slot'], d['cut_leaves']
            indices = d['candidate_subgraph_indices']
            if expected.pop((root, slot)) != (leaves, indices) or d['classification'] != 'INTRA_REGION':
                raise ValueError(f'Filtered candidate membership mismatch at {root}:{slot}')
            arrivals = [levels.level[x] for x in leaves]
            depths, outside = original_leaf_depths(partition.aig, root, leaves, levels.level)
            own = candidate_root_level(arrivals, depths)
            fixed = [identity, root, slot]
            ordered = json.dumps(leaves, separators=(',', ':'))
            writer.writerow([*fixed, 0, None, ordered, json.dumps(depths), 0, 0, 0, own, levels.level[root], outside])
            counts['noop'] += 1
            if own != levels.level[root]:
                save_json(work / 'noop_discrepancy.json', dict(input_sha256=identity, root=root, cut_slot=slot,
                          leaves=leaves, leaf_levels=arrivals, leaf_depths=depths, outside_arrival=outside,
                          cut_prediction=own, actual_root_level=levels.level[root], decision=d))
                raise ValueError(f'Candidate 0 at {root}:{slot} predicts {own}, original level is {levels.level[root]}')
            for choice, (sub, added) in enumerate(zip(indices, d['candidate_nodes_added'], strict=True), 1):
                depths = library.get(d, sub)
                predicted = candidate_root_level(arrivals, depths)
                writer.writerow([*fixed, choice, sub, ordered, json.dumps(depths), added, d['saved_count'],
                                 added - d['saved_count'], predicted, levels.level[root], None])
                counts['physical'] += 1
                if identity.startswith('2602b10f131a') and (root, slot, sub) == (151, 5, 0):
                    fixture = dict(root=root, cut_slot=slot, subgraph_index=sub, leaves=leaves, leaf_depths=depths,
                                   truth=d['truth'], truth_class=d['truth_class'], predicted_level=predicted)
                    if leaves != [21, 34, 145, 146] or depths != (4, 4, 3, 3):
                        raise ValueError('Hand-traced permuted c1355 candidate differs')
    if expected or (counts['noop'], counts['physical']) != (audit['verified_decisions'], audit['verified_physical']):
        raise ValueError('Exported pool differs from the filtered native pool')
    if identity.startswith('2602b10f131a') and fixture is None:
        raise ValueError('Missing c1355 hand-traced candidate')
    audit.update(noops=counts['noop'], physical=counts['physical'], permuted_fixture=fixture,
                 library_sha256=digest(LIBRARY))
    save_json(work / 'pool_check.json', audit)
    return audit
