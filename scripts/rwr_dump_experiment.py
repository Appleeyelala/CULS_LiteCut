"""Export ABC rewrite candidates for the pilot circuits and check them.

Everything runs inside one Slurm job from a committed snapshot. The checks are
streaming: cabac_ucontext is expected to hold hundreds of thousands of cuts and
the whole export does not fit comfortably in memory as Python objects.

Two invariants are asserted rather than sampled:

* ``leaf_depths`` is indexed by the position in the cut's ``leaves`` array, not
  by the ``slot`` field. The slot records where the leaf came from in the raw
  ABC cut after the NPN permutation; the array order is the order the evaluator
  used. Any analysis that reorders by slot silently mismatches the depths.
* For every candidate, ``root_level`` must equal
  ``max(level[leaf_i] + leaf_depths[i])`` computed in array order.

The first mismatch aborts the run.

ABC numbers objects PI, PO, AND while AIGER numbers PI, AND, so AND ids differ
by a constant offset equal to the PO count. The offset is verified node by node
instead of assumed, and a mismatch aborts the run.
"""
import argparse
import csv
import gzip
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from global_timing.level_weights import original_leaf_depths, propagate_levels
from global_timing.structure import aig_structure_digest, parse_aig

CIRCUITS = [
    ('c1355', '2602b10f131a6ba5996521036ba130aab128a4aa15e3ab49ce44f2f02047f0b0'),
    ('c5315', 'd1c0cea55baf5a407495d82eb4385a9ddea8528cde21a762f46bb0774fc93afd'),
    ('intra_pred_PE', '889b40b408ff0e3333a81f73e99887ee379fb60c6e8c1b5f23d1056ce7c088a4'),
    ('cabac_ucontext', '77fdf62aa46fa9a5bdb67a49ede156bff75bc8ea121faff67b74fa89a2a78707'),
]
PAPER_INPUTS = Path('/data/wushangge/CULS_LiteCut/runs/20261119T020347Z-litecut-ours14-af7ccd61/input/circuits')
STAT_RE = re.compile(r'i/o\s*=\s*(\d+)/\s*(\d+)\s+lat\s*=\s*(\d+)\s+and\s*=\s*(\d+)\s+lev\s*=\s*(\d+)')
GZIP_THRESHOLD = 1024 ** 3
EXAMPLE_CAP = 200


def file_digest(path):
    sha = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            sha.update(chunk)
    return sha.hexdigest()


def run_abc(abc, commands, log_path, env=None, timeout=7200):
    proc = subprocess.run([str(abc), '-c', commands], capture_output=True, text=True,
                          env=env, timeout=timeout)
    Path(log_path).write_text(proc.stdout + proc.stderr)
    return proc


def stats(abc, commands, log_path, env=None):
    proc = run_abc(abc, commands, log_path, env)
    found = STAT_RE.findall(proc.stdout + proc.stderr)
    return [dict(io=(int(a), int(b)), lat=int(c), ands=int(d), lev=int(e)) for a, b, c, d, e in found], proc


def stage_inputs(run, wanted):
    staged = []
    folder = run / 'input/circuits'
    folder.mkdir(parents=True, exist_ok=True)
    available = {file_digest(path): path for path in sorted(PAPER_INPUTS.glob('*.aig'))}
    for name, sha in CIRCUITS:
        if name not in wanted:
            continue
        if name == 'c1355':
            source = run / 'input/design.aig'
        else:
            source = available.get(sha)
            if source is None:
                raise SystemExit(f'{name}: no file under {PAPER_INPUTS} matches {sha}')
        target = folder / f'{name}.aig'
        shutil.copyfile(source, target)
        if file_digest(target) != sha:
            raise SystemExit(f'{name}: input digest differs from the frozen cohort')
        staged.append(dict(circuit=name, source=str(source), path=str(target), sha256=sha))
    (folder / 'inventory.json').write_text(json.dumps(staged, indent=2) + '\n')
    return {row['circuit']: Path(row['path']) for row in staged}


def identifier_mapping(nodes_tsv, aig, circuit):
    """Verify that ABC AND ids equal AIGER ids shifted by a single offset."""
    rows = [r for r in csv.DictReader(open(nodes_tsv), delimiter='\t') if r['kind'] == 'AND']
    if len(rows) != aig.and_count:
        raise SystemExit(f'{circuit}: ABC has {len(rows)} ANDs, AIGER has {aig.and_count}')
    abc_ids = sorted(int(r['abc_id']) for r in rows)
    aig_ids = sorted(aig.fanin_literals)
    offsets = {a - b for a, b in zip(abc_ids, aig_ids)}
    if len(offsets) != 1:
        raise SystemExit(f'{circuit}: AND numbering is not a constant shift: {sorted(offsets)[:5]}')
    offset = offsets.pop()
    if offset != aig.po_count:
        raise SystemExit(f'{circuit}: unexpected offset {offset}, PO count is {aig.po_count}')
    checked = swapped = 0
    for row in rows:
        node = int(row['abc_id']) - offset
        pair = (int(row['fanin0_lit']), int(row['fanin1_lit']))
        truth = aig.fanin_literals.get(node)
        if truth is None:
            raise SystemExit(f'{circuit}: shifted id {node} missing from the AIG')
        mapped = tuple(lit if lit >> 1 <= aig.pi_count else lit - 2 * offset for lit in pair)
        if mapped == tuple(truth):
            checked += 1
        elif (mapped[1], mapped[0]) == tuple(truth):
            checked += 1
            swapped += 1
        else:
            raise SystemExit(f'{circuit}: node {node} fanins {mapped} differ from {tuple(truth)}')
    return dict(circuit=circuit, offset=offset, po_count=aig.po_count,
                and_nodes=len(rows), verified=checked, fanin_order_swapped=swapped)


def check_regression(run, abc_orig, abc_dump, abc_frozen, circuits):
    """Acceptance 1: patched and unpatched builds from one source must agree."""
    out = run / 'output'
    (out / 'aig').mkdir(parents=True, exist_ok=True)
    rows = []
    for name, path in circuits.items():
        record = dict(circuit=name, input_sha256=file_digest(path))
        written = {}
        for tag, abc in [('orig', abc_orig), ('dump', abc_dump), ('frozen', abc_frozen)]:
            target = out / 'aig' / f'{name}.{tag}.aig'
            measured, proc = stats(abc, f'read_aiger {path}; print_stats; rewrite; print_stats;'
                                        f' write_aiger {target}', out / 'logs' / f'{name}.{tag}.log')
            if len(measured) != 2:
                raise SystemExit(f'{name}/{tag}: expected two print_stats blocks, got {len(measured)}')
            record[tag] = dict(before=measured[0], after=measured[1], returncode=proc.returncode,
                               digest=aig_structure_digest(target) if target.exists() else None)
            written[tag] = target
        cec_log = out / 'logs' / f'{name}.cec.log'
        run_abc(abc_orig, f'read_aiger {written["orig"]}; cec {written["dump"]}', cec_log)
        text = cec_log.read_text().lower()
        record['cec_orig_vs_dump'] = ('equivalent' in text) and ('not equivalent' not in text)
        record['primary_match'] = (record['orig']['after']['ands'] == record['dump']['after']['ands']
                                   and record['orig']['after']['lev'] == record['dump']['after']['lev']
                                   and record['orig']['digest'] == record['dump']['digest'])
        record['frozen_matches_dump'] = (record['frozen']['after'] == record['dump']['after']
                                         and record['frozen']['digest'] == record['dump']['digest'])
        rows.append(record)
        print(f"{name}: primary_match={record['primary_match']} cec={record['cec_orig_vs_dump']} "
              f"frozen_match={record['frozen_matches_dump']}", flush=True)
    (out / 'regression.json').write_text(json.dumps(rows, indent=2) + '\n')
    return rows


def export_candidates(run, abc_dump, path, name):
    """Acceptance 2: the export run must leave the network untouched."""
    folder = run / 'output/dump' / name
    if folder.exists():
        raise SystemExit(f'refusing to overwrite an existing export: {folder}')
    folder.mkdir(parents=True)
    log = run / 'output/logs' / f'{name}.export.log'
    env = dict(os.environ, CULS_RWR_DUMP=str(folder))
    measured, proc = stats(abc_dump, f'read_aiger {path}; print_stats; rewrite; print_stats', log, env)
    if len(measured) != 2:
        raise SystemExit(f'{name}: export run did not print two stat blocks')
    info = dict(circuit=name, before=measured[0], after=measured[1],
                untouched=measured[0] == measured[1], returncode=proc.returncode)
    print(f'{name}: export untouched={info["untouched"]}', flush=True)
    (folder / 'export_check.json').write_text(json.dumps(info, indent=2) + '\n')
    return folder


def compress_if_needed(folder):
    target = folder / 'cuts.jsonl'
    size = target.stat().st_size
    if size <= GZIP_THRESHOLD:
        return dict(path=str(target), bytes=size, sha256=file_digest(target), compressed=False)
    archived = folder / 'cuts.jsonl.gz'
    with target.open('rb') as source, gzip.open(archived, 'wb', compresslevel=6) as sink:
        shutil.copyfileobj(source, sink)
    sha = file_digest(target)
    target.unlink()
    return dict(path=str(archived), bytes=archived.stat().st_size, sha256=sha,
                original_bytes=size, compressed=True,
                note='解压方式: gzip -dc cuts.jsonl.gz > cuts.jsonl')


def depths_from_edges(n_leaves, n_size, root_node, edges):
    dist = [-1] * n_size
    if 0 <= root_node < n_size:
        dist[root_node] = 0
    for k in range(n_size - 1, n_leaves - 1, -1):
        if dist[k] < 0:
            continue
        for child in (edges[k - n_leaves][0], edges[k - n_leaves][2]):
            if dist[child] < dist[k] + 1:
                dist[child] = dist[k] + 1
    return dist[:n_leaves]


def leaf_depths_and_outside(aig, root, leaves, levels):
    """Mirror of original_leaf_depths that also returns the worst outside node."""
    boundary = set(leaves)
    lowest = min(leaves)
    interior = set()
    pending = [root]
    while pending:
        node = pending.pop()
        if node in boundary or node < lowest or node <= aig.pi_count or node in interior:
            continue
        interior.add(node)
        pending.extend(lit >> 1 for lit in aig.fanin_literals[node])
    distances = {root: 0}
    for node in sorted(interior, reverse=True):
        depth = distances.pop(node) + 1
        for literal in aig.fanin_literals[node]:
            pred = literal >> 1
            distances[pred] = max(distances.get(pred, -1), depth)
    depths = tuple(distances.get(leaf) for leaf in leaves)
    outside = [(node, d) for node, d in distances.items() if node not in boundary]
    if outside:
        node_out, dist_out = max(outside, key=lambda pair: (levels[pair[0]] + pair[1], -pair[0]))
    else:
        node_out, dist_out = None, None
    return depths, node_out, dist_out


def check_circuit(folder, aig_path, offset, circuit):
    """Acceptance 3 to 6, streaming over the export."""
    aig = parse_aig(aig_path)
    levels = propagate_levels(aig)
    fanins = aig.fanin_literals

    def to_aig(node_id):
        """ABC ids shift ANDs past the POs; PIs and the constant keep their id."""
        return node_id if node_id <= aig.pi_count else node_id - offset

    def valid_cut(root, leaves):
        boundary = set(leaves)
        if root in boundary:
            return False
        seen, pending = set(), [root]
        while pending:
            node = pending.pop()
            if node in boundary or node in seen:
                continue
            seen.add(node)
            if node <= aig.pi_count:
                return False
            preds = fanins.get(node)
            if not preds:
                return False
            pending.extend(lit >> 1 for lit in preds)
        return True

    best = defaultdict(lambda: -1)
    totals = dict(cuts=0, candidates=0, candidates_accepted=0, candidates_self=0)
    invalid = []
    depth_checked = 0
    unrebuildable = []
    unrebuildable_total = 0
    unrebuildable_roots = set()
    rng = random.Random(20261123)
    reservoir = []
    seen_candidates = 0

    with (folder / 'cuts.jsonl').open() as stream:
        for line in stream:
            cut = json.loads(line)
            totals['cuts'] += 1
            node = cut['node']
            leaves = cut['leaves']
            leaf_ids = [to_aig(leaf['id']) for leaf in leaves]
            root = to_aig(node)
            usable = (len(leaf_ids) == 4 and len(set(leaf_ids)) == 4
                      and root in fanins and valid_cut(root, leaf_ids))
            if not usable and len(invalid) < EXAMPLE_CAP:
                invalid.append(dict(node=node, leaves=leaf_ids))
            for cand in cut['candidates']:
                totals['candidates'] += 1
                if cand['self']:
                    totals['candidates_self'] += 1
                elif cand['accepted_by_default']:
                    totals['candidates_accepted'] += 1
                    gain = cut['n_nodes_saved'] - cand['nodes_added']
                    if gain > best[node]:
                        best[node] = gain
                if cand['n_size'] > 0:
                    depths = depths_from_edges(cand['n_leaves'], cand['n_size'],
                                               cand['root_node'], cand['edges'])
                    if list(depths) != cand['leaf_depths']:
                        raise SystemExit(
                            f'{circuit}: leaf_depths mismatch at node {node} subgraph '
                            f'{cand["subgraph"]}: recorded {cand["leaf_depths"]} '
                            f'recomputed {list(depths)}')
                    value = max((leaf['level'] + d for leaf, d in zip(leaves, depths) if d >= 0),
                                default=0)
                    if value != cand['root_level']:
                        raise SystemExit(
                            f'{circuit}: root_level mismatch at node {node} subgraph '
                            f'{cand["subgraph"]}: recorded {cand["root_level"]}, array-order '
                            f'recomputation {value}, leaf levels '
                            f'{[leaf["level"] for leaf in leaves]}, depths {list(depths)}')
                    depth_checked += 1
                seen_candidates += 1
                entry = dict(node=node, subgraph=cand['subgraph'],
                             leaf_depths=cand['leaf_depths'], root_level=cand['root_level'])
                if len(reservoir) < 20:
                    reservoir.append(entry)
                else:
                    slot = rng.randrange(seen_candidates)
                    if slot < 20:
                        reservoir[slot] = entry
            if usable:
                depths, out_node, out_dist = leaf_depths_and_outside(aig, root, leaf_ids, levels.level)
                lib_depths, lib_outside = original_leaf_depths(aig, root, leaf_ids, levels.level)
                if tuple(depths) != tuple(lib_depths):
                    raise SystemExit(f'{circuit}: cone mirror disagrees with level_weights at {node}')
                if lib_outside is not None:
                    if out_node is None or levels[out_node] + out_dist != lib_outside:
                        raise SystemExit(f'{circuit}: outside arrival disagrees with level_weights at {node}')
                rebuilt = max((levels.level[leaf] + d for leaf, d in zip(leaf_ids, depths)
                               if d is not None), default=0)
                if rebuilt != levels.level[root]:
                    unrebuildable_total += 1
                    unrebuildable_roots.add(node)
                    if len(unrebuildable) < EXAMPLE_CAP:
                        unrebuildable.append(dict(
                            node=node, true_level=levels.level[root], rebuilt=rebuilt,
                            gap=levels.level[root] - rebuilt,
                            outside_node=out_node,
                            outside_level=None if out_node is None else levels.level[out_node],
                            outside_distance=out_dist,
                            outside_arrival=None if out_node is None else levels[out_node] + out_dist,
                            leaves=leaf_ids, depths=list(depths)))

    nodes = [json.loads(line) for line in (folder / 'nodes.jsonl').open()]
    mismatched_gain = [dict(node=row['node'], recorded=row['gain_best'], recomputed=best[row['node']])
                       for row in nodes
                       if row['reason'] == 'evaluated' and row['gain_best'] != best[row['node']]]

    counters = json.loads((folder / 'counters.json').read_text())
    counters['derived'] = dict(
        cuts_examined_nontrivial=(counters['cuts_exported'] + counters['cuts_lt4_leaves']
                                  + counters['cuts_single_fanout_gt2'] + counters['cuts_null_leaf']),
        note=('counters.json 的 cuts_total 只在进入导出路径时递增，因此等于 cuts_exported；'
              '非平凡 cut 总数由上式计算；平凡 cut（节点自身）每个被评估节点各跳过一次。'))

    return dict(
        circuit=circuit, offset=offset,
        counters=counters,
        coverage=dict(
            nodes_exported=len(nodes),
            nodes_with_cuts=sum(1 for row in nodes
                                if row['reason'] == 'evaluated' and row['cuts_exported'] > 0),
            **totals),
        check3_gain_mismatches=len(mismatched_gain), check3_examples=mismatched_gain[:5],
        check4_invalid_cuts=len(invalid), check4_examples=invalid[:5],
        check5_asserted_candidates=depth_checked,
        check5_depth_mismatches=0,
        check5_root_level_mismatches=0,
        check5_sample=reservoir,
        check6_unrebuildable_cuts=unrebuildable_total,
        check6_unrebuildable_roots=sorted(unrebuildable_roots),
        check6_examples=unrebuildable,
        check6_truncated=unrebuildable_total > len(unrebuildable),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--build', required=True, type=Path, help='build run holding both ABC variants')
    parser.add_argument('--frozen', required=True, type=Path)
    parser.add_argument('--circuits', default=','.join(name for name, _ in CIRCUITS))
    args = parser.parse_args()
    run = args.run.resolve()
    out = run / 'output'
    (out / 'logs').mkdir(parents=True, exist_ok=True)
    if not os.getenv('SLURM_JOB_ID') or not run.is_relative_to('/data/wushangge'):
        raise SystemExit('run this only inside the committed /data Slurm job')

    wanted = [name.strip() for name in args.circuits.split(',') if name.strip()]
    known = {name for name, _ in CIRCUITS}
    if set(wanted) - known:
        raise SystemExit(f'unknown circuit(s): {sorted(set(wanted) - known)}')

    abc_orig = args.build.resolve() / 'build/abc_orig/abc'
    abc_dump = args.build.resolve() / 'build/abc_dump/abc'
    for path in (abc_orig, abc_dump, args.frozen):
        if not Path(path).is_file():
            raise SystemExit(f'missing binary: {path}')

    circuits = stage_inputs(run, wanted)
    regression = check_regression(run, abc_orig, abc_dump, args.frozen, circuits)

    results, exports = {}, {}
    for name in wanted:
        folder = export_candidates(run, abc_dump, circuits[name], name)
        mapping = identifier_mapping(folder / 'nodes.tsv', parse_aig(circuits[name]), name)
        results[name] = check_circuit(folder, circuits[name], mapping['offset'], name)
        results[name]['identifier_mapping'] = mapping
        exports[name] = compress_if_needed(folder)
        print(f"{name}: export {exports[name]['bytes']} bytes, cut6 "
              f"{results[name]['check6_unrebuildable_cuts']}", flush=True)
        (out / f'{name}.checks.json').write_text(json.dumps(results[name], indent=2) + '\n')

    summary = dict(regression=regression, circuits=results, exports=exports,
                   binaries=dict(orig_sha256=file_digest(abc_orig),
                                 dump_sha256=file_digest(abc_dump),
                                 frozen_sha256=file_digest(args.frozen)))
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')

    for name in wanted:
        row = results[name]
        print(f"RESULT {name}: check3={row['check3_gain_mismatches']} "
              f"check4={row['check4_invalid_cuts']} check5_depth={row['check5_depth_mismatches']} "
              f"check6={row['check6_unrebuildable_cuts']}/{row['coverage']['cuts']}", flush=True)


if __name__ == '__main__':
    sys.exit(main())
