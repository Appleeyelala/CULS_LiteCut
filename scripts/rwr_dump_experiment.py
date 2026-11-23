"""Run the ABC rewrite-candidate export and its six acceptance checks.

Everything runs inside one Slurm job from a committed snapshot. The script does
not trust the export: it re-derives the identifier mapping, cut validity, leaf
depths, the default gain and the original-structure level from the parsed AIG
and from the raw decomposition edges carried by each candidate.

ABC assigns object IDs in the order PI, PO, AND, while AIGER numbers PIs and
then ANDs. On these inputs the AND numbering therefore differs by a constant
offset equal to the PO count. The offset is verified node by node here instead
of being assumed, and a mismatch aborts the run.
"""
import argparse
import csv
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


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_abc(abc, commands, log_path, env=None, timeout=7200):
    proc = subprocess.run([str(abc), '-c', commands], capture_output=True, text=True,
                          env=env, timeout=timeout)
    Path(log_path).write_text(proc.stdout + proc.stderr)
    return proc


def stats(abc, commands, log_path, env=None):
    proc = run_abc(abc, commands, log_path, env)
    found = STAT_RE.findall(proc.stdout + proc.stderr)
    return [dict(io=(int(a), int(b)), lat=int(c), ands=int(d), lev=int(e)) for a, b, c, d, e in found], proc


def stage_inputs(run):
    staged = []
    folder = run / 'input/circuits'
    folder.mkdir(parents=True, exist_ok=True)
    available = {digest(path): path for path in sorted(PAPER_INPUTS.glob('*.aig'))}
    for name, sha in CIRCUITS:
        if name == 'c1355':
            source = run / 'input/design.aig'
        else:
            source = available.get(sha)
            if source is None:
                raise SystemExit(f'{name}: no file under {PAPER_INPUTS} matches {sha}')
        target = folder / f'{name}.aig'
        shutil.copyfile(source, target)
        if digest(target) != sha:
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
        mapped = tuple(0 if lit >> 1 == 0 else lit for lit in pair)
        mapped = tuple(lit if lit >> 1 <= aig.pi_count else lit - 2 * offset for lit in mapped)
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
        record = dict(circuit=name, input_sha256=digest(path))
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
    folder.mkdir(parents=True, exist_ok=True)
    log = run / 'output/logs' / f'{name}.export.log'
    env = dict(os.environ, CULS_RWR_DUMP=str(folder))
    measured, proc = stats(abc_dump, f'read_aiger {path}; print_stats; rewrite; print_stats', log, env)
    if len(measured) != 2:
        raise SystemExit('export run did not print two stat blocks')
    info = dict(circuit=name, before=measured[0], after=measured[1],
                untouched=measured[0] == measured[1], returncode=proc.returncode)
    print(f'{name}: export untouched={info["untouched"]}', flush=True)
    (folder / 'export_check.json').write_text(json.dumps(info, indent=2) + '\n')
    return folder, info


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


def check_exports(folder, aig_path, offset):
    """Acceptance 3 to 6 on one exported circuit."""
    aig = parse_aig(aig_path)
    levels = propagate_levels(aig)
    fanins = aig.fanin_literals
    cuts = [json.loads(line) for line in (folder / 'cuts.jsonl').open()]
    nodes = [json.loads(line) for line in (folder / 'nodes.jsonl').open()]

    def to_aig(node_id):
        """ABC ids shift ANDs past the POs; PIs and the constant keep their id."""
        return node_id if node_id <= aig.pi_count else node_id - offset

    # Acceptance 3: the recorded best gain must equal the best accepted candidate.
    best = defaultdict(lambda: -1)
    for cut in cuts:
        for cand in cut['candidates']:
            if cand['self'] or not cand['accepted_by_default']:
                continue
            gain = cut['n_nodes_saved'] - cand['nodes_added']
            if gain > best[cut['node']]:
                best[cut['node']] = gain
    mismatched_gain = [dict(node=row['node'], recorded=row['gain_best'], recomputed=best[row['node']])
                       for row in nodes
                       if row['reason'] == 'evaluated' and row['gain_best'] != best[row['node']]]

    # Acceptance 4: every path from the root down to a PI must pass a cut leaf.
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

    invalid_cuts = []
    for cut in cuts:
        leaves = [to_aig(leaf['id']) for leaf in cut['leaves']]
        if len(leaves) != 4 or len(set(leaves)) != 4 or not valid_cut(to_aig(cut['node']), leaves):
            invalid_cuts.append(dict(node=cut['node'], leaves=leaves))

    # Acceptance 5: recompute leaf depths from the raw edges, no C traversal involved.
    rng = random.Random(20261123)
    pool = [cand for cut in cuts for cand in cut['candidates'] if cand['n_size'] > 0]
    sample = rng.sample(pool, min(20, len(pool)))
    depth_mismatched = []
    for cand in sample:
        recomputed = depths_from_edges(cand['n_leaves'], cand['n_size'], cand['root_node'], cand['edges'])
        if recomputed != cand['leaf_depths']:
            depth_mismatched.append(dict(subgraph=cand['subgraph'], recorded=cand['leaf_depths'],
                                         recomputed=recomputed))
    level_mismatched = []
    for cut in cuts:
        for cand in cut['candidates']:
            if cand['n_size'] <= 0 or cand['self']:
                continue
            recomputed = depths_from_edges(cand['n_leaves'], cand['n_size'], cand['root_node'], cand['edges'])
            value = max((leaf['level'] + d for leaf, d in zip(cut['leaves'], recomputed) if d >= 0),
                        default=0)
            if value != cand['root_level']:
                level_mismatched.append(dict(node=cut['node'], subgraph=cand['subgraph'],
                                             recorded=cand['root_level'], recomputed=value))

    # Acceptance 6: cuts whose leaf cone cannot rebuild the original root level.
    unrebuildable = []
    for cut in cuts:
        leaves = [to_aig(leaf['id']) for leaf in cut['leaves']]
        root = to_aig(cut['node'])
        if len(set(leaves)) != len(leaves) or root not in fanins:
            continue
        depths, outside = original_leaf_depths(aig, root, leaves, levels.level)
        rebuilt = max((levels.level[leaf] + d for leaf, d in zip(leaves, depths) if d is not None),
                      default=0)
        if rebuilt != levels.level[root]:
            unrebuildable.append(dict(node=cut['node'], true_level=levels.level[root],
                                      rebuilt=rebuilt, outside=outside,
                                      leaves=leaves, depths=list(depths)))

    with_cuts = sum(1 for row in nodes if row['reason'] == 'evaluated' and row['cuts_exported'] > 0)
    return dict(
        circuit=aig_path.stem, offset=offset,
        cuts=len(cuts), nodes_exported=len(nodes), nodes_with_cuts=with_cuts,
        candidates=sum(len(cut['candidates']) for cut in cuts),
        candidates_accepted=sum(1 for cut in cuts for cand in cut['candidates']
                                if cand['accepted_by_default']),
        candidates_self=sum(1 for cut in cuts for cand in cut['candidates'] if cand['self']),
        check3_gain_mismatches=len(mismatched_gain), check3_examples=mismatched_gain[:5],
        check4_invalid_cuts=len(invalid_cuts), check4_examples=invalid_cuts[:5],
        check5_sampled=len(sample), check5_depth_mismatches=len(depth_mismatched),
        check5_examples=depth_mismatched[:5],
        check5_root_level_mismatches=len(level_mismatched), check5_root_examples=level_mismatched[:5],
        check6_unrebuildable_cuts=len(unrebuildable), check6_examples=unrebuildable[:10],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--build', required=True, type=Path, help='build run holding both ABC variants')
    parser.add_argument('--frozen', required=True, type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    out = run / 'output'
    (out / 'logs').mkdir(parents=True, exist_ok=True)
    if not os.getenv('SLURM_JOB_ID') or not run.is_relative_to('/data/wushangge'):
        raise SystemExit('run this only inside the committed /data Slurm job')

    abc_orig = args.build.resolve() / 'build/abc_orig/abc'
    abc_dump = args.build.resolve() / 'build/abc_dump/abc'
    for path in (abc_orig, abc_dump, args.frozen):
        if not Path(path).is_file():
            raise SystemExit(f'missing binary: {path}')

    circuits = stage_inputs(run)
    regression = check_regression(run, abc_orig, abc_dump, args.frozen, circuits)

    folder, export_info = export_candidates(run, abc_dump, circuits['c1355'], 'c1355')
    mapping = identifier_mapping(folder / 'nodes.tsv', parse_aig(circuits['c1355']), 'c1355')
    checks = check_exports(folder, circuits['c1355'], mapping['offset'])

    result = dict(regression=regression, export=export_info, identifier_mapping=mapping, checks=checks,
                  binaries=dict(orig_sha256=digest(abc_orig), dump_sha256=digest(abc_dump),
                                frozen_sha256=digest(args.frozen)))
    (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(mapping=mapping, checks=checks), indent=2)[:5000])


if __name__ == '__main__':
    sys.exit(main())
