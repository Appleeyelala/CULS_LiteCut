"""Label-blind family grouping; diagnostic selection is explicitly separate."""
from collections import defaultdict
import hashlib
from pathlib import PurePosixPath
import re


def family_name(relative_path):
    stem = PurePosixPath(relative_path).stem.lower()
    stem = re.sub(r'(?:[_-](?:orig|comb|optimized|opt))+$', '', stem)
    stem = re.sub(r'\d+', '#', stem)
    return re.sub(r'[_-]+', '_', stem)


def rank_key(seed, value):
    return hashlib.sha256(f'{seed}:{value}'.encode()).hexdigest()


def grouped_split(inventory, seed=20261117):
    """Same hash or normalized family cannot occur in different partitions."""
    rows = sorted(inventory, key=lambda r: r['index'])
    if len({r['index'] for r in rows}) != len(rows):
        raise ValueError('Duplicate circuit index.')
    parent = {r['index']: r['index'] for r in rows}
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for row in rows:
        for key in [('hash', row['sha256']), ('family', family_name(row['relative_path']))]:
            if key in seen:
                a, b = find(row['index']), find(seen[key])
                parent[max(a, b)] = min(a, b)
            seen[key] = row['index']
    groups = defaultdict(list)
    for row in rows:
        groups[find(row['index'])].append(row)
    fractions = {'train': .6, 'validation': .2, 'test': .2}
    counts = {k: 0 for k in fractions}
    stratum_counts = defaultdict(lambda: {k: 0 for k in fractions})
    def stratum(row):
        n = row['and_count']
        return (row['relative_path'].split('/')[0], 0 if n < 1000 else 1 if n < 10000 else 2 if n < 100000 else 3)
    totals = defaultdict(int)
    for row in rows:
        totals[stratum(row)] += 1
    result = []
    ordered = sorted(groups.items(), key=lambda kv: (-len(kv[1]), rank_key(seed, kv[0])))
    for group, members in ordered:
        # Balance count and source/size strata without inspecting QoR or status.
        def cost(split):
            global_after = (counts[split] + len(members)) / (len(rows) * fractions[split])
            local = sum((stratum_counts[stratum(r)][split] + 1) /
                        (totals[stratum(r)] * fractions[split]) for r in members) / len(members)
            return global_after + local, list(fractions).index(split)
        split = min(fractions, key=cost)
        counts[split] += len(members)
        for r in members:
            stratum_counts[stratum(r)][split] += 1
            result.append({'circuit_id': r['index'], 'input': r['relative_path'],
                           'input_sha256': r['sha256'], 'and_count': r['and_count'],
                           'family': family_name(r['relative_path']), 'group_id': group,
                           'split': split, 'size_bin': stratum(r)[1]})
    return sorted(result, key=lambda r: r['circuit_id'])


def blind_panel(rows, split, count, seed=20261117):
    strata = defaultdict(list)
    for row in rows:
        if row['split'] == split:
            strata[(row['input'].split('/')[0], row['size_bin'])].append(row)
    for members in strata.values():
        members.sort(key=lambda r: rank_key(seed, r['input_sha256'] + ':' + str(r['circuit_id'])))
    selected = []
    while len(selected) < count and any(strata.values()):
        for key in sorted(strata):
            if strata[key] and len(selected) < count:
                selected.append(strata[key].pop(0)['circuit_id'])
    if len(selected) != count:
        raise ValueError(f'{split} contains too few circuits for {count} requests.')
    return selected


def compare_delay(candidate, original):
    tolerance = max(2e-5, 2e-6 * max(abs(candidate), abs(original)))
    return -1 if candidate < original - tolerance else 1 if candidate > original + tolerance else 0
