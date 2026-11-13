#!/usr/bin/env python3
"""Freeze a bounded evaluation panel from input metadata, never QoR results."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


DEVELOPMENT_FAMILIES = {'c1355', 'ss_pcm', 'dma', 'div', 'hyp', 'voter',
                        'back_end_with_decode', 'division'}
BINS = [(128, 4096), (4096, 8192), (8192, 16384), (16384, 50001)]


def family(path):
    return re.sub(r'_(orig|comb)$', '', Path(path).stem.lower())


def select_panel(manifest, per_bin=4):
    used_families, used_hashes, source_counts = set(), set(), Counter()
    result = []
    for lower, upper in BINS:
        pool = [row for row in manifest['rows']
                if row['classification'] == 'COMBINATIONAL'
                and lower <= row['and_count'] < upper
                and family(row['relative_path']) not in DEVELOPMENT_FAMILIES]
        for _ in range(per_bin):
            remaining = [row for row in pool if family(row['relative_path']) not in used_families
                         and row['sha256'] not in used_hashes]
            if not remaining:
                raise ValueError(f'Insufficient distinct families in [{lower}, {upper}).')
            # Balance dataset source when available, then use only input SHA.
            row = min(remaining, key=lambda r: (source_counts[r['relative_path'].split('/')[0]], r['sha256']))
            source = row['relative_path'].split('/')[0]
            used_families.add(family(row['relative_path']))
            used_hashes.add(row['sha256'])
            source_counts[source] += 1
            result.append({k: row[k] for k in ['index', 'relative_path', 'sha256', 'pi_count',
                                             'po_count', 'and_count', 'staged_path']}
                          | {'family': family(row['relative_path']), 'and_bin': [lower, upper]})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = args.inventory.read_bytes()
    rows = select_panel(json.loads(data))
    record = {'schema': 'gpu_joint_panel_v1', 'inventory': str(args.inventory.resolve()),
              'inventory_sha256': hashlib.sha256(data).hexdigest(),
              'seeds': [20261111, 20261112, 20261113],
              'selection': 'Four input-scale bins, four designs per bin; least-used available dataset source then ascending input SHA256; unique normalized basename families and input hashes; no measured QoR, candidate availability or runtime used.',
              'development_families_excluded': sorted(DEVELOPMENT_FAMILIES),
              'scope': 'Frozen evaluation panel, not unseen data: the earlier 905 inputs were already analyzed. Scale cap bounds experiment cost; unrelated family aliases beyond basename normalization are not established.',
              'rows': rows}
    with args.output.open('x') as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    for row in rows:
        print(row['index'], row['and_count'], row['relative_path'], row['sha256'])


if __name__ == '__main__':
    main()
