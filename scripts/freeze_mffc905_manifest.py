#!/usr/bin/env python3
"""Freeze all905 identities plus existing structural metadata; no EDA/models."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    if not os.environ.get('SLURM_JOB_ID') or not run.is_relative_to('/data/wushangge'):
        parser.error('All-input staging/hash/metadata inspection requires a /data Slurm run.')
    def sha(path):
        value = hashlib.sha256()
        with path.open('rb') as stream:
            for data in iter(lambda: stream.read(1024 * 1024), b''):
                value.update(data)
        return value.hexdigest()
    expected_inventory = 'e4211308a06fc5b2bb02614b965c4b93dcf08f6f7ea2a73af2f2e1405da1e167'
    if sha(args.inventory) != expected_inventory:
        raise ValueError('Do not substitute a selected inventory for the complete fixed905.')
    upstream = json.loads(args.inventory.read_text())
    rows = upstream['rows']
    if len(rows) != 905 or len({r['index'] for r in rows}) != 905 or len({r['sha256'] for r in rows}) != 905:
        raise ValueError('Complete905 input identity/uniqueness failure.')
    historical = [Path('/data/wushangge/CULS_LiteCut/runs') / name for name in [
        '20261111T184752Z-consistent-full-shard0-1568c1b9', '20261111T184753Z-consistent-full-shard1-175169c2']]
    result = []
    for row in rows:
        staged = Path(row['staged_path'])
        if not staged.resolve().is_relative_to('/data/wushangge') or sha(staged) != row['sha256']:
            raise ValueError('Frozen binary AIG missing or changed: ' + str(staged))
        with staged.open('rb') as stream:
            header = stream.readline().split()
        if (len(header) != 6 or header[0] != b'aig' or int(header[3]) != 0 or
                [int(header[i]) for i in (2, 4, 5)] != [row['pi_count'], row['po_count'], row['and_count']]):
            raise ValueError('AIG interface/count metadata mismatch.')
        case = historical[row['index'] % 2] / 'output/suite' / f'case_{row["index"]:04d}'
        cfg = json.loads((case / 'config.json').read_text())
        if cfg['input_sha256'] != row['sha256']:
            raise ValueError('Historical structural metadata belongs to another input.')
        partition_path, catalogue_path = case / 'partition/partition.json', case / 'cut_validation/verified_catalogue.json'
        partition = json.loads(partition_path.read_text())
        regions = len(partition['regions'])
        parents = [[] for _ in range(regions)]
        for a, b in partition['edges']:
            parents[b].append(a)
        levels = {}
        for node in partition['topological_order']:
            levels[node] = 1 + max((levels[parent] for parent in parents[node]), default=0)
        if len(levels) != regions:
            raise ValueError('Historical Region DAG metadata is incomplete.')
        catalogue = json.loads(catalogue_path.read_text())
        reference = {'regions': regions, 'region_edges': len(partition['edges']),
                     'region_dag_layers': max(levels.values(), default=0),
                     'max_layer_width': max(Counter(levels.values()).values(), default=0),
                     'decisions': len(catalogue['decisions']),
                     'candidates': sum(d['candidate_count'] for d in catalogue['decisions']),
                     'partition_path': str(partition_path), 'partition_sha256': sha(partition_path),
                     'verified_catalogue_path': str(catalogue_path), 'verified_catalogue_sha256': sha(catalogue_path),
                     'scope': 'Historical positive-gain domain/strict structure metadata only; new methods still pay for all preparation, infer again and report actual verified counts.'}
        result.append({**row, 'historical_structure_metadata': reference})
    payload = {'schema': 'complete905_mffc_gpu_inputs_v1', 'upstream_inventory': str(args.inventory),
               'upstream_inventory_sha256': expected_inventory, 'rows': result,
               'source_counts': dict(Counter(r['relative_path'].split('/')[0] for r in result)),
               'scope': 'Immutable complete existing905 identity list, every staged AIG hash and header verified. Structural metadata is descriptive and never a free optimization/model/timing cache.',
               'prepared_by_commit': json.loads((run / 'run.json').read_text())['source_commit'],
               'slurm_job_id': os.environ['SLURM_JOB_ID']}
    path = run / 'output/benchmark905_manifest.json'
    with path.open('x') as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
    with (run / 'output/manifest_validation.json').open('x') as stream:
        json.dump({'status': 'PASS', 'inputs': len(result), 'all_hashes_and_headers_verified': True,
                   'manifest_sha256': sha(path), 'source_counts': payload['source_counts']}, stream, indent=2)
    print(json.dumps({'status': 'PASS', 'inputs': len(result), 'manifest': str(path), 'sha256': sha(path)}))


if __name__ == '__main__':
    main()
