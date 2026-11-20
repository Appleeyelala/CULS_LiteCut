"""Read the primary run for each circuit from the frozen publication indices."""
import csv
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / 'analysis'
RESULTS = ANALYSIS / 'paper_joint_evidence'


def read_json(path):
    return json.loads(path.read_text())


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def new_output(path):
    path = path.resolve()
    if not os.environ.get('SLURM_JOB_ID') or not path.is_relative_to('/data/wushangge'):
        raise ValueError('Diagnostics require a new /data output in a Slurm job')
    path.mkdir(parents=True, exist_ok=False)
    return path


def primary_runs():
    manifests = read_csv(ANALYSIS / 'paper14_fullflow/circuits.csv')
    hashes = {row['circuit']: row['sha256'] for row in manifests}
    development = read_csv(ANALYSIS / 'paper14_fullflow/results.csv')
    noop = {r['circuit']: r['noop_ps'] for r in read_csv(RESULTS / 'development.csv')}
    external = read_csv(RESULTS / 'validation.csv')
    if len(development) != 14 or len(external) != 24:
        raise ValueError('Unexpected cohort size')
    for r in development:
        name = r['circuit']
        yield ('development', name, name, Path(r['work']), hashes[name],
               float(r['original_ps']), float(noop[name]))
    for r in external:
        yield ('external', Path(r['source']).stem, r['family'], Path(r['work']) / 'raw',
               r['aig_sha256'], float(r['original_ps']), float(r['noop_ps']))
