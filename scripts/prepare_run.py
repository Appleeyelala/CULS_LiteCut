#!/usr/bin/env python3
"""Prepare a committed source snapshot under /data; never run or submit a job."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import uuid


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args]).decode().strip()


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def select_design(config, requested=None):
    root = Path(config['dataset_root']).resolve(strict=True)
    relative = Path(requested if requested is not None else config['default_design'])
    if relative.is_absolute():
        raise ValueError('--design must be relative to the configured dataset root.')
    source = (root / relative).resolve(strict=True)
    if not source.is_relative_to(root) or not source.is_file():
        raise ValueError('Design must be a file inside the configured dataset root.')
    if source.suffix.lower() not in ('.aig', '.aag'):
        raise ValueError('Design must be an AIG or AAG file.')
    return root, source, digest(source)


def stage_design(source, destination, expected_sha256):
    with source.open('rb') as incoming, destination.open('xb') as outgoing:
        shutil.copyfileobj(incoming, outgoing)
    if digest(destination) != expected_sha256:
        raise ValueError('Dataset input changed while preparing its run snapshot.')


def extract_source(archive, destination):
    with tarfile.open(archive, 'r:') as stream:
        for member in stream.getmembers():
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError('Source archive escapes run directory: ' + member.name)
            if not (member.isfile() or member.isdir() or member.issym()):
                raise ValueError('Unsupported source archive member: ' + member.name)
            if member.issym():
                link = (target.parent / member.linkname).resolve()
                if not link.is_relative_to(destination.resolve()):
                    raise ValueError('Source symlink escapes run directory: ' + member.name)
        stream.extractall(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True, help='short lowercase name for the authorized task')
    parser.add_argument('--purpose', required=True, help='concrete purpose of the authorized task')
    parser.add_argument('--design', help='dataset-relative AIG/AAG path; defaults to config default_design')
    args = parser.parse_args()
    if not re.fullmatch(r'[a-z][a-z0-9_-]{0,47}', args.name):
        parser.error('--name must start with a-z and contain at most 48 lowercase letters, digits, - or _')
    if not args.purpose.strip():
        parser.error('--purpose must not be empty')
    repo = Path(__file__).resolve().parents[1]
    config = json.loads((repo / 'config/workspace.json').read_text())
    data = Path(config['data_root']).resolve()
    if not data.is_relative_to(Path('/data/wushangge')) or data == Path('/data/wushangge'):
        raise SystemExit('The project data root must be a dedicated directory below /data/wushangge.')
    if not data.is_dir():
        raise SystemExit('Project data root does not exist: ' + str(data))
    if git(repo, 'status', '--porcelain', '--untracked-files=normal'):
        raise SystemExit('Commit source/configuration changes before preparing an immutable run.')
    try:
        dataset_root, design_source, design_sha256 = select_design(config, args.design)
    except (OSError, ValueError) as exc:
        raise SystemExit('Invalid dataset input: ' + str(exc)) from exc
    commit = git(repo, 'rev-parse', 'HEAD')
    # Keep the historical asset registry intact; new runs take their design from the dataset.
    assets = [asset for asset in json.loads((repo / 'provenance/assets.json').read_text())
              if asset['name'] != 'c1355_orig']
    for asset in assets:
        path = Path(asset['path']).resolve()
        if not path.is_relative_to(data) or not path.is_file():
            raise SystemExit('Missing or external asset: ' + asset['path'])
        if digest(path) != asset['sha256']:
            raise SystemExit('Frozen asset checksum changed: ' + asset['path'])
    runs = data / config['runs_dir']
    if not runs.resolve().is_relative_to(data):
        raise SystemExit('Run directory escapes project data root.')
    runs.mkdir(exist_ok=True)
    now = datetime.now(timezone.utc)
    name = now.strftime('%Y%m%dT%H%M%SZ') + '-' + args.name + '-' + uuid.uuid4().hex[:8]
    run = runs / name
    run.mkdir(exist_ok=False)
    staged_design = run / 'input' / ('design' + design_source.suffix.lower())
    metadata = {'name': name, 'purpose': args.purpose, 'created_at': now.isoformat(),
                'source_commit': commit, 'source_repository': str(repo), 'source_snapshot': str(run / 'src'),
                'state': 'PREPARING', 'slurm_job_id': None, 'command': None,
                'resources': None, 'environment': None, 'assets': assets,
                'dataset': {'root': str(dataset_root), 'source': str(design_source),
                            'relative_path': str(design_source.relative_to(dataset_root)),
                            'sha256': design_sha256, 'staged_path': str(staged_design)},
                'resource_budget': config['resource_budget'], 'experiment_executed': False}
    manifest = run / 'run.json'

    def save():
        manifest.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')

    save()
    try:
        for sub in ['src', 'input', 'output', 'logs', 'tmp', 'cache', 'build']:
            (run / sub).mkdir()
        archive = run / 'tmp/source.tar'
        subprocess.run(['git', '-C', str(repo), 'archive', '--format=tar', '--output=' + str(archive), commit], check=True)
        extract_source(archive, run / 'src')
        archive.unlink()
        for asset in assets:
            (run / 'input' / asset['name']).symlink_to(Path(asset['path']))
        stage_design(design_source, staged_design, design_sha256)
        if design_source.relative_to(dataset_root) == Path('OpenABC/c1355_orig.aig'):
            (run / 'input' / 'c1355_orig').symlink_to(staged_design.name)
        metadata['state'] = 'PREPARED_NOT_SUBMITTED'
        save()
        (run / 'README.txt').write_text(
            'Prepared source snapshot only. No program or Slurm job has been run.\n'
            'Review the task command and adapt historical absolute paths explicitly.\n'
            'Inspect cluster configuration and the aggregate user resource budget before submission.\n'
            'Keep job.slurm, logs, build, cache, TMPDIR and outputs in this run directory.\n'
            'Use the dataset.staged_path recorded in run.json as the circuit input.\n'
            'Record command/environment/resources/job ID and completion status in run.json.\n')
    except Exception as exc:
        metadata['state'] = 'PREPARATION_FAILED'
        metadata['error'] = str(exc)
        save()
        raise
    print(run)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
