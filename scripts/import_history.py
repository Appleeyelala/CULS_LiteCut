#!/usr/bin/env python3
"""One-time, non-destructive import of the existing local research workspace.

Copies files and records provenance only. Never executes research entry points.
Refuses to overwrite an existing import or data workspace.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from datetime import datetime, timezone

REPO = Path(__file__).resolve().parents[1]
DATA = Path('/data/wushangge/CULS_LiteCut')
CULS = Path('/home/wushangge/CULS')
LITECUT = Path('/home/wushangge/cutLibOrigin')
GATE1 = Path('/data/wushangge/gate1a_AIGRewrite')
ABC = Path('/home/wushangge/AIGRewrite/abc')
RUNS = Path('/data/wushangge/runs')
SKIP = {'.git', '__pycache__', '.pytest_cache', '.env'}
ROWS = []


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def git(root, *args):
    return subprocess.check_output(
        ['git', '-C', str(root), *args], env={**os.environ, 'GIT_OPTIONAL_LOCKS': '0'})


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def copy(source, dest, category):
    if dest.exists() or dest.is_symlink():
        raise FileExistsError(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    digest = sha(dest)
    if digest != sha(source):
        raise RuntimeError('Copy checksum mismatch: ' + str(source))
    storage = 'repository' if dest.is_relative_to(REPO) else 'data'
    ROWS.append({'source': str(source), 'destination': str(dest.relative_to(REPO if storage == 'repository' else DATA)),
                 'storage': storage, 'category': category, 'bytes': dest.stat().st_size, 'sha256': digest})


def tree(source, dest, category, predicate=lambda p: True):
    for path in sorted(source.rglob('*')):
        rel = path.relative_to(source)
        if set(rel.parts) & SKIP or path.suffix in {'.pyc', '.o', '.d'}:
            continue
        if path.is_file() and predicate(path):
            copy(path, dest / rel, category)


def main():
    if DATA.exists() or (REPO / 'provenance/import_manifest.jsonl').exists():
        raise SystemExit('Existing data/import found; refusing to overwrite it.')
    for name in ['runs', 'assets/models', 'assets/inputs', 'assets/libraries', 'assets/bin',
                 'history/imports', 'history/links', 'cache', 'tmp', 'registries']:
        (DATA / name).mkdir(parents=True, exist_ok=True)

    snapshots = []
    for key, root in [('culs', CULS), ('litecut', LITECUT), ('gate1', GATE1), ('abc', ABC)]:
        status = git(root, 'status', '--porcelain=v1').decode()
        snapshots.append({'name': key, 'source': str(root), 'head': git(root, 'rev-parse', 'HEAD').decode().strip(),
                          'status': status, 'policy': 'current working-tree bytes, including selected untracked research files'})
        dest = REPO / 'provenance' / (key + '_tracked_changes.patch')
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(git(root, 'diff', 'HEAD', '--', 'src', 'include', 'CMakeLists.txt'))

    for origin, dest in [(CULS, REPO / 'projects/culs'), (GATE1, REPO / 'legacy/gate1_aigrewrite')]:
        for name in ['src', 'include']:
            tree(origin / name, dest / name, 'source')
        for name in ['CMakeLists.txt', 'LICENSE', 'README.md']:
            copy(origin / name, dest / name, 'source')
    tree(CULS / 'tools', REPO / 'projects/culs/tools', 'source')
    tree(CULS / 'experiments', REPO / 'projects/culs/experiments', 'source',
         lambda p: 'results' not in p.relative_to(CULS / 'experiments').parts)
    tree(LITECUT / 'src', REPO / 'projects/litecut/src', 'source')
    for p in sorted(LITECUT.iterdir()):
        if p.is_file() and (p.name.startswith('gate2') and p.suffix in {'.py', '.sh', '.slurm'}
                            or p.name in {'README.md', 'params.json', 'train_gnn.sh'}):
            copy(p, REPO / 'projects/litecut' / p.name, 'source')
    for p in sorted(GATE1.iterdir()):
        if p.is_file() and p.name.startswith('gate1') and p.suffix in {'.py', '.sh', '.slurm'}:
            copy(p, REPO / 'legacy/gate1_aigrewrite' / p.name, 'source')
    abc_paths = [Path(p.decode()) for p in git(ABC, 'ls-files', '-z').split(b'\0') if p]
    for rel in abc_paths:
        if (ABC / rel).is_file():
            copy(ABC / rel, REPO / 'third_party/abc' / rel, 'upstream_source')

    # Preserve exact changed ABC source files from each available historical variant.
    # These run copies have no .git; compare actual bytes to the recorded clean base.
    variants = []
    for run in sorted(RUNS.glob('CULS-*')):
        root = run / 'abc'
        if not (root / 'src/base/abc/abc.h').is_file():
            continue
        changed, missing = [], []
        for rel in abc_paths:
            p = root / rel
            if not p.is_file():
                missing.append(str(rel))
            elif sha(p) != sha(ABC / rel):
                copy(p, REPO / 'history/abc_variants' / run.name / rel, 'abc_variant_overlay')
                changed.append(str(rel))
        extras = []
        known = set(abc_paths)
        for p in sorted((root / 'src').rglob('*')):
            rel = p.relative_to(root)
            if p.is_file() and rel not in known and p.suffix in {'.c', '.h', '.cpp', '.cc', '.cxx', '.hpp'}:
                copy(p, REPO / 'history/abc_variants' / run.name / rel, 'abc_variant_overlay')
                extras.append(str(rel))
        variants.append({'run': str(run), 'source': str(root), 'overlay': 'history/abc_variants/' + run.name,
                         'changed_files': changed, 'extra_source_files': extras,
                         'missing_base_files': missing, 'base': 'third_party/abc'})

    archive_roots = []
    for p in sorted(CULS.glob('gate3*_out')):
        archive_roots.append(('culs/' + p.name, p, p.name.removesuffix('_out')))
    for p in sorted(GATE1.glob('gate1*_out')):
        archive_roots.append(('gate1/' + p.name, p, p.name.removesuffix('_out')))
    for name in ['gate2a_mapping_free_out', 'gate2a_mapping_free_repeat_final', 'gate2a_r_out', 'gate2b_out', 'gate2c_out']:
        archive_roots.append(('litecut/' + name, LITECUT / name, name.removesuffix('_out')))
    archive_roots.append(('culs/path_delay_model_results', CULS / 'experiments/path_delay_model/results', 'path_delay_model'))
    for rel, origin, label in archive_roots:
        tree(origin, DATA / 'history/imports' / rel, 'historical_artifact')
        # Small evidence copies are tracked; full AIGs, traces, binaries and logs live in /data.
        tree(origin, REPO / 'evidence' / label, 'evidence',
             lambda p: p.suffix in {'.md', '.json', '.csv', '.png'} and p.stat().st_size < 2 * 1024 * 1024)
    for p in sorted(LITECUT.glob('gate2*')):
        if p.is_file() and p.suffix in {'.md', '.json'}:
            copy(p, REPO / 'evidence/litecut_recovery' / p.name, 'evidence')

    assets = [
        ('student_checkpoint', LITECUT / 'best.pt', 'assets/models/litecut_student_best.pt'),
        ('c1355_orig', LITECUT / 'open-sourceDataset/OpenABC/c1355_orig.aig', 'assets/inputs/c1355_orig.aig'),
        ('asap7', RUNS / 'CULS-gate3f-20260922-gate2babc/input/asap7.lib', 'assets/libraries/asap7.lib'),
        ('abc_gate3f', RUNS / 'CULS-gate3f-20260922-gate2babc/bin/abc', 'assets/bin/abc_gate3f'),
        ('gpuls_gate3f', RUNS / 'CULS-gate3f-20260922-gate2babc/bin/gpuls', 'assets/bin/gpuls_gate3f'),
    ]
    asset_rows = []
    for key, source, rel in assets:
        copy(source, DATA / rel, 'asset')
        asset_rows.append({'name': key, 'source': str(source), 'path': str(DATA / rel), 'sha256': sha(DATA / rel)})
    # Preserve the large historical run directories in place, with one central index.
    links = []
    roots = sorted(RUNS.glob('CULS-*')) + [GATE1] + sorted(Path('/data/wushangge/gate2c_runs').glob('*'))
    for root in roots:
        if root.is_dir():
            link = DATA / 'history/links' / root.name
            link.symlink_to(root, target_is_directory=True)
            links.append({'name': root.name, 'path': str(link), 'source': str(root), 'kind': 'reference_not_independent_backup'})
    dump(REPO / 'provenance/source_repositories.json', snapshots)
    dump(REPO / 'provenance/abc_variants.json', variants)
    dump(REPO / 'provenance/assets.json', asset_rows)
    dump(REPO / 'provenance/historical_runs.json', links)
    manifest = REPO / 'provenance/import_manifest.jsonl'
    with manifest.open('x', encoding='utf-8') as stream:
        for row in ROWS:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + '\n')
    dump(REPO / 'provenance/import_summary.json', {
        'created_at': datetime.now(timezone.utc).isoformat(), 'data_root': str(DATA), 'source_root': str(REPO),
        'copied_files': len(ROWS), 'repository_files': sum(r['storage'] == 'repository' for r in ROWS),
        'data_files': sum(r['storage'] == 'data' for r in ROWS), 'historical_run_links': len(links),
        'abc_variants': len(variants), 'source_modified': False, 'experiment_executed': False,
        'excluded': ['.git internals', '.env', 'compiled objects', 'Python caches', 'full dataset', 'duplicate mffc_cover_bundle'],
        'dataset_source': str(LITECUT / 'open-sourceDataset'),
        'dataset_policy': 'Only the fixed c1355 input is copied; other inputs must be staged into /data before use.'})
    print(json.dumps({'copied_files': len(ROWS), 'abc_variants': len(variants), 'historical_links': len(links)}))


if __name__ == '__main__':
    main()
