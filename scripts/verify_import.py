#!/usr/bin/env python3
"""Read-only verification of imported files and central historical links."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-originals', action='store_true', help='also hash original source paths')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    config = json.loads((repo / 'config/workspace.json').read_text())
    data = Path(config['data_root'])
    failures, checked = [], 0
    for line in (repo / 'provenance/import_manifest.jsonl').read_text().splitlines():
        row = json.loads(line)
        base = repo if row['storage'] == 'repository' else data
        targets = [base / row['destination']]
        if args.check_originals:
            targets.append(Path(row['source']))
        for path in targets:
            checked += 1
            if not path.is_file():
                failures.append({'path': str(path), 'reason': 'missing'})
            elif path.stat().st_size != row['bytes'] or digest(path) != row['sha256']:
                failures.append({'path': str(path), 'reason': 'content differs from import baseline'})
    links = json.loads((repo / 'provenance/historical_runs.json').read_text())
    for row in links:
        path = Path(row['path'])
        if not path.is_symlink() or not path.is_dir() or path.resolve() != Path(row['source']).resolve():
            failures.append({'path': str(path), 'reason': 'historical reference missing or redirected'})
    print(json.dumps({'checked_file_copies': checked, 'checked_historical_links': len(links),
                      'failures': failures, 'status': 'PASS' if not failures else 'FAIL'},
                     ensure_ascii=False, indent=2))
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
