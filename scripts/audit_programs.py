#!/usr/bin/env python3
"""Inventory every source/program and perform non-executing syntax checks.

Runtime checks remain separate: syntax PASS never means runtime PASS. Imported
entrypoints may launch training or overwrite historical files even on import.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess


SUFFIXES = {'.py', '.sh', '.slurm', '.c', '.cc', '.cpp', '.cxx', '.cu',
            '.h', '.hpp', '.cuh', '.tcl', '.pl', '.ipynb', '.sql'}
DATABASE = re.compile(r'\b(sqlite3|sqlalchemy|psycopg\w*|pymongo|mysql|lmdb|h5py)\b')


def audit(root: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    db_hits = []
    for path in sorted(root.rglob('*')):
        if not path.is_file() or any(p in {'.git', '__pycache__', 'open-sourceDataset'} for p in path.relative_to(root).parts):
            continue
        if path.suffix.lower() in {'.db', '.sqlite', '.sqlite3', '.sql'}:
            db_hits.append({'path': str(path.relative_to(root)), 'kind': 'database_file'})
        if path.suffix not in SUFFIXES and path.name not in {'Makefile', 'CMakeLists.txt'}:
            continue
        content = path.read_bytes()
        text = content.decode('utf-8', errors='replace')
        kind = path.suffix or path.name
        row = {'path': str(path.relative_to(root)), 'kind': kind,
               'sha256': hashlib.sha256(content).hexdigest(),
               'syntax': 'NOT_RUN', 'import': 'NOT_RUN', 'runtime': 'NOT_RUN',
               'runtime_reason': 'Awaiting module-specific tests; not covered by syntax checks.'}
        row['entrypoint'] = bool(re.search(r'__name__\s*==|\bmain\s*\(', text)) or kind in {'.sh', '.slurm', '.pl', '.tcl', '.ipynb'}
        if kind == '.py':
            try:
                tree = ast.parse(content, filename=str(path))
                compile(tree, str(path), 'exec')
                row['syntax'] = 'PASS'
                row['imports'] = sorted({n.module or '' for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} |
                                        {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names})
            except (SyntaxError, ValueError) as exc:
                row['syntax'] = 'FAIL'
                row['syntax_reason'] = str(exc)
        elif kind in {'.sh', '.slurm'}:
            result = subprocess.run(['bash', '-n', str(path)], capture_output=True, text=True)
            row['syntax'] = 'PASS' if result.returncode == 0 else 'FAIL'
            row['syntax_reason'] = result.stderr.strip()
        elif kind == '.ipynb':
            try:
                notebook = json.loads(text)
                row['syntax'] = 'PASS' if isinstance(notebook.get('cells'), list) else 'FAIL'
                row['syntax_scope'] = 'Notebook JSON only; IPython cells require separate execution.'
            except ValueError as exc:
                row['syntax'] = 'FAIL'
                row['syntax_reason'] = str(exc)
        else:
            row['syntax_reason'] = 'Requires project compiler/interpreter and dependencies; build tests must supply evidence.'
        hits = sorted(set(DATABASE.findall(text))) if kind in {'.py', '.sql'} else []
        if hits:
            db_hits.append({'path': row['path'], 'kind': 'database_api_reference', 'matches': hits})
        rows.append(row)
    summary = {'program_source_count': len(rows), 'entrypoint_count': sum(r['entrypoint'] for r in rows),
               'syntax_counts': dict(Counter(r['syntax'] for r in rows)),
               'runtime_counts': dict(Counter(r['runtime'] for r in rows)),
               'database_hits': db_hits,
               'scope': 'All tracked snapshot source/program files, including ABC and variant overlays; no implicit runtime PASS.'}
    (output / 'program_inventory.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n')
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.root, args.output), ensure_ascii=False))


if __name__ == '__main__':
    main()
