"""Run the frozen paper circuits independently through the raw production research entry."""
import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('cases', nargs='+')
    args = parser.parse_args()
    with (ROOT / 'analysis/paper14_fullflow/circuits.csv').open() as manifest:
        entries = {row['circuit']: row for row in csv.DictReader(manifest)}
    args.output.mkdir(parents=True)
    failures = []
    for name in args.cases:
        row = entries[name]
        source = args.input_dir / row['filename']
        if hashlib.sha256(source.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError(f'{name} input checksum mismatch')
        case = args.output / name
        cmd = [sys.executable, '-B', str(ROOT / 'scripts/load_rewrite.py'), '--original', str(source),
               '--output',str(case), '--tools', str(ROOT / 'config/load_rewrite_tools.json'),
               '--alpha','0.5','--mode','delay','--method','gradient','--device','cuda','--steps','8']
        print(name,'START',flush=True)
        start = time.monotonic()
        with (args.output / (name+'.log')).open('w') as log:
            process = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        elapsed = time.monotonic()-start
        (args.output / (name+'.attempt.json')).write_text(json.dumps(dict(command=cmd,exit_code=process.returncode,wall_s=elapsed),indent=2))
        if process.returncode:
            failures.append(name)
        print(name, process.returncode, round(elapsed,2), flush=True)
    if failures:
        raise SystemExit('Failed: '+', '.join(failures))

if __name__ == '__main__':
    main()
