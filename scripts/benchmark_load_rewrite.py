"""Compare three selectors on the same raw AIG with fixed calls and fixed search time."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import time

def benchmark(args):
    rows = []
    args.output.mkdir(parents=True, exist_ok=False)
    for budget in ('calls', 'seconds'):
        for method in ('gradient', 'greedy', 'random'):
            directory = args.output / f'{budget}_{method}'
            command = [sys.executable, '-B', str(Path(__file__).with_name('load_rewrite.py')),
                       '--original', str(args.original), '--tools', str(args.tools),
                       '--output', str(directory), '--method', method, '--seed', str(args.seed),
                       '--steps', str(args.calls if budget == 'calls' else 100000)]
            if budget == 'seconds':
                command.extend(('--seconds', str(args.seconds)))
            start = time.monotonic()
            subprocess.run(command, check=True)
            wall = time.monotonic() - start
            result = json.loads((directory / 'result.json').read_text())
            proposal = json.loads((directory / 'proposal.json').read_text())
            fields = ('status', 'original_delay_ps', 'proposal_delay_ps', 'delay_ps',
                      'proposal_area_ratio', 'area_ratio', 'cec', 'candidate_cec',
                      'unique_vectors', 'search_seconds', 'mapped_evaluations',
                      'template_match_passes', 'cuda_peak_bytes', 'process_maxrss_kib')
            row = {key: result[key] for key in fields}
            row.update(budget=budget, method=method, hard_calls=proposal['hard_calls'],
                       edits=sum(bool(c) for c in proposal['vector']), wall_s=wall,
                       call_budget=args.calls if budget == 'calls' else '',
                       seconds_budget=args.seconds if budget == 'seconds' else '')
            rows.append(row)
    with (args.output / 'controls.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('original', 'tools', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--calls', type=int, default=8)
    parser.add_argument('--seconds', type=float, default=3.0)
    parser.add_argument('--seed', type=int, default=20261117)
    benchmark(parser.parse_args())
