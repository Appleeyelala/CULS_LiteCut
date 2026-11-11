#!/usr/bin/env python3
"""Read-only diagnostics of exported cuts against original AIG structure."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from global_timing.structure import parse_aig
from global_timing.common import save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise SystemExit('Use Slurm for batch diagnostics.')
    report = []
    for case in args.cases:
        cfg = json.loads((case / 'config.json').read_text())
        aig = parse_aig(Path(cfg['input']))
        catalogue = json.loads((case / 'candidates/catalogue.json').read_text())
        counts = {'structural_verified': 0, 'functional_support': 0, 'structural_mismatch': 0}
        examples = []
        for decision in catalogue['decisions']:
            values = dict(zip(decision['cut_leaves'], [0xaaaa, 0xcccc, 0xf0f0, 0xff00]))
            values.setdefault(0, 0)
            pending, interior, external = [decision['root']], set(), set()
            while pending:
                node = pending.pop()
                if node in values or node in interior or node in external:
                    continue
                if node <= aig.pi_count:
                    external.add(node)
                    continue
                interior.add(node)
                pending.extend(literal >> 1 for literal in aig.fanin_literals[node])
            if external:
                counts['functional_support'] += 1
                continue
            for node in sorted(interior):
                left, right = aig.fanin_literals[node]
                values[node] = (values[left >> 1] ^ (65535 if left & 1 else 0)) & (values[right >> 1] ^ (65535 if right & 1 else 0))
            expected = int(decision['truth']) & 65535
            actual = values[decision['root']]
            if actual == expected:
                counts['structural_verified'] += 1
            else:
                counts['structural_mismatch'] += 1
                if len(examples) < 10:
                    examples.append({'decision': decision, 'actual_truth': actual, 'expected_truth': expected,
                        'complement_only': actual == expected ^ 65535,
                        'original_fanins': {node: aig.fanin_literals[node] for node in sorted(interior)}})
        report.append({'case': str(case), 'input': cfg['input'], 'counts': counts, 'examples': examples})
        print(json.dumps({k: v for k, v in report[-1].items() if k != 'examples'}), flush=True)
    save_json(args.output, report)


if __name__ == '__main__':
    main()
