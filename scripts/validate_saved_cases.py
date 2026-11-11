#!/usr/bin/env python3
"""Recheck saved exports and no-op AIGs using CPU proof; no GPU rerun."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from global_timing.candidates import Catalogue
from global_timing.common import save_json
from global_timing.cut_validation import validate_catalogue
from global_timing.evaluation import Evaluator
from global_timing.structure import make_partition, parse_aig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--abc', type=Path, required=True)
    parser.add_argument('--cases', type=Path, nargs='+', required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not str(args.run.resolve()).startswith('/data/wushangge/'):
        raise SystemExit('Use a prepared Slurm /data run.')
    records = []
    for index, case in enumerate(args.cases):
        work = args.run / 'output' / f'saved_case_{index}'
        work.mkdir()
        config = json.loads((case / 'config.json').read_text())
        aig = Path(config['input'])
        raw = json.loads((case / 'candidates/catalogue.json').read_text())
        catalogue = Catalogue(raw['decisions'], [(c['left'], c['right']) for c in raw['conflicts']],
                              {int(k): v for k, v in raw['region_decisions'].items()})
        record = {'case': str(case), 'status': 'RUNNING'}
        try:
            _, audit = validate_catalogue(args.abc, make_partition(parse_aig(aig)), catalogue, work / 'cut_validation')
            record['cut_counts'] = {k: audit[k] for k in ['raw_executable_decisions', 'verified_decisions', 'excluded_decisions']}
            candidate = case / 'replay_noop/eval_0000/rewrite/raw/aigs/choice.aig'
            if candidate.is_file():
                record['replay_noop'] = Evaluator(args.abc, args.run / 'input/asap7', aig).evaluate(candidate, work / 'noop')
            record['status'] = 'PASS'
        except Exception as exc:
            record.update(status='FAIL', reason=f'{type(exc).__name__}: {exc}')
        save_json(work / 'summary.json', record)
        records.append(record)
        print(json.dumps({k: v for k, v in record.items() if k != 'replay_noop'}), flush=True)
    save_json(args.run / 'output/saved_case_summary.json', records)


if __name__ == '__main__':
    main()
