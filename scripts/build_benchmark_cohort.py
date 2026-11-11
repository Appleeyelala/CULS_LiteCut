#!/usr/bin/env python3
"""Freeze explicitly ordered suites, preserving failed and pending attempts."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from global_timing.cohort import build_cohort, read_cohort


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--suites', type=Path, nargs='+', required=True,
                        help='Oldest priority first. Later terminal failures also supersede earlier successes.')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not args.output.resolve().is_relative_to('/data/wushangge'):
        parser.error('Use Slurm and a new /data output directory.')
    report = build_cohort(args.manifest, args.suites, args.output)
    read_cohort(args.output / 'cohort.json')
    print(json.dumps({key: report[key] for key in ['recorded_inputs', 'expected_inputs', 'coverage_complete',
                                                  'case_status_counts', 'attempt_status_counts',
                                                  'selected_source_versions']}), flush=True)


if __name__ == '__main__':
    main()
