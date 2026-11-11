"""CLI entrypoint; source preparation and Slurm submission are separate."""
import argparse
import os
from pathlib import Path

from .pipeline import Experiment, SearchConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ['input', 'gpuls', 'abc', 'library', 'checkpoint', 'output']:
        parser.add_argument('--' + option, type=Path, required=True)
    parser.add_argument('--mode', choices=['area_constraint', 'joint', 'both'], default='both')
    parser.add_argument('--seed', type=int, default=20261111)
    parser.add_argument('--evaluations', type=int, default=24)
    parser.add_argument('--warmup', type=int, default=6)
    parser.add_argument('--gradient-steps', type=int, default=60)
    parser.add_argument('--calibration-steps', type=int, default=80)
    parser.add_argument('--area-limit-ratio', type=float, default=1.05)
    parser.add_argument('--area-weight', type=float, default=0.5)
    parser.add_argument('--timeout', type=int, default=300)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not args.output.resolve().is_relative_to(Path('/data/wushangge')):
        parser.error('Experiments must run through Slurm with outputs under /data/wushangge.')
    config = SearchConfig(seed=args.seed, evaluation_budget=args.evaluations, warmup_evaluations=args.warmup,
                          gradient_steps=args.gradient_steps, calibration_steps=args.calibration_steps,
                          area_limit_ratio=args.area_limit_ratio, area_weight=args.area_weight, timeout_s=args.timeout,
                          modes=('area_constraint', 'joint') if args.mode == 'both' else (args.mode,))
    Experiment(args.input, args.gpuls, args.abc, args.library, args.checkpoint, args.output, config).run()


if __name__ == '__main__':
    main()
