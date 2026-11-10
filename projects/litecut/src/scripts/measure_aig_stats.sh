#!/bin/bash
set -euo pipefail

PYTHONPATH=src python3 -u src/validate/measure_aig_stats.py \
    --picks_csv ./docs/main_table_picks.csv \
    --aig_dir ./validate_cases \
    --abc_bin ./abc/abc \
    --output_csv ./log/aig_stats_picks.csv
