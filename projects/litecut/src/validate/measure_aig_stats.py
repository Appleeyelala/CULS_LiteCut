"""Measure AIG statistics (Nodes, Levels) for the 14 selected cases using ABC 'read; ps'."""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
from datetime import datetime


def _now() -> str:
    return datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect AIG node/level stats via ABC ps command")
    p.add_argument("--picks_csv", required=True, help="main_table_picks.csv")
    p.add_argument("--aig_dir", default="./validate_cases", help="Root dir of AIG files")
    p.add_argument("--abc_bin", default="./abc/abc")
    p.add_argument("--output_csv", required=True, help="Output CSV path")
    return p.parse_args()


def run_abc_ps(abc_bin: str, aig_path: str) -> dict[str, int | None]:
    """Run ABC 'read <aig>; ps' and parse i/o, and-nodes, levels."""
    cmd = f'read {aig_path}; ps'
    proc = subprocess.run(
        ["script", "-qc", f'{abc_bin} -c "{cmd}"', "/dev/null"],
        capture_output=True, text=True, timeout=120,
    )
    output = re.sub(r'\x1b\[[0-9;]*m', '', proc.stdout)

    result: dict[str, int | None] = {
        "pi": None, "po": None, "lat": None, "and_nodes": None, "levels": None,
    }

    m = re.search(r'i/o\s*=\s*(\d+)\s*/\s*(\d+)', output)
    if m:
        result["pi"] = int(m.group(1))
        result["po"] = int(m.group(2))

    m = re.search(r'lat\s*=\s*(\d+)', output)
    if m:
        result["lat"] = int(m.group(1))

    m = re.search(r'and\s*=\s*(\d+)', output)
    if m:
        result["and_nodes"] = int(m.group(1))

    m = re.search(r'lev\s*=\s*(\d+)', output)
    if m:
        result["levels"] = int(m.group(1))

    return result


def main() -> int:
    print(f"CurPID: {os.getpid()}")
    args = parse_args()

    with open(args.picks_csv, encoding="utf-8") as f:
        picks = list(csv.DictReader(f))
    print(f"{_now()} Loaded {len(picks)} cases from {args.picks_csv}")

    results = []
    for row in picks:
        ds, case = row["dataset"], row["case"]
        if ds == "OpenABC":
            aig_file = os.path.join(args.aig_dir, ds, f"{case}_orig.aig")
        else:
            aig_file = os.path.join(args.aig_dir, ds, f"{case}.aig")

        if not os.path.isfile(aig_file):
            print(f"{_now()} SKIP {ds}/{case}: {aig_file} not found")
            continue

        print(f"{_now()} {ds}/{case}")
        stats = run_abc_ps(args.abc_bin, aig_file)
        print(f"{_now()}   PI={stats['pi']} PO={stats['po']} "
              f"AND={stats['and_nodes']} LEV={stats['levels']}")

        results.append({
            "dataset": ds,
            "case": case,
            "pi": stats["pi"],
            "po": stats["po"],
            "and_nodes": stats["and_nodes"],
            "levels": stats["levels"],
        })

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "case", "pi", "po", "and_nodes", "levels"])
        w.writeheader()
        w.writerows(results)

    print(f"{_now()} Saved {len(results)} rows to {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
