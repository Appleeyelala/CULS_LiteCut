"""Measure wall-clock runtime of `rw` (ABC baseline) and `rw -s` (ours)
for selected cases listed in main_table_picks.csv."""

from __future__ import annotations

import argparse
import csv
import os
import shlex
import subprocess
import time
from datetime import datetime
from pathlib import Path


def _now() -> str:
    return datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")


def _abc_literal(path: Path) -> str:
    return '"' + str(path).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _run_abc_timed(abc_bin: str, command: str) -> tuple[float, str]:
    """Run an ABC command and return (wall_clock_seconds, stdout)."""
    shell_command = f"{shlex.quote(str(abc_bin))} -c {shlex.quote(command)}"
    t0 = time.perf_counter()
    completed = subprocess.run(
        ["script", "-q", "-c", shell_command, "/dev/null"],
        check=False,
        text=True,
        capture_output=True,
    )
    elapsed = time.perf_counter() - t0
    if completed.returncode != 0:
        raise RuntimeError(
            f"ABC failed (rc={completed.returncode}): {command}\n"
            f"stderr: {completed.stderr.strip()}"
        )
    return elapsed, completed.stdout


def _resolve_aig(dataset_path: Path, dataset: str, case: str) -> Path:
    suffix = "_orig.aig" if dataset == "OpenABC" else ".aig"
    return dataset_path / dataset / f"{case}{suffix}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Measure rw / rw -s runtime for selected cases")
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--abc_bin", required=True)
    p.add_argument("--lib_path", required=True)
    p.add_argument("--picks_csv", required=True, help="main_table_picks.csv")
    p.add_argument("--output_csv", required=True, help="Output CSV with runtime columns")
    p.add_argument("--repeats", type=int, default=3, help="Number of repeats for averaging")
    return p.parse_args()


def main() -> int:
    print(f"CurPID: {os.getpid()}")
    args = parse_args()
    dataset_path = Path(args.dataset_path)
    abc_bin = args.abc_bin
    lib_path = Path(args.lib_path)
    picks_csv = Path(args.picks_csv)
    output_csv = Path(args.output_csv)
    repeats = args.repeats

    # Read picks
    with open(picks_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        picks = list(reader)

    print(f"{_now()} Loaded {len(picks)} cases from {picks_csv}")
    print(f"{_now()} Repeats per measurement: {repeats}")

    results = []
    for i, row in enumerate(picks):
        dataset = row["dataset"]
        case = row["case"]
        aig = _resolve_aig(dataset_path, dataset, case)

        if not aig.is_file():
            print(f"{_now()} SKIP {dataset}/{case}: AIG not found at {aig}")
            continue

        print(f"{_now()} [{i+1}/{len(picks)}] {dataset}/{case}")

        # Measure `rw` (ABC baseline)
        rw_cmd = f"read {_abc_literal(lib_path)}; read {_abc_literal(aig)}; rw"
        rw_times = []
        for r in range(repeats):
            elapsed, _ = _run_abc_timed(abc_bin, rw_cmd)
            rw_times.append(elapsed)
            print(f"{_now()}   rw  run {r+1}: {elapsed:.4f}s")

        # Measure `rw -s` (ours)
        rws_cmd = f"read {_abc_literal(aig)}; rw -sq"
        rws_times = []
        for r in range(repeats):
            elapsed, _ = _run_abc_timed(abc_bin, rws_cmd)
            rws_times.append(elapsed)
            print(f"{_now()}   rw -s run {r+1}: {elapsed:.4f}s")

        rw_avg = sum(rw_times) / len(rw_times)
        rws_avg = sum(rws_times) / len(rws_times)

        print(f"{_now()}   rw avg={rw_avg:.4f}s, rw -s avg={rws_avg:.4f}s")

        results.append({
            "dataset": dataset,
            "case": case,
            "rw_runtime_s": f"{rw_avg:.4f}",
            "rws_runtime_s": f"{rws_avg:.4f}",
            "rw_times": ";".join(f"{t:.4f}" for t in rw_times),
            "rws_times": ";".join(f"{t:.4f}" for t in rws_times),
        })

    # Write output
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "case", "rw_runtime_s", "rws_runtime_s", "rw_times", "rws_times"])
        writer.writeheader()
        writer.writerows(results)

    print(f"{_now()} Saved {len(results)} rows to {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
