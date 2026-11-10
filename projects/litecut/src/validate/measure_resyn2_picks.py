"""Run `resyn2; map; topo; stime` on cases listed in main_table_picks.csv and
collect Area/Delay (from stime) and total wall-clock runtime per case."""

from __future__ import annotations

import argparse
import csv
import os
import re
import shlex
import subprocess
import time
from datetime import datetime
from pathlib import Path


CSV_COLUMNS = [
    "dataset",
    "case",
    "area",
    "delay_ps",
    "runtime_s",
    "status",
    "error_message",
]

_STIME_METRICS_RE = re.compile(
    r"Area\s*=\s*([0-9]+(?:\.[0-9]+)?)"
    r".*?Delay\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*ps",
    re.IGNORECASE,
)
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")


def _now() -> str:
    return datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")


def _abc_literal(path: Path) -> str:
    return '"' + str(path).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _resolve_aig(dataset_path: Path, dataset: str, case: str) -> Path:
    suffix = "_orig.aig" if dataset == "OpenABC" else ".aig"
    return dataset_path / dataset / f"{case}{suffix}"


def _run_abc_timed(abc_bin: str, command: str, timeout_s: float) -> tuple[float, str]:
    shell_command = f"{shlex.quote(str(abc_bin))} -c {shlex.quote(command)}"
    t0 = time.perf_counter()
    completed = subprocess.run(
        ["script", "-q", "-c", shell_command, "/dev/null"],
        check=False,
        text=True,
        capture_output=True,
        timeout=timeout_s,
    )
    elapsed = time.perf_counter() - t0
    if completed.returncode != 0:
        raise RuntimeError(
            f"ABC failed (rc={completed.returncode}): {command}\n"
            f"stdout: {completed.stdout.strip()}\n"
            f"stderr: {completed.stderr.strip()}"
        )
    return elapsed, completed.stdout


def _parse_stime_metrics(output: str) -> tuple[float, float]:
    cleaned = _ANSI_ESCAPE_RE.sub("", output)
    last: tuple[float, float] | None = None
    for line in cleaned.splitlines():
        m = _STIME_METRICS_RE.search(line)
        if m:
            last = (float(m.group(1)), float(m.group(2)))
    if last is None:
        raise ValueError("Area/Delay summary not found in stime output")
    return last


def _write_csv(output_csv: Path, rows: list[dict]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run resyn2 on picks and collect Area/Delay/runtime")
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--abc_bin", required=True)
    p.add_argument("--lib_path", required=True)
    p.add_argument("--picks_csv", required=True)
    p.add_argument("--output_csv", required=True)
    p.add_argument("--timeout_s", type=float, default=1200.0)
    return p.parse_args()


def main() -> int:
    print(f"CurPID: {os.getpid()}")
    args = parse_args()
    dataset_path = Path(args.dataset_path)
    abc_bin = args.abc_bin
    lib_path = Path(args.lib_path)
    picks_csv = Path(args.picks_csv)
    output_csv = Path(args.output_csv)
    timeout_s = args.timeout_s

    with open(picks_csv, encoding="utf-8") as f:
        picks = list(csv.DictReader(f))

    print(f"{_now()} Loaded {len(picks)} cases from {picks_csv}")
    print(f"{_now()} abc_bin={abc_bin}  lib={lib_path}  timeout={timeout_s}s")

    rows: list[dict] = []
    for i, row in enumerate(picks):
        dataset = row["dataset"]
        case = row["case"]
        aig = _resolve_aig(dataset_path, dataset, case)
        out_row = {
            "dataset": dataset,
            "case": case,
            "area": "",
            "delay_ps": "",
            "runtime_s": "",
            "status": "error",
            "error_message": "",
        }

        print(f"{_now()} [{i+1}/{len(picks)}] {dataset}/{case}")

        if not aig.is_file():
            msg = f"AIG not found: {aig}"
            print(f"{_now()}   SKIP: {msg}")
            out_row["status"] = "missing"
            out_row["error_message"] = msg
            rows.append(out_row)
            _write_csv(output_csv, rows)
            continue

        command = (
            f"read {_abc_literal(aig)}; "
            f"read {_abc_literal(lib_path)}; "
            f"resyn2; map; topo; stime"
        )

        try:
            elapsed, stdout = _run_abc_timed(abc_bin, command, timeout_s)
            area, delay_ps = _parse_stime_metrics(stdout)
            out_row.update({
                "area": f"{area}",
                "delay_ps": f"{delay_ps}",
                "runtime_s": f"{elapsed:.4f}",
                "status": "ok",
            })
            print(
                f"{_now()}   ok area={area} delay={delay_ps}ps runtime={elapsed:.4f}s"
            )
        except subprocess.TimeoutExpired as exc:
            out_row["status"] = "timeout"
            out_row["error_message"] = f"timeout after {timeout_s}s"
            out_row["runtime_s"] = f"{timeout_s:.4f}"
            print(f"{_now()}   TIMEOUT: {exc}")
        except Exception as exc:
            out_row["error_message"] = str(exc)
            print(f"{_now()}   ERROR: {exc}")

        rows.append(out_row)
        _write_csv(output_csv, rows)

    print(f"{_now()} Saved {len(rows)} rows to {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
