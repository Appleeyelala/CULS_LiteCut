from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import logging
import re
import shlex
import subprocess
from pathlib import Path

from extract_cutlib_teacher.logger import setup_logger
import os


DATASET_CASES: dict[str, list[str]] = {
    "OpenABC": ["aes", "aes_secworks", "wb_conmax"],
    "ACE": ["DMA_comb", "ethernet_comb", "hyp"],
    "forgeEDA": ["zigzag", "viterbi_data_forward", "uart_fifo1"],
    "iwls2022": ["ex27", "ex61", "ex67"],
    "iwls2024": ["ex21", "ex31", "ex96"],
}

CSV_COLUMNS = [
    "dataset",
    "case",
    "orig_area",
    "pred_area",
    "orig_delay_ps",
    "pred_delay_ps",
    "status",
    "error_message",
]

_STIME_METRICS_RE = re.compile(
    r"Area\s*=\s*([0-9]+(?:\.[0-9]+)?)"
    r".*?Delay\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*ps",
    re.IGNORECASE,
)
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run rw -qs validation and collect stime metrics")
    parser.add_argument("--dataset_path", required=True, help="Root directory containing dataset subdirectories")
    parser.add_argument("--abc_bin", required=True, help="Path to ABC executable")
    parser.add_argument("--lib_path", required=True, help="Path to liberty file used by ABC")
    parser.add_argument("--pred_dir", required=True, help="Directory for rewritten AIG outputs")
    parser.add_argument("--summary_path", required=True, help="CSV summary output path")
    parser.add_argument("--log_path", required=True, help="Log file output path")
    parser.add_argument("--all", action="store_true", help="Scan all dataset directories and .aig files under dataset_path")
    parser.add_argument("--jobs", type=int, default=1, help="Number of cases to process in parallel")
    parser.add_argument("--flush-every", type=int, default=1, help="Rewrite summary CSV every N completed cases")
    return parser.parse_args()


def _abc_literal(path: Path) -> str:
    return '"' + str(path).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _run_abc_command(abc_bin: str | Path, command: str) -> subprocess.CompletedProcess[str]:
    shell_command = f"{shlex.quote(str(abc_bin))} -c {shlex.quote(command)}"
    completed = subprocess.run(
        ["script", "-q", "-c", shell_command, "/dev/null"],
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"ABC command failed with exit code {completed.returncode}: {command}\n"
            f"stdout:\n{completed.stdout.strip()}\n"
            f"stderr:\n{completed.stderr.strip()}"
        )
    return completed


def _parse_stime_metrics(output: str) -> tuple[float, float]:
    cleaned_output = _ANSI_ESCAPE_RE.sub("", output)
    for line in cleaned_output.splitlines():
        match = _STIME_METRICS_RE.search(line)
        if match:
            return float(match.group(1)), float(match.group(2))
    raise ValueError("Area/Delay summary not found in stime output")


def _rewrite_command(lib_path: Path, orig_aig: Path, pred_aig: Path, logger: logging.Logger | None = None) -> str:
    abc_cmd = f"read {_abc_literal(lib_path)}; read {_abc_literal(orig_aig)}; rw -qs; write {_abc_literal(pred_aig)}"
    if logger:
        logger.debug("ABC rewrite command: %s", abc_cmd)
    return abc_cmd


def _stime_command(lib_path: Path, aig_path: Path) -> str:
    return f"read {_abc_literal(lib_path)}; read {_abc_literal(aig_path)}; map; topo; stime"

def _ori_sitme_command(lib_path: Path, aig_path: Path) -> str:
    return f"read {_abc_literal(lib_path)}; read {_abc_literal(aig_path)}; rw; map; topo; stime"

def _blank_row(dataset: str, case: str) -> dict[str, str]:
    return {
        "dataset": dataset,
        "case": case,
        # "orig_aig": str(orig_aig),
        # "pred_aig": str(pred_aig),
        "orig_area": "",
        "pred_area": "",
        "orig_delay_ps": "",
        "pred_delay_ps": "",
        "status": "error",
        "error_message": "",
    }


def _resolve_orig_aig_path(dataset_path: Path, dataset: str, case: str) -> Path:
    suffix = "_orig.aig" if dataset == "OpenABC" else ".aig"
    return dataset_path / dataset / f"{case}{suffix}"


def _iter_dataset_cases(dataset_path: Path, use_all: bool) -> list[tuple[str, str]]:
    if not use_all:
        return [(dataset, case) for dataset, cases in DATASET_CASES.items() for case in cases]

    items: list[tuple[str, str]] = []
    for dataset_dir in sorted(path for path in dataset_path.iterdir() if path.is_dir()):
        dataset = dataset_dir.name
        if dataset == "OpenABC":
            for aig_path in sorted(dataset_dir.glob("*_orig.aig")):
                items.append((dataset, aig_path.stem[: -len("_orig")]))
            continue
        for aig_path in sorted(dataset_dir.glob("*.aig")):
            items.append((dataset, aig_path.stem))
    return items


def _run_case(
    *,
    dataset: str,
    case: str,
    dataset_path: Path,
    abc_bin: str | Path,
    lib_path: Path,
    pred_dir: Path,
    logger: logging.Logger | None = None,
) -> dict[str, str]:
    orig_aig = _resolve_orig_aig_path(dataset_path, dataset, case)
    pred_aig = pred_dir / dataset / f"{case}.aig"
    row = _blank_row(dataset, case)
    active_logger = logger or LOGGER

    try:
        active_logger.info("Case start: %s/%s", dataset, case)
        if not orig_aig.is_file():
            raise FileNotFoundError(f"Original AIG not found: {orig_aig}")

        pred_aig.parent.mkdir(parents=True, exist_ok=True)
        if pred_aig.is_file():
            active_logger.info("Skip rewrite because pred AIG already exists: %s", pred_aig)
        else:
            _run_abc_command(abc_bin, _rewrite_command(lib_path, orig_aig, pred_aig, active_logger))
            if not pred_aig.is_file():
                raise FileNotFoundError(f"Predicted AIG not created: {pred_aig}")

        orig_stdout = _run_abc_command(abc_bin, _ori_sitme_command(lib_path, orig_aig)).stdout
        pred_stdout = _run_abc_command(abc_bin, _stime_command(lib_path, pred_aig)).stdout
        orig_area, orig_delay_ps = _parse_stime_metrics(orig_stdout)
        pred_area, pred_delay_ps = _parse_stime_metrics(pred_stdout)

        row.update(
            {
                "orig_area": f"{orig_area}",
                "pred_area": f"{pred_area}",
                "orig_delay_ps": f"{orig_delay_ps}",
                "pred_delay_ps": f"{pred_delay_ps}",
                "status": "ok",
            }
        )
        active_logger.info("Case done: %s/%s", dataset, case)
    except Exception as exc:
        row["error_message"] = str(exc)
        active_logger.exception("Case failed: %s/%s", dataset, case)
    return row


def _write_summary(summary_path: Path, rows: list[dict[str, str]]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _batch_exit_code(rows: list[dict[str, str]]) -> int:
    return 1 if any(row.get("status") == "error" for row in rows) else 0


def _should_flush(completed_count: int, total_count: int, flush_every: int) -> bool:
    if completed_count == 0:
        return False
    if completed_count == total_count:
        return True
    return completed_count % flush_every == 0


def main() -> int:
    global LOGGER
    args = parse_args()
    dataset_path = Path(args.dataset_path)
    abc_bin = args.abc_bin
    lib_path = Path(args.lib_path)
    pred_dir = Path(args.pred_dir)
    summary_path = Path(args.summary_path)
    LOGGER = setup_logger(args.log_path, name="rw_qs_stime_eval")
    jobs = max(1, args.jobs)
    flush_every = max(1, args.flush_every)
    dataset_cases = _iter_dataset_cases(dataset_path, use_all=args.all)
    
    LOGGER.info("cur PID: %d", os.getpid())
    LOGGER.info("RW-QS stime validation started")
    LOGGER.info("Dataset path: %s", dataset_path)
    LOGGER.info("Prediction dir: %s", pred_dir)
    LOGGER.info("Use all datasets/files: %s", args.all)
    LOGGER.info("Jobs: %d", jobs)
    LOGGER.info("Flush every: %d", flush_every)

    rows: list[dict[str, str]] = []
    total_count = len(dataset_cases)

    if jobs == 1:
        for dataset, case in dataset_cases:
            rows.append(
                _run_case(
                    dataset=dataset,
                    case=case,
                    dataset_path=dataset_path,
                    abc_bin=abc_bin,
                    lib_path=lib_path,
                    pred_dir=pred_dir,
                )
            )
            if _should_flush(len(rows), total_count, flush_every):
                _write_summary(summary_path, rows)
    else:
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            future_map = {
                executor.submit(
                    _run_case,
                    dataset=dataset,
                    case=case,
                    dataset_path=dataset_path,
                    abc_bin=abc_bin,
                    lib_path=lib_path,
                    pred_dir=pred_dir,
                ): (dataset, case)
                for dataset, case in dataset_cases
            }
            for future in as_completed(future_map):
                rows.append(future.result())
                if _should_flush(len(rows), total_count, flush_every):
                    _write_summary(summary_path, rows)

    exit_code = _batch_exit_code(rows)
    if exit_code != 0:
        LOGGER.warning("RW-QS stime validation finished with errors")
    LOGGER.info("Wrote summary CSV to %s", summary_path)
    LOGGER.info("RW-QS stime validation finished")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
