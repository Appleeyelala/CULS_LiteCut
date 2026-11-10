from __future__ import annotations

import argparse
import csv
from pathlib import Path


OUTPUT_COLUMNS = [
    "dataset",
    "case",
    "orig_area",
    "pred_area",
    "orig_delay_ps",
    "pred_delay_ps",
    "delay_ratio",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze rw-qs stime summary and sort cases by pred_delay/orig_delay"
    )
    parser.add_argument("--summary_path", required=True, help="Input summary CSV path")
    parser.add_argument("--output_path", required=True, help="Output CSV path for sorted analysis")
    return parser.parse_args()


def _load_summary_rows(summary_path: Path) -> list[dict[str, str]]:
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _build_analysis_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    analyzed_rows: list[dict[str, str]] = []
    for row in rows:
        if row.get("status") != "ok":
            continue

        orig_delay_ps = float(row["orig_delay_ps"])
        pred_delay_ps = float(row["pred_delay_ps"])
        if orig_delay_ps <= 0:
            continue

        analyzed_rows.append(
            {
                "dataset": row["dataset"],
                "case": row["case"],
                "orig_area": row["orig_area"],
                "pred_area": row["pred_area"],
                "orig_delay_ps": row["orig_delay_ps"],
                "pred_delay_ps": row["pred_delay_ps"],
                "delay_ratio": f"{pred_delay_ps / orig_delay_ps:.6f}",
            }
        )

    analyzed_rows.sort(key=lambda row: (float(row["delay_ratio"]), row["dataset"], row["case"]))
    return analyzed_rows


def _write_analysis_rows(output_path: Path, rows: list[dict[str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    summary_path = Path(args.summary_path)
    output_path = Path(args.output_path)

    rows = _load_summary_rows(summary_path)
    analyzed_rows = _build_analysis_rows(rows)
    _write_analysis_rows(output_path, analyzed_rows)

    print(f"Wrote {len(analyzed_rows)} rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
