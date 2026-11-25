#!/usr/bin/env python3
"""Evaluate locked fanout-aware adaptive-tau 4B across all 905 open-sourceDataset circuits."""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from global_timing.evaluation import Evaluator
from global_timing.level_weights import propagate_levels
from global_timing.structure import parse_aig
from scripts.step4_adaptive_tau_pilot import run_adaptive_arm
from scripts.step4_fanout_pilot import ABC_EVAL, LIB_ASAP7
from scripts.step4_pilot_replay import ABC_DUMP, ABC_ORIG, run_abc_dump

DATASET_ROOT = Path("/home/wushangge/CULS_LiteCut/open-sourceDataset")


def _read_aig_header(path: Path) -> tuple[int, int, int, int, int]:
    with path.open("rb") as fh:
        fields = fh.readline().decode("ascii", errors="replace").strip().split()
    if len(fields) < 6 or fields[0] != "aig":
        raise ValueError(f"Unsupported AIG header in {path}: {fields[:6]}")
    return tuple(int(x) for x in fields[1:6])  # M, I, L, O, A


def _run_native_abc(orig_p: Path, cmd: str, out_aig: Path, log_path: Path, timeout: int = 600) -> None:
    out_aig.parent.mkdir(parents=True, exist_ok=True)
    abc_cmd = f"read_aiger {orig_p}; strash; {cmd}; write_aiger {out_aig}"
    proc = subprocess.run([str(ABC_ORIG), "-c", abc_cmd], capture_output=True, text=True, timeout=timeout)
    log_path.write_text(proc.stdout + proc.stderr)
    if proc.returncode != 0 or not out_aig.is_file():
        raise RuntimeError(f"ABC {cmd} failed rc={proc.returncode}")


def _eval_aig(orig_p: Path, cand_p: Path, map_dir: Path, timeout: int = 600) -> dict:
    ev = Evaluator(ABC_EVAL, LIB_ASAP7, orig_p, timeout=timeout)
    res = ev.evaluate(cand_p, map_dir)
    aig = parse_aig(cand_p)
    lv = propagate_levels(aig)
    return {
        "status": res["status"],
        "cec": res["cec"],
        "aig_depth": lv.depth,
        "aig_ands": aig.and_count,
        "delay_ps": res["delay_ps"],
        "area": res["area"],
        "mapped_gates": res["mapped_gates"],
        "adp": res["adp"],
        "map_runtime_s": res["runtime_s"],
    }


def _slug(rel_path: str) -> str:
    return rel_path.replace("/", "__").removesuffix(".aig")


def process_circuit(task: tuple[int, str, str, str, int]) -> dict:
    idx, rel_path, orig_str, out_root_str, timeout = task
    orig_p = Path(orig_str)
    out_root = Path(out_root_str)
    slug = _slug(rel_path)
    per_dir = out_root / "per_circuit"
    per_dir.mkdir(parents=True, exist_ok=True)
    done_json = per_dir / f"{slug}.json"
    if done_json.is_file():
        return json.loads(done_json.read_text())

    work_dir = out_root / "work" / slug
    if work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    _, pi_cnt, latch_cnt, po_cnt, orig_ands = _read_aig_header(orig_p)
    subset = rel_path.split("/", 1)[0]
    row: dict = {
        "index": idx,
        "name": orig_p.stem,
        "relative_path": rel_path,
        "subset": subset,
        "pi_count": pi_cnt,
        "po_count": po_cnt,
        "orig_ands": orig_ands,
    }

    if latch_cnt > 0:
        row["status"] = "SKIP_SEQUENTIAL"
        done_json.write_text(json.dumps(row, ensure_ascii=False) + "\n")
        shutil.rmtree(work_dir, ignore_errors=True)
        return row

    try:
        orig_aig = parse_aig(orig_p)
        cuts_p = run_abc_dump(ABC_DUMP, orig_p, work_dir / "r1_dump", work_dir / "r1_dump.log")
        replay = run_adaptive_arm(
            name=slug,
            orig_aig_path=orig_p,
            orig_aig=orig_aig,
            cuts_path=cuts_p,
            work_dir=work_dir / "4B_fo020_ann_i3",
            arm="4B_fo020_ann_i3",
            fanout_alpha=0.20,
            scales=(1.5, 1.0, 0.5),
            max_tau=6.0,
            lam=0.05,
            fanout_reg=0.0005,
            verify_each_round=False,
        )

        bal_aig = work_dir / "abc_balance.aig"
        rw_aig = work_dir / "abc_rewrite.aig"
        res2_aig = work_dir / "abc_resyn2.aig"
        _run_native_abc(orig_p, "balance", bal_aig, work_dir / "abc_balance.log", timeout=timeout)
        _run_native_abc(orig_p, "rewrite", rw_aig, work_dir / "abc_rewrite.log", timeout=timeout)
        _run_native_abc(
            orig_p,
            "balance; rewrite; refactor; balance; rewrite; rewrite -z; balance; refactor -z; rewrite -z; balance",
            res2_aig,
            work_dir / "abc_resyn2.log",
            timeout=timeout,
        )

        methods = {
            "orig": orig_p,
            "abc_balance": bal_aig,
            "abc_rewrite": rw_aig,
            "abc_resyn2": res2_aig,
            "4B_fo020_ann_i3": Path(replay["final_aig"]),
        }
        mapped = {}
        for mname, mp in methods.items():
            try:
                mapped[mname] = _eval_aig(orig_p, mp, work_dir / f"map_{mname}", timeout=timeout)
            except Exception as mexc:
                mapped[mname] = {"status": f"FAIL: {type(mexc).__name__}: {mexc}", "cec": "FAIL"}

        # Keep summary lightweight; remove bulky dumps/verilog while retaining metrics
        replay_compact = {k: v for k, v in replay.items() if k != "final_aig"}
        row["status"] = "OK" if mapped["4B_fo020_ann_i3"].get("status") == "PASS" else "PARTIAL"
        row["total_runtime_s"] = time.perf_counter() - t0
        row["replay"] = replay_compact
        row["mapping"] = mapped
    except Exception as exc:
        row["status"] = f"ERROR: {type(exc).__name__}: {exc}"
        row["total_runtime_s"] = time.perf_counter() - t0
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    tmp_json = per_dir / f".tmp.{slug}.{os.getpid()}.json"
    tmp_json.write_text(json.dumps(row, ensure_ascii=False) + "\n")
    tmp_json.replace(done_json)
    return row


def _geomean(vals: list[float]) -> float | None:
    pos = [v for v in vals if v > 0]
    return math.exp(sum(math.log(v) for v in pos) / len(pos)) if pos else None


def _band(ands: int) -> str:
    if ands < 1000:
        return "<1k"
    if ands < 10000:
        return "1k-10k"
    if ands < 50000:
        return "10k-50k"
    if ands <= 100000:
        return "50k-100k"
    return ">100k"


def summarize(rows: list[dict]) -> dict:
    methods = ("abc_balance", "abc_rewrite", "abc_resyn2", "4B_fo020_ann_i3")
    all_ok = [
        r for r in rows
        if r.get("mapping")
        and r["mapping"].get("orig", {}).get("status") == "PASS"
        and r["mapping"].get("4B_fo020_ann_i3", {}).get("status") == "PASS"
        and r["mapping"].get("4B_fo020_ann_i3", {}).get("cec") == "PASS"
    ]
    comparable = [
        r for r in all_ok
        if all(
            r["mapping"].get(m, {}).get("status") == "PASS" and r["mapping"].get(m, {}).get("cec") == "PASS"
            for m in ("abc_balance", "abc_rewrite", "abc_resyn2")
        )
    ]

    def stats_for(subset_rows: list[dict]) -> dict:
        out: dict = {"count": len(subset_rows)}
        if not subset_rows:
            return out
        for m in methods:
            dr = [r["mapping"][m]["delay_ps"] / r["mapping"]["orig"]["delay_ps"] for r in subset_rows]
            ar = [r["mapping"][m]["area"] / r["mapping"]["orig"]["area"] for r in subset_rows]
            out[m] = {
                "geomean_delay_vs_orig": _geomean(dr),
                "geomean_area_vs_orig": _geomean(ar),
                "wins_vs_orig": sum(1 for x in dr if x < 1.0 - 1e-6),
                "ties_vs_orig": sum(1 for x in dr if abs(x - 1.0) <= 1e-6),
            }
        for base in ("abc_balance", "abc_rewrite", "abc_resyn2"):
            rr = [r["mapping"]["4B_fo020_ann_i3"]["delay_ps"] / r["mapping"][base]["delay_ps"] for r in subset_rows]
            out[f"4B_vs_{base}"] = {
                "geomean_delay_ratio": _geomean(rr),
                "wins": sum(1 for x in rr if x < 1.0 - 1e-6),
                "ties": sum(1 for x in rr if abs(x - 1.0) <= 1e-6),
                "losses": sum(1 for x in rr if x > 1.0 + 1e-6),
            }
        return out

    by_subset: dict[str, list[dict]] = {}
    by_band: dict[str, list[dict]] = {}
    for r in comparable:
        by_subset.setdefault(r["subset"], []).append(r)
        by_band.setdefault(_band(r["orig_ands"]), []).append(r)

    return {
        "total_circuits": len(rows),
        "4B_and_orig_pass_cec": len(all_ok),
        "all_5_methods_pass_cec": len(comparable),
        "failures": [
            {"relative_path": r["relative_path"], "orig_ands": r["orig_ands"], "status": r["status"]}
            for r in rows if r not in comparable
        ],
        "overall_comparable": stats_for(comparable),
        "overall_4B_vs_orig": {
            "count": len(all_ok),
            "geomean_delay_vs_orig": _geomean([r["mapping"]["4B_fo020_ann_i3"]["delay_ps"] / r["mapping"]["orig"]["delay_ps"] for r in all_ok]),
            "geomean_area_vs_orig": _geomean([r["mapping"]["4B_fo020_ann_i3"]["area"] / r["mapping"]["orig"]["area"] for r in all_ok]),
            "wins_vs_orig": sum(1 for r in all_ok if r["mapping"]["4B_fo020_ann_i3"]["delay_ps"] < r["mapping"]["orig"]["delay_ps"] * (1 - 1e-6)),
        },
        "by_subset": {k: stats_for(v) for k, v in sorted(by_subset.items())},
        "by_size_band": {k: stats_for(v) for k, v in sorted(by_band.items())},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=60)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    run = args.run.resolve()
    if not os.getenv("SLURM_JOB_ID"):
        raise SystemExit("Must run inside a Slurm job")

    out_dir = run / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    aigs = sorted(DATASET_ROOT.rglob("*.aig"))
    sized = []
    for p in aigs:
        rel = str(p.relative_to(DATASET_ROOT))
        try:
            _, _, _, _, ands = _read_aig_header(p)
        except Exception:
            ands = 0
        sized.append((ands, rel, str(p)))
    # LPT schedule: largest circuits first so stragglers don't sit at the tail
    sized.sort(key=lambda x: (-x[0], x[1]))

    tasks = [(i, rel, p_str, str(out_dir), args.timeout) for i, (_, rel, p_str) in enumerate(sized)]
    print(f"Starting full {len(tasks)}-circuit open-sourceDataset evaluation with {args.workers} workers...", flush=True)

    rows = []
    done = 0
    t_start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(process_circuit, t) for t in tasks]
        for fut in as_completed(futs):
            row = fut.result()
            rows.append(row)
            done += 1
            if done % 50 == 0 or done == len(tasks) or row["status"] != "OK":
                elapsed = time.perf_counter() - t_start
                extra = ""
                if row["status"] == "OK":
                    mp = row["mapping"]
                    d0 = mp["orig"]["delay_ps"]
                    d4 = mp["4B_fo020_ann_i3"]["delay_ps"]
                    drs = mp["abc_resyn2"].get("delay_ps")
                    extra = f" | orig={d0:.1f} resyn2={drs} 4B={d4:.1f}"
                print(f"[{done:3d}/{len(tasks)} t={elapsed:6.1f}s] {row['relative_path']} (AND={row['orig_ands']}): {row['status']}{extra}", flush=True)

    rows.sort(key=lambda r: r["relative_path"])
    agg = summarize(rows)
    (out_dir / "summary.json").write_text(json.dumps({
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "aggregate": agg,
        "circuits": rows,
    }, indent=2, ensure_ascii=False) + "\n")
    print("=== AGGREGATE ===", flush=True)
    print(json.dumps(agg["overall_comparable"], indent=2, ensure_ascii=False), flush=True)
    print("DONE step4_full905_benchmark", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
