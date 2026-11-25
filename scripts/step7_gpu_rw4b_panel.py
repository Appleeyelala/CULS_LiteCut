#!/usr/bin/env python3
"""Benchmark GPU CULS rw4b (pure & conditional GPU balance) on the 16-circuit target panel."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from scripts.step5_longchain_panel import DS_ROOT, TARGET_PANEL, _eval_aig

PREV_FINAL_V4_SUMMARY = Path(
    "/data/wushangge/CULS_LiteCut/runs/20261125T072836Z-step6-final-v4-e5081d9a/output/summary.json"
)
TOTAL_RE = re.compile(
    r"\[rw4b\] total=([0-9.]+)s \(gpu_timing=([0-9.]+)s gpu_enum=([0-9.]+)s "
    r"gpu_eval=([0-9.]+)s cpu_sel_rebuild=([0-9.]+)s\)"
)


def _run_gpuls(gpuls_bin: Path, in_aig: Path, out_aig: Path, rw4b_flags: str, log_path: Path) -> dict:
    cmd_str = f"read {in_aig}; rw4b {rw4b_flags}; write {out_aig}"
    t0 = time.perf_counter()
    proc = subprocess.run(
        [str(gpuls_bin), "-c", cmd_str],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    wall_s = time.perf_counter() - t0
    log_path.write_text(proc.stdout, encoding="utf-8", errors="replace")
    if proc.returncode != 0 or not out_aig.exists():
        raise RuntimeError(f"gpuls failed (rc={proc.returncode}) on {in_aig}: {proc.stdout[-800:]}")
    breakdown = {}
    m = TOTAL_RE.search(proc.stdout)
    if m:
        breakdown = {
            "rw4b_total_s": float(m.group(1)),
            "gpu_timing_s": float(m.group(2)),
            "gpu_enum_s": float(m.group(3)),
            "gpu_eval_s": float(m.group(4)),
            "cpu_sel_rebuild_s": float(m.group(5)),
        }
    return {
        "wall_s": wall_s,
        "breakdown": breakdown,
    }


def _eval_pair(item: tuple[str, str, str, str]) -> dict:
    rel_path, out_pure_str, out_bal_str, eval_root_str = item
    orig_p = DS_ROOT / rel_path
    eval_root = Path(eval_root_str)
    eval_root.mkdir(parents=True, exist_ok=True)
    m_pure = _eval_aig(orig_p, Path(out_pure_str), eval_root / "map_gpu_rw4b")
    m_bal = _eval_aig(orig_p, Path(out_bal_str), eval_root / "map_gpu_bal_rw4b")
    shutil.rmtree(eval_root, ignore_errors=True)
    return {
        "relative_path": rel_path,
        "gpu_rw4b": m_pure,
        "gpu_bal_rw4b": m_bal,
    }


def _gm(xs: list[float]) -> float:
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--gpuls", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    if not os.getenv("SLURM_JOB_ID"):
        raise SystemExit("Must run inside a Slurm job")

    run = args.run.resolve()
    gpuls_bin = args.gpuls.resolve()
    if not gpuls_bin.exists():
        raise SystemExit(f"gpuls binary not found: {gpuls_bin}")

    out_dir = run / "output"
    work_dir = out_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    prev = {
        r["relative_path"]: r["mapping"]
        for r in json.loads(PREV_FINAL_V4_SUMMARY.read_text())["circuits"]
    }

    # 1. Warmup GPU context once on c1355
    warm_aig = DS_ROOT / TARGET_PANEL[0]
    _run_gpuls(gpuls_bin, warm_aig, work_dir / "_warm.aig", "-r 1", work_dir / "_warm.log")

    # 2. Sequential GPU execution on all 16 circuits (uncontended GPU timing)
    gpu_runs: dict[str, dict] = {}
    eval_items: list[tuple[str, str, str, str]] = []
    for rel in TARGET_PANEL:
        orig_p = DS_ROOT / rel
        slug = rel.replace("/", "__").removesuffix(".aig")
        c_dir = work_dir / slug
        c_dir.mkdir(parents=True, exist_ok=True)

        out_pure = c_dir / "gpu_rw4b.aig"
        out_bal = c_dir / "gpu_bal_rw4b.aig"
        t_pure = _run_gpuls(gpuls_bin, orig_p, out_pure, "-r 3 -v", c_dir / "gpu_rw4b.log")
        t_bal = _run_gpuls(gpuls_bin, orig_p, out_bal, "-b -r 3 -v", c_dir / "gpu_bal_rw4b.log")
        gpu_runs[rel] = {"gpu_rw4b": t_pure, "gpu_bal_rw4b": t_bal}
        print(
            f"[GPU] {rel:34s} | pure wall={t_pure['wall_s']:6.3f}s (core={t_pure['breakdown'].get('rw4b_total_s', 0):5.3f}s) | "
            f"bal wall={t_bal['wall_s']:6.3f}s (core={t_bal['breakdown'].get('rw4b_total_s', 0):5.3f}s)",
            flush=True,
        )
        eval_items.append((rel, str(out_pure), str(out_bal), str(c_dir / "eval")))

    # 3. Parallel ABC CEC + ASAP7 mapping on CPU workers
    eval_results: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for res in pool.map(_eval_pair, eval_items):
            eval_results[res["relative_path"]] = res

    rows = []
    print("\n=== Per-Circuit Results (16-circuit target panel) ===", flush=True)
    for rel in TARGET_PANEL:
        mp = {
            **prev[rel],
            "gpu_rw4b": eval_results[rel]["gpu_rw4b"],
            "gpu_bal_rw4b": eval_results[rel]["gpu_bal_rw4b"],
        }
        gt = gpu_runs[rel]
        rows.append({"relative_path": rel, "mapping": mp, "gpu_timing": gt})
        print(
            f"{rel:34s} | orig={mp['orig']['delay_ps']:8.1f} resyn2={mp['abc_resyn2']['delay_ps']:8.1f} | "
            f"py_v4s={mp['4B_v4_slew']['delay_ps']:8.1f} gpu_4b={mp['gpu_rw4b']['delay_ps']:8.1f} ({gt['gpu_rw4b']['wall_s']:5.2f}s) | "
            f"py_bal={mp['bal_v3g_v4s']['delay_ps']:8.1f} gpu_bal={mp['gpu_bal_rw4b']['delay_ps']:8.1f} ({gt['gpu_bal_rw4b']['wall_s']:5.2f}s) | "
            f"cec={mp['gpu_rw4b']['cec']}/{mp['gpu_bal_rw4b']['cec']}",
            flush=True,
        )

    print("\n=== Aggregate across 16-circuit target panel ===", flush=True)
    for m in ("abc_resyn2", "4B_v4_slew", "gpu_rw4b", "bal_v3g_v4s", "gpu_bal_rw4b"):
        dr = [r["mapping"][m]["delay_ps"] / r["mapping"]["orig"]["delay_ps"] for r in rows]
        ar = [r["mapping"][m]["area"] / r["mapping"]["orig"]["area"] for r in rows]
        w_orig = sum(1 for x in dr if x < 1.0 - 1e-6)
        w_res = sum(1 for r in rows if r["mapping"][m]["delay_ps"] < r["mapping"]["abc_resyn2"]["delay_ps"] - 1e-6)
        cec_ok = sum(1 for r in rows if r["mapping"][m]["cec"] == "PASS")
        extra = ""
        if m in ("gpu_rw4b", "gpu_bal_rw4b"):
            ws = [r["gpu_timing"][m]["wall_s"] for r in rows]
            cs = [r["gpu_timing"][m]["breakdown"].get("rw4b_total_s", 0.0) for r in rows]
            extra = f", wall_med={statistics.median(ws):.3f}s, wall_sum={sum(ws):.2f}s, core_sum={sum(cs):.2f}s"
        print(
            f"  {m:14s}: delay={_gm(dr):.4f}x ({(_gm(dr)-1)*100:+6.2f}%), "
            f"area={_gm(ar):.4f}x ({(_gm(ar)-1)*100:+6.2f}%), "
            f"wins_orig={w_orig}/16, wins_resyn2={w_res}/16, cec={cec_ok}/16{extra}",
            flush=True,
        )

    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "slurm_job_id": os.getenv("SLURM_JOB_ID"),
                "gpuls_bin": str(gpuls_bin),
                "circuits": rows,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    print("DONE step7_gpu_rw4b_panel", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
