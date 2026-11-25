#!/usr/bin/env python3
"""Full 905-circuit benchmark of mainline GPU struct_rw4b (rw4b -m -r 3) vs cached CPU/ABC baselines."""
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
from collections import defaultdict
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path

from scripts.step5_longchain_panel import DS_ROOT, _eval_aig

STEP4_FULL905_SUMMARY = Path(
    "/data/wushangge/CULS_LiteCut/runs/20261125T032740Z-step4-full905-a7d3f955/output/summary.json"
)
TOTAL_RE = re.compile(
    r"rw4b SUMMARY: ands=(\d+) level=(\d+) applied=(\d+) \| total=([0-9.]+)ms "
    r"\(gpu_timing=([0-9.]+)ms gpu_enum=([0-9.]+)ms gpu_eval=([0-9.]+)ms cpu_sel_rebuild=([0-9.]+)ms\)"
)
MACRO_RE = re.compile(r"rw4b \[pre_struct\]:.*->\s+(\w+)\s+\(D=([0-9.]+)\)")


def _run_gpuls(gpuls_bin: Path, in_aig: Path, out_aig: Path, rw4b_flags: str, timeout_s: int = 180) -> dict:
    cmd_str = f"read {in_aig}; rw4b {rw4b_flags}; write {out_aig}"
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            [str(gpuls_bin), "-c", cmd_str],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"status": "TIMEOUT", "wall_s": time.perf_counter() - t0}
    wall_s = time.perf_counter() - t0
    out = proc.stdout or ""
    if proc.returncode != 0 or not out_aig.exists() or "not recognized" in out:
        return {
            "status": f"ERROR_RC_{proc.returncode}",
            "wall_s": wall_s,
            "error_tail": out[-400:],
        }
    info: dict = {"status": "OK", "wall_s": wall_s}
    m = TOTAL_RE.search(out)
    if m:
        info.update({
            "ands": int(m.group(1)),
            "level": int(m.group(2)),
            "applied": int(m.group(3)),
            "rw4b_total_s": float(m.group(4)) / 1000.0,
            "gpu_timing_s": float(m.group(5)) / 1000.0,
            "gpu_enum_s": float(m.group(6)) / 1000.0,
            "gpu_eval_s": float(m.group(7)) / 1000.0,
            "cpu_sel_rebuild_s": float(m.group(8)) / 1000.0,
        })
    mm = MACRO_RE.search(out)
    if mm:
        info["pre_macro"] = mm.group(1)
    return info


def _eval_one(args_tuple: tuple[str, str, str]) -> dict:
    rel_path, out_aig_str, work_dir_str = args_tuple
    orig_p = DS_ROOT / rel_path
    out_p = Path(out_aig_str)
    w_dir = Path(work_dir_str)
    try:
        res = _eval_aig(orig_p, out_p, w_dir / "map")
    except Exception as exc:
        res = {"status": "ERROR", "cec": "ERROR", "error": f"{type(exc).__name__}: {exc}"}
    finally:
        shutil.rmtree(w_dir, ignore_errors=True)
    return res


def _gm(xs: list[float]) -> float:
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def _pct(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[idx]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--gpuls", required=True, type=Path)
    parser.add_argument("--flags", default="-m -r 3 -v 1")
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    if not os.getenv("SLURM_JOB_ID"):
        raise SystemExit("Must run inside a Slurm job")

    run = args.run.resolve()
    gpuls_bin = args.gpuls.resolve()
    out_dir = run / "output"
    work_dir = out_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "results.jsonl"

    base_data = json.loads(STEP4_FULL905_SUMMARY.read_text())
    base_circuits = base_data["circuits"]
    # LPT ordering: largest circuits first so slow ABC cec/map overlaps with fast GPU runs
    ordered = sorted(base_circuits, key=lambda c: c.get("orig_ands", 0), reverse=True)

    # Warmup GPU once on a small circuit
    warm_c = ordered[-1]["relative_path"]
    _run_gpuls(gpuls_bin, DS_ROOT / warm_c, work_dir / "_warm.aig", "-r 1 -v 0")
    if (work_dir / "_warm.aig").exists():
        (work_dir / "_warm.aig").unlink()

    t_all0 = time.perf_counter()
    pending: list[tuple[dict, dict, Future | None]] = []
    gpu_ok = 0
    gpu_err = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, c in enumerate(ordered, start=1):
            rel = c["relative_path"]
            orig_p = DS_ROOT / rel
            slug = rel.replace("/", "__").removesuffix(".aig")
            c_dir = work_dir / slug
            c_dir.mkdir(parents=True, exist_ok=True)
            out_aig = c_dir / "gpu_struct_rw4b.aig"

            ginfo = _run_gpuls(gpuls_bin, orig_p, out_aig, args.flags, timeout_s=180)
            if ginfo["status"] == "OK":
                gpu_ok += 1
                fut = pool.submit(_eval_one, (rel, str(out_aig), str(c_dir)))
            else:
                gpu_err += 1
                shutil.rmtree(c_dir, ignore_errors=True)
                fut = None
            pending.append((c, ginfo, fut))

            if i % 50 == 0 or i == len(ordered):
                elapsed = time.perf_counter() - t_all0
                done_eval = sum(1 for _, _, f in pending if f is None or f.done())
                print(
                    f"[GPU {i:3d}/{len(ordered)}] ok={gpu_ok} err={gpu_err} | "
                    f"eval_done={done_eval}/{i} | elapsed={elapsed:.1f}s",
                    flush=True,
                )

        gpu_phase_s = time.perf_counter() - t_all0
        print(f"=== GPU phase finished in {gpu_phase_s:.1f}s (ok={gpu_ok}, err={gpu_err}). Waiting for CPU ABC eval... ===", flush=True)

        rows = []
        with jsonl_path.open("w", encoding="utf-8") as jf:
            for idx, (c, ginfo, fut) in enumerate(pending, start=1):
                if fut is not None:
                    ev = fut.result()
                else:
                    ev = {"status": ginfo["status"], "cec": "SKIP"}
                row = {
                    "relative_path": c["relative_path"],
                    "subset": c.get("subset", "unknown"),
                    "orig_ands": c.get("orig_ands", 0),
                    "pi_count": c.get("pi_count", 0),
                    "po_count": c.get("po_count", 0),
                    "cpu_4b_replay_s": (c.get("replay") or {}).get("replay_runtime_s"),
                    "gpu_info": ginfo,
                    "gpu_struct_rw4b": ev,
                    "baselines": c.get("mapping", {}),
                }
                rows.append(row)
                jf.write(json.dumps(row, ensure_ascii=False) + "\n")
                if idx % 100 == 0 or idx == len(pending):
                    jf.flush()
                    print(f"[EVAL {idx:3d}/{len(pending)}] elapsed={time.perf_counter() - t_all0:.1f}s", flush=True)

    shutil.rmtree(work_dir, ignore_errors=True)

    # Aggregate statistics
    gpu_pass = [r for r in rows if r["gpu_struct_rw4b"].get("cec") == "PASS" and r["gpu_struct_rw4b"].get("delay_ps", 0) > 0]
    gpu_and_orig = [
        r for r in gpu_pass
        if (r["baselines"].get("orig") or {}).get("delay_ps", 0) > 0
    ]
    all_comp = [
        r for r in gpu_and_orig
        if all((r["baselines"].get(m) or {}).get("cec") == "PASS" and (r["baselines"].get(m) or {}).get("delay_ps", 0) > 0
               for m in ("abc_balance", "abc_rewrite", "abc_resyn2", "4B_fo020_ann_i3"))
    ]
    failures = [
        {
            "relative_path": r["relative_path"],
            "orig_ands": r["orig_ands"],
            "gpu_status": r["gpu_info"].get("status"),
            "cec": r["gpu_struct_rw4b"].get("cec"),
            "error": r["gpu_struct_rw4b"].get("error") or r["gpu_info"].get("error_tail", ""),
        }
        for r in rows
        if r["gpu_struct_rw4b"].get("cec") != "PASS"
    ]

    def _summarize(subset_rows: list[dict]) -> dict:
        if not subset_rows:
            return {"count": 0}
        out_m = {"count": len(subset_rows)}
        for m in ("abc_balance", "abc_rewrite", "abc_resyn2", "4B_fo020_ann_i3", "gpu_struct_rw4b"):
            def _get(r, key):
                return r["gpu_struct_rw4b"][key] if m == "gpu_struct_rw4b" else r["baselines"][m][key]
            dr = [_get(r, "delay_ps") / r["baselines"]["orig"]["delay_ps"] for r in subset_rows]
            ar = [_get(r, "area") / r["baselines"]["orig"]["area"] for r in subset_rows]
            adpr = [(_get(r, "delay_ps") * _get(r, "area")) / (r["baselines"]["orig"]["delay_ps"] * r["baselines"]["orig"]["area"]) for r in subset_rows]
            w_orig = sum(1 for x in dr if x < 1.0 - 1e-6)
            w_res = sum(1 for r in subset_rows if _get(r, "delay_ps") < r["baselines"]["abc_resyn2"]["delay_ps"] - 1e-6)
            w_cpu4b = sum(1 for r in subset_rows if _get(r, "delay_ps") < r["baselines"]["4B_fo020_ann_i3"]["delay_ps"] - 1e-6)
            out_m[m] = {
                "delay_geomean": _gm(dr),
                "delay_pct": (_gm(dr) - 1.0) * 100.0,
                "area_geomean": _gm(ar),
                "area_pct": (_gm(ar) - 1.0) * 100.0,
                "adp_geomean": _gm(adpr),
                "adp_pct": (_gm(adpr) - 1.0) * 100.0,
                "wins_orig": w_orig,
                "wins_resyn2": w_res,
                "wins_cpu4b": w_cpu4b,
            }
        return out_m

    # Timing & Speedup stats on circuits with valid CPU 4B runtime + GPU runtime
    gpu_wall = [r["gpu_info"]["wall_s"] for r in gpu_and_orig if "wall_s" in r["gpu_info"]]
    gpu_core = [r["gpu_info"]["rw4b_total_s"] for r in gpu_and_orig if "rw4b_total_s" in r["gpu_info"]]
    cpu_pair = [
        (r["cpu_4b_replay_s"], r["gpu_info"]["wall_s"], r["gpu_info"].get("rw4b_total_s"))
        for r in all_comp
        if r.get("cpu_4b_replay_s") and r["cpu_4b_replay_s"] > 0 and r["gpu_info"].get("wall_s", 0) > 0
    ]
    cpu_times = [x[0] for x in cpu_pair]
    wall_speedups = [x[0] / x[1] for x in cpu_pair]
    core_speedups = [x[0] / x[2] for x in cpu_pair if x[2] and x[2] > 0]

    timing_summary = {
        "n_gpu_ok": len(gpu_wall),
        "gpu_wall_median_s": statistics.median(gpu_wall) if gpu_wall else None,
        "gpu_wall_p90_s": _pct(gpu_wall, 0.90),
        "gpu_wall_sum_s": sum(gpu_wall),
        "gpu_core_median_s": statistics.median(gpu_core) if gpu_core else None,
        "gpu_core_p90_s": _pct(gpu_core, 0.90),
        "gpu_core_sum_s": sum(gpu_core),
        "cpu_4b_median_s": statistics.median(cpu_times) if cpu_times else None,
        "cpu_4b_p90_s": _pct(cpu_times, 0.90),
        "cpu_4b_sum_s": sum(cpu_times),
        "speedup_wall_median": statistics.median(wall_speedups) if wall_speedups else None,
        "speedup_wall_geomean": _gm(wall_speedups) if wall_speedups else None,
        "speedup_wall_total": (sum(cpu_times) / sum(x[1] for x in cpu_pair)) if cpu_pair else None,
        "speedup_core_median": statistics.median(core_speedups) if core_speedups else None,
        "speedup_core_geomean": _gm(core_speedups) if core_speedups else None,
        "speedup_core_total": (sum(cpu_times) / sum(x[2] for x in cpu_pair if x[2])) if core_speedups else None,
    }

    by_subset_rows: dict[str, list[dict]] = defaultdict(list)
    for r in all_comp:
        by_subset_rows[r["subset"]].append(r)
    by_subset = {k: _summarize(v) for k, v in sorted(by_subset_rows.items())}

    overall = _summarize(all_comp)
    summary = {
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "gpuls_bin": str(gpuls_bin),
        "flags": args.flags,
        "total_circuits": len(rows),
        "gpu_pass_cec": len(gpu_pass),
        "gpu_and_orig_pass": len(gpu_and_orig),
        "all_comparable_count": len(all_comp),
        "failures": failures,
        "timing": timing_summary,
        "overall_comparable": overall,
        "by_subset": by_subset,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    print("\n================ FULL 905 SUMMARY ================", flush=True)
    print(f"Total={len(rows)} | GPU cec=PASS: {len(gpu_pass)}/{len(rows)} | All-comparable: {len(all_comp)} | Failures: {len(failures)}", flush=True)
    for m in ("abc_balance", "abc_rewrite", "abc_resyn2", "4B_fo020_ann_i3", "gpu_struct_rw4b"):
        st = overall[m]
        print(
            f"  {m:16s}: delay={st['delay_geomean']:.4f}x ({st['delay_pct']:+6.2f}%), "
            f"area={st['area_geomean']:.4f}x ({st['area_pct']:+6.2f}%), "
            f"ADP={st['adp_geomean']:.4f}x ({st['adp_pct']:+6.2f}%), "
            f"wins_orig={st['wins_orig']}/{len(all_comp)}, wins_resyn2={st['wins_resyn2']}/{len(all_comp)}",
            flush=True,
        )
    print("Timing & Speedup (vs CPU 4B_fo020_ann_i3):", flush=True)
    print(json.dumps(timing_summary, indent=2), flush=True)
    print("DONE step8_gpu_struct_full905", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
