#!/usr/bin/env python3
"""Evaluate locked fanout-aware adaptive-tau 4B across the 29 open-sourceDataset dev circuits."""
from __future__ import annotations

import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from global_timing.evaluation import Evaluator
from global_timing.level_weights import propagate_levels
from global_timing.structure import parse_aig
from scripts.step4_adaptive_tau_pilot import run_adaptive_arm
from scripts.step4_fanout_pilot import ABC_EVAL, LIB_ASAP7

EXPORT3 = Path("/data/wushangge/CULS_LiteCut/runs/20261123T090318Z-level-step3v3-export3-b5c9bc64/output")
NATIVE4_AIG = Path("/data/wushangge/CULS_LiteCut/runs/20261123T090800Z-level-step3v3-native4-be1f16b7/output/aig")


def _eval_aig(orig_p: Path, cand_p: Path, map_dir: Path) -> dict:
    ev = Evaluator(ABC_EVAL, LIB_ASAP7, orig_p, timeout=600)
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


def eval_one_circuit(item: tuple[dict, str]) -> dict:
    rec, out_root_str = item
    name = rec["name"]
    orig_p = Path(rec["path"])
    cuts_p = EXPORT3 / "dump" / name / "cuts.jsonl"
    work_dir = Path(out_root_str) / name
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        orig_aig = parse_aig(orig_p)
        replay = run_adaptive_arm(
            name=name,
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
        )
        methods = {
            "orig": orig_p,
            "abc_balance": NATIVE4_AIG / f"{name}.balance.aig",
            "abc_rewrite": NATIVE4_AIG / f"{name}.rewrite.aig",
            "abc_resyn2": NATIVE4_AIG / f"{name}.resyn2.aig",
            "4B_fo020_ann_i3": Path(replay["final_aig"]),
        }
        mapped = {}
        for mname, mp in methods.items():
            mapped[mname] = _eval_aig(orig_p, mp, work_dir / f"map_{mname}")

        return {
            "name": name,
            "relative_path": rec["relative_path"],
            "band": rec.get("band"),
            "orig_ands": rec["ands"],
            "status": "OK",
            "replay": replay,
            "mapping": mapped,
        }
    except Exception as exc:
        return {
            "name": name,
            "relative_path": rec["relative_path"],
            "band": rec.get("band"),
            "orig_ands": rec["ands"],
            "status": f"ERROR: {type(exc).__name__}: {exc}",
        }


def _geomean(vals: list[float]) -> float | None:
    pos = [v for v in vals if v > 0]
    if not pos:
        return None
    return math.exp(sum(math.log(v) for v in pos) / len(pos))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    run = args.run.resolve()
    if not os.getenv("SLURM_JOB_ID"):
        raise SystemExit("Must run inside a Slurm job")

    out_dir = run / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    cohort = json.loads((EXPORT3 / "cohort.json").read_text())
    items = []
    for r in cohort["sampled"]:
        name = r["name"]
        if (NATIVE4_AIG / f"{name}.resyn2.aig").is_file() and (EXPORT3 / "dump" / name / "cuts.jsonl").is_file():
            items.append((r, str(out_dir / "circuits")))

    print(f"Benchmarking {len(items)} open-sourceDataset circuits with {args.workers} workers...", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for res in pool.map(eval_one_circuit, items):
            rows.append(res)
            if res["status"] == "OK":
                mp = res["mapping"]
                d_orig = mp["orig"]["delay_ps"]
                d_res = mp["abc_resyn2"]["delay_ps"]
                d_4b = mp["4B_fo020_ann_i3"]["delay_ps"]
                a_orig = mp["orig"]["area"]
                a_4b = mp["4B_fo020_ann_i3"]["area"]
                cec = mp["4B_fo020_ann_i3"]["cec"]
                t_rep = res["replay"]["replay_runtime_s"]
                print(
                    f"  {res['name']:34s} ({res['relative_path']:32s}): "
                    f"orig={d_orig:8.1f} resyn2={d_res:8.1f} 4B={d_4b:8.1f} ps "
                    f"({(d_4b/d_orig - 1)*100:+5.1f}% vs orig, {(d_4b/d_res - 1)*100:+5.1f}% vs resyn2) | "
                    f"area={a_4b/a_orig:.3f}x cec={cec} t={t_rep:.1f}s",
                    flush=True,
                )
            else:
                print(f"  {res['name']:34s}: {res['status']}", flush=True)

    ok_rows = [r for r in rows if r["status"] == "OK" and r["mapping"]["4B_fo020_ann_i3"]["cec"] == "PASS"]
    ratios_vs_orig = [r["mapping"]["4B_fo020_ann_i3"]["delay_ps"] / r["mapping"]["orig"]["delay_ps"] for r in ok_rows]
    ratios_vs_resyn2 = [r["mapping"]["4B_fo020_ann_i3"]["delay_ps"] / r["mapping"]["abc_resyn2"]["delay_ps"] for r in ok_rows]
    resyn2_vs_orig = [r["mapping"]["abc_resyn2"]["delay_ps"] / r["mapping"]["orig"]["delay_ps"] for r in ok_rows]
    area_vs_orig = [r["mapping"]["4B_fo020_ann_i3"]["area"] / r["mapping"]["orig"]["area"] for r in ok_rows]
    area_resyn2_vs_orig = [r["mapping"]["abc_resyn2"]["area"] / r["mapping"]["orig"]["area"] for r in ok_rows]

    agg = {
        "total_attempted": len(items),
        "total_ok_cec": len(ok_rows),
        "wins_delay_vs_orig": sum(1 for x in ratios_vs_orig if x < 1.0 - 1e-6),
        "wins_delay_vs_resyn2": sum(1 for x in ratios_vs_resyn2 if x < 1.0 - 1e-6),
        "geomean_4B_delay_vs_orig": _geomean(ratios_vs_orig),
        "geomean_resyn2_delay_vs_orig": _geomean(resyn2_vs_orig),
        "geomean_4B_delay_vs_resyn2": _geomean(ratios_vs_resyn2),
        "geomean_4B_area_vs_orig": _geomean(area_vs_orig),
        "geomean_resyn2_area_vs_orig": _geomean(area_resyn2_vs_orig),
    }
    print("=== Aggregate (OK & CEC=PASS) ===", flush=True)
    print(json.dumps(agg, indent=2), flush=True)

    (out_dir / "summary.json").write_text(json.dumps({
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "aggregate": agg,
        "circuits": rows,
    }, indent=2, ensure_ascii=False) + "\n")
    print("DONE step4_cohort_benchmark", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
