#!/usr/bin/env python3
"""Evaluate real ASAP7 technology mapping delay & area for ABC baselines vs 4B on the 4 pilots."""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from global_timing.evaluation import Evaluator
from global_timing.level_weights import propagate_levels
from global_timing.structure import parse_aig

PILOTS = ("c1355", "c5315", "intra_pred_PE", "cabac_ucontext")
ABC_EVAL = Path("/data/wushangge/CULS_LiteCut/runs/20261111T173355Z-acyclic-replay-tool-build-64cf764e/build/framework_tools/abc/abc")
LIB_ASAP7 = Path("/data/wushangge/CULS_LiteCut/assets/libraries/asap7.lib")
EXPORT3 = Path("/data/wushangge/CULS_LiteCut/runs/20261123T090318Z-level-step3v3-export3-b5c9bc64/output")
NATIVE4_AIG = Path("/data/wushangge/CULS_LiteCut/runs/20261123T090800Z-level-step3v3-native4-be1f16b7/output/aig")
STEP4_OUT = Path("/data/wushangge/CULS_LiteCut/runs/20261124T082423Z-step4-pilot-replay-387df399/output")


def eval_one(task: tuple[str, str, Path, Path, Path]) -> tuple[str, str, dict]:
    circuit, method, orig_path, cand_path, work_dir = task
    ev = Evaluator(ABC_EVAL, LIB_ASAP7, orig_path, timeout=600)
    res = ev.evaluate(cand_path, work_dir)
    aig = parse_aig(cand_path)
    lv = propagate_levels(aig)
    return circuit, method, {
        "status": res["status"],
        "cec": res["cec"],
        "aig_depth": lv.depth,
        "aig_ands": aig.and_count,
        "delay_ps": res["delay_ps"],
        "area": res["area"],
        "mapped_gates": res["mapped_gates"],
        "adp": res["adp"],
        "runtime_s": res["runtime_s"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    run = args.run.resolve()
    if not os.getenv("SLURM_JOB_ID"):
        raise SystemExit("Must run inside a Slurm job")

    cohort = json.loads((EXPORT3 / "cohort.json").read_text())
    orig_map = {r["name"]: Path(r["path"]) for r in cohort["pilots"]}
    out_dir = run / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for name in PILOTS:
        orig_p = orig_map[name]
        candidates = {
            "orig": orig_p,
            "abc_balance": NATIVE4_AIG / f"{name}.balance.aig",
            "abc_rewrite": NATIVE4_AIG / f"{name}.rewrite.aig",
            "abc_resyn2": NATIVE4_AIG / f"{name}.resyn2.aig",
            "4B_r1_area": STEP4_OUT / name / "4B_r1_area" / f"{name}.4B_r1_area.r1.aig",
            "4B_iter3_area": STEP4_OUT / name / "4B_iter3_area" / f"{name}.4B_iter3_area.r3.aig",
            "4B_iter3_free": STEP4_OUT / name / "4B_iter3_free" / f"{name}.4B_iter3_free.r3.aig",
        }
        for method, cand_p in candidates.items():
            if not cand_p.is_file():
                raise SystemExit(f"Missing AIG: {cand_p}")
            tasks.append((name, method, orig_p, cand_p, out_dir / name / method))

    results = {name: {} for name in PILOTS}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for circuit, method, row in pool.map(eval_one, tasks):
            results[circuit][method] = row
            print(
                f"{circuit:14s} {method:14s}: aig_D={row['aig_depth']:3d} AND={row['aig_ands']:5d} | "
                f"delay={row['delay_ps']:9.3f} ps, area={row['area']:10.3f}, gates={row['mapped_gates']:5d}, cec={row['cec']}",
                flush=True,
            )

    (out_dir / "summary.json").write_text(json.dumps({"slurm_job_id": os.getenv("SLURM_JOB_ID"), "pilots": results}, indent=2) + "\n")
    print("DONE step4_pilot_mapping", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
