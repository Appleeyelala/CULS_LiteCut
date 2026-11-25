#!/usr/bin/env python3
"""Tune adaptive tau and fanout alpha for 4B on the 4 pilot circuits (fanout-aware only)."""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from global_timing.aig_replay import (
    apply_and_write_aig,
    compute_adaptive_tau,
    global_po_sensitivity,
    output_tns_metrics,
    propagate_fanout_delays,
    verify_cec,
)
from global_timing.level_weights import propagate_levels
from global_timing.structure import parse_aig, references
from scripts.step4_fanout_pilot import PREV_MAP_SUMMARY, map_one, select_4b_fanout
from scripts.step4_pilot_replay import ABC_DUMP, ABC_ORIG, PILOTS, load_pilot_inputs, run_abc_dump


def run_adaptive_arm(
    name: str,
    orig_aig_path: Path,
    orig_aig,
    cuts_path: Path,
    work_dir: Path,
    arm: str,
    fanout_alpha: float,
    scales: tuple[float, ...],
    max_tau: float = 6.0,
    lam: float = 0.05,
    fanout_reg: float = 0.0005,
    verify_each_round: bool = True,
    pi_fanout_alpha: float = 0.0,
    allow_slack_area: bool = False,
) -> dict:
    t0 = time.perf_counter()
    work_dir.mkdir(parents=True, exist_ok=True)
    orig_lv = propagate_levels(orig_aig)

    cur_aig = orig_aig
    cur_aig_path = orig_aig_path
    cur_cuts = cuts_path
    round_logs = []
    applied_total = 0
    cec_all = True

    for rnd, scale in enumerate(scales, start=1):
        if rnd > 1:
            cur_cuts = run_abc_dump(
                ABC_DUMP, cur_aig_path, work_dir / f"round{rnd}_dump", work_dir / f"round{rnd}_dump.log"
            )
        fanouts = references(cur_aig)
        delay = propagate_fanout_delays(
            cur_aig, fanouts=fanouts, fanout_alpha=fanout_alpha, pi_fanout_alpha=pi_fanout_alpha
        )
        tau = compute_adaptive_tau(cur_aig, delay, min_tau=1.0, max_tau=max_tau, spread_coeff=0.2, scale=scale)
        _, sens = global_po_sensitivity(cur_aig, delay, tau=tau)
        ops = select_4b_fanout(
            cur_aig,
            delay,
            sens,
            cur_cuts,
            lam_area=lam,
            fanout_alpha=fanout_alpha,
            fanout_reg=fanout_reg,
            tau=tau,
            pi_fanout_alpha=pi_fanout_alpha,
            allow_slack_area=allow_slack_area,
        )
        if not ops:
            cur_lv = propagate_levels(cur_aig)
            m = output_tns_metrics(cur_aig, cur_lv, orig_lv.depth)
            round_logs.append({"round": rnd, "tau": tau, "applied_ops": 0, **m, "cec": True})
            break
        next_aig_path = work_dir / f"{name}.{arm}.r{rnd}.aig"
        next_aig = apply_and_write_aig(cur_aig, ops, next_aig_path)
        cec_ok = verify_cec(ABC_ORIG, orig_aig_path, next_aig_path, work_dir / f"{arm}.r{rnd}.cec.log") if verify_each_round else True
        cec_all = cec_all and cec_ok
        next_lv = propagate_levels(next_aig)
        next_fo = references(next_aig)
        m = output_tns_metrics(next_aig, next_lv, orig_lv.depth)
        applied_total += len(ops)
        round_logs.append({
            "round": rnd,
            "tau": tau,
            "applied_ops": len(ops),
            **m,
            "max_fo": max((next_fo[n] for n in range(next_aig.pi_count + 1, next_aig.maximum_variable + 1)), default=0),
            "cec": cec_ok,
        })
        cur_aig, cur_aig_path = next_aig, next_aig_path
        if not cec_ok:
            break

    cur_lv = propagate_levels(cur_aig)
    cur_fo = references(cur_aig)
    final_m = output_tns_metrics(cur_aig, cur_lv, orig_lv.depth)
    return {
        "arm": arm,
        "rounds": round_logs,
        "applied_total": applied_total,
        "cec": cec_all,
        "final_aig": str(cur_aig_path),
        "max_and_fo": max((cur_fo[n] for n in range(cur_aig.pi_count + 1, cur_aig.maximum_variable + 1)), default=0),
        "replay_runtime_s": time.perf_counter() - t0,
        **final_m,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    run = args.run.resolve()
    if not os.getenv("SLURM_JOB_ID"):
        raise SystemExit("Must run inside a Slurm job")

    out_dir = run / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    arm_specs = [
        ("4B_fo015_adap_i3", 0.15, (1.0, 1.0, 1.0), 6.0),
        ("4B_fo020_adap_i3", 0.20, (1.0, 1.0, 1.0), 6.0),
        ("4B_fo020_ann_i3", 0.20, (1.5, 1.0, 0.5), 6.0),
        ("4B_fo025_ann_i3", 0.25, (1.5, 1.0, 0.5), 6.0),
    ]

    pilots = load_pilot_inputs()
    prev_map = json.loads(PREV_MAP_SUMMARY.read_text())["pilots"]
    summary = {"slurm_job_id": os.getenv("SLURM_JOB_ID"), "pilots": {n: {"arms": {}} for n in PILOTS}}

    map_tasks = []
    for name in PILOTS:
        orig_p, cuts_p = pilots[name]
        orig_aig = parse_aig(orig_p)
        print(f"=== Replay {name} ===", flush=True)
        for arm, alpha, scales, max_tau in arm_specs:
            r_res = run_adaptive_arm(name, orig_p, orig_aig, cuts_p, out_dir / name / arm, arm, alpha, scales, max_tau=max_tau)
            summary["pilots"][name]["arms"][arm] = r_res
            taus = [round(r["tau"], 2) for r in r_res["rounds"]]
            print(
                f"  {arm:18s}: D={r_res['depth']:3d} AND={r_res['ands']:5d} maxFO={r_res['max_and_fo']:4d} "
                f"taus={taus} ops={r_res['applied_total']:5d} cec={r_res['cec']} t={r_res['replay_runtime_s']:.1f}s",
                flush=True,
            )
            map_tasks.append((name, arm, orig_p, Path(r_res["final_aig"]), out_dir / name / arm / "map"))

    print("=== ASAP7 Mapping ===", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for name, arm, m_res in pool.map(map_one, map_tasks):
            summary["pilots"][name]["arms"][arm]["mapping"] = m_res
            orig_delay = prev_map[name]["orig"]["delay_ps"]
            resyn_delay = prev_map[name]["abc_resyn2"]["delay_ps"]
            print(
                f"  {name:14s} {arm:18s}: delay={m_res['delay_ps']:9.3f} ps "
                f"(orig={orig_delay:.1f}, resyn2={resyn_delay:.1f}) area={m_res['area']:9.1f} cec={m_res['cec']}",
                flush=True,
            )

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print("DONE step4_adaptive_tau_pilot", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
