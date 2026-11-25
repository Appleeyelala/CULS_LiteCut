#!/usr/bin/env python3
"""Evaluate Claude v4 optimizations (PI phase-aware, slew coupling, strash-aware, fanout budget) on 16-circuit panel."""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from global_timing.aig_replay import (
    apply_and_write_aig,
    compute_adaptive_tau,
    global_po_sensitivity,
    output_tns_metrics,
    propagate_fanout_delays,
)
from global_timing.level_weights import propagate_levels
from global_timing.select4b import Select4BConfig, select_4b_v4
from global_timing.structure import parse_aig, references
from scripts.step4_pilot_replay import ABC_DUMP, run_abc_dump
from scripts.step5_longchain_panel import DS_ROOT, TARGET_PANEL, _eval_aig, _run_abc

PREV_V3_SUMMARY = Path("/data/wushangge/CULS_LiteCut/runs/20261125T055400Z-step5-panel-v3-f71e661f/output/summary.json")


def _aig_stats_v4(p: Path, fanout_alpha: float = 0.20, pi_fanout_alpha: float = 0.10, slew_beta: float = 0.005):
    aig = parse_aig(p)
    fos = references(aig)
    dp = propagate_fanout_delays(
        aig, fos, fanout_alpha=fanout_alpha, pi_fanout_alpha=pi_fanout_alpha,
        slew_beta=slew_beta, pi_phase_aware=True,
    )
    max_fo = max((fos[n] for n in range(1, aig.maximum_variable + 1)), default=0)
    excess_fo2 = sum(max(0, fos[n] - 6) ** 2 for n in range(1, aig.maximum_variable + 1))
    return aig, dp, max_fo, excess_fo2


def _run_v4_rounds(
    slug: str,
    start_p: Path,
    start_aig,
    r1_cuts: Path,
    work_dir: Path,
    arm: str,
    cfg: Select4BConfig,
    slew_beta: float,
    pi_phase_aware: bool = True,
    scales: tuple[float, ...] = (1.5, 1.0, 0.5),
) -> dict:
    t0 = time.perf_counter()
    work_dir.mkdir(parents=True, exist_ok=True)
    orig_lv = propagate_levels(start_aig)
    cur_aig = start_aig
    cur_p = start_p
    cur_cuts = r1_cuts
    applied_total = 0
    stats_all: dict = {}

    for rnd, scale in enumerate(scales, start=1):
        if rnd > 1:
            cur_cuts = run_abc_dump(ABC_DUMP, cur_p, work_dir / f"round{rnd}_dump", work_dir / f"round{rnd}_dump.log")
        fos = references(cur_aig)
        dp = propagate_fanout_delays(
            cur_aig, fos, fanout_alpha=cfg.fanout_alpha, pi_fanout_alpha=cfg.pi_fanout_alpha,
            slew_beta=slew_beta, pi_phase_aware=pi_phase_aware,
        )
        tau = compute_adaptive_tau(cur_aig, dp, min_tau=1.0, max_tau=6.0, spread_coeff=0.2, scale=scale)
        _, sens = global_po_sensitivity(cur_aig, dp, tau=tau)
        r_stats: dict = {}
        ops = select_4b_v4(cur_aig, dp, sens, cur_cuts, tau=tau, cfg=cfg, stats=r_stats)
        for k, v in r_stats.items():
            stats_all[k] = stats_all.get(k, 0) + v
        if not ops:
            break
        next_p = work_dir / f"{slug}.{arm}.r{rnd}.aig"
        cur_aig = apply_and_write_aig(cur_aig, ops, next_p)
        cur_p = next_p
        applied_total += len(ops)

    lv = propagate_levels(cur_aig)
    return {
        "final_aig": str(cur_p),
        "applied_total": applied_total,
        "stats": stats_all,
        "replay_runtime_s": time.perf_counter() - t0,
        **output_tns_metrics(cur_aig, lv, orig_lv.depth),
    }


def _cond_bal(cur_p: Path, work_dir: Path, tag: str, slew_beta: float) -> tuple[Path, bool]:
    _, cur_dp, cur_fo, cur_ex2 = _aig_stats_v4(cur_p, slew_beta=slew_beta)
    cand_p = work_dir / f"{tag}.aig"
    _run_abc(cur_p, "balance", cand_p, work_dir / f"{tag}.log")
    _, dp, fo, ex2 = _aig_stats_v4(cand_p, slew_beta=slew_beta)
    if fo <= max(64, cur_fo) and ex2 <= max(1000, int(cur_ex2 * 1.02)) and dp.depth <= cur_dp.depth - 1.0:
        return cand_p, True
    return cur_p, False


def _select_best_macro_v4(orig_p: Path, work_dir: Path, slew_beta: float) -> tuple[str, Path, object]:
    orig_a, orig_dp, orig_fo, orig_ex2 = _aig_stats_v4(orig_p, slew_beta=slew_beta)
    cands = [("orig", orig_p, orig_a, orig_dp, orig_ex2)]
    for tag, cmd in [("bal", "balance"), ("rfz", "refactor -z"), ("bal_rfz", "balance; refactor -z")]:
        cp = work_dir / f"pre_{tag}.aig"
        _run_abc(orig_p, cmd, cp, work_dir / f"pre_{tag}.log")
        a, dp, fo, ex2 = _aig_stats_v4(cp, slew_beta=slew_beta)
        if fo <= max(64, orig_fo) and ex2 <= max(1000, int(orig_ex2 * 1.02)) and dp.depth <= orig_dp.depth - 0.5:
            cands.append((tag, cp, a, dp, ex2))
    best = min(cands, key=lambda x: (round(x[3].depth, 3), x[4], x[2].and_count))
    return best[0], best[1], best[2]


def eval_circuit(item: tuple[str, str]) -> dict:
    rel_path, out_root_str = item
    orig_p = DS_ROOT / rel_path
    slug = rel_path.replace("/", "__").removesuffix(".aig")
    work_dir = Path(out_root_str) / slug
    if work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    orig_aig = parse_aig(orig_p)
    r1_cuts = run_abc_dump(ABC_DUMP, orig_p, work_dir / "orig_dump", work_dir / "orig_dump.log")

    cfg_v4 = Select4BConfig(
        lam_area=0.05, fanout_alpha=0.20, pi_fanout_alpha=0.10,
        fanout_reg=0.0005, fanout_pivot=6, allow_slack_area=False,
        strash_aware=True, per_root_k=2, fanout_budget=True,
    )

    # 1. Pure 4B_v4_slew (no balance/rfz)
    rep_pure = _run_v4_rounds(slug, orig_p, orig_aig, r1_cuts, work_dir / "4B_v4_slew", "4B_v4_slew", cfg_v4, slew_beta=0.005)

    # 2. bal_4B_v4_noslew (slew_beta=0.0, pi_phase_aware + strash + budget + k=2)
    b0_dir = work_dir / "bal_4B_v4_noslew"
    b0_dir.mkdir(parents=True, exist_ok=True)
    p_b0, acc_b0_pre = _cond_bal(orig_p, b0_dir, "pre_bal", slew_beta=0.0)
    a_b0 = parse_aig(p_b0) if acc_b0_pre else orig_aig
    cuts_b0 = run_abc_dump(ABC_DUMP, p_b0, b0_dir / "r1_dump", b0_dir / "r1_dump.log") if acc_b0_pre else r1_cuts
    rep_b0 = _run_v4_rounds(slug, p_b0, a_b0, cuts_b0, b0_dir / "core", "bal_4B_v4_noslew", cfg_v4, slew_beta=0.0)
    p_b0_fin, acc_b0_post = _cond_bal(Path(rep_b0["final_aig"]), b0_dir, "post_bal", slew_beta=0.0)

    # 3. bal_4B_v4_slew (slew_beta=0.005 + pi_phase_aware + strash + budget + k=2)
    b1_dir = work_dir / "bal_4B_v4_slew"
    b1_dir.mkdir(parents=True, exist_ok=True)
    p_b1, acc_b1_pre = _cond_bal(orig_p, b1_dir, "pre_bal", slew_beta=0.005)
    a_b1 = parse_aig(p_b1) if acc_b1_pre else orig_aig
    cuts_b1 = run_abc_dump(ABC_DUMP, p_b1, b1_dir / "r1_dump", b1_dir / "r1_dump.log") if acc_b1_pre else r1_cuts
    rep_b1 = _run_v4_rounds(slug, p_b1, a_b1, cuts_b1, b1_dir / "core", "bal_4B_v4_slew", cfg_v4, slew_beta=0.005)
    p_b1_fin, acc_b1_post = _cond_bal(Path(rep_b1["final_aig"]), b1_dir, "post_bal", slew_beta=0.005)

    # 4. struct_4B_v4_slew (best macro {orig, bal, rfz, bal_rfz} + v4_slew + post_bal)
    s_dir = work_dir / "struct_4B_v4_slew"
    s_dir.mkdir(parents=True, exist_ok=True)
    m_tag, p_s1, a_s1 = _select_best_macro_v4(orig_p, s_dir, slew_beta=0.005)
    cuts_s1 = r1_cuts if m_tag == "orig" else run_abc_dump(ABC_DUMP, p_s1, s_dir / "r1_dump", s_dir / "r1_dump.log")
    rep_s1 = _run_v4_rounds(slug, p_s1, a_s1, cuts_s1, s_dir / "core", "struct_4B_v4_slew", cfg_v4, slew_beta=0.005)
    p_s1_fin, acc_s1_post = _cond_bal(Path(rep_s1["final_aig"]), s_dir, "post_bal", slew_beta=0.005)

    arm_aigs = {
        "4B_v4_slew": Path(rep_pure["final_aig"]),
        "bal_4B_v4_noslew": p_b0_fin,
        "bal_4B_v4_slew": p_b1_fin,
        "struct_4B_v4_slew": p_s1_fin,
    }
    mapped = {m: _eval_aig(orig_p, mp, work_dir / f"map_{m}") for m, mp in arm_aigs.items()}
    shutil.rmtree(work_dir, ignore_errors=True)

    return {
        "relative_path": rel_path,
        "meta": {
            "bal_4B_v4_noslew": [acc_b0_pre, acc_b0_post],
            "bal_4B_v4_slew": [acc_b1_pre, acc_b1_post],
            "struct_4B_v4_slew": [m_tag, acc_s1_post],
        },
        "stats": {
            "4B_v4_slew": rep_pure["stats"],
            "bal_4B_v4_slew": rep_b1["stats"],
        },
        "mapping": mapped,
    }


def _gm(xs: list[float]) -> float:
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    run = args.run.resolve()
    if not os.getenv("SLURM_JOB_ID"):
        raise SystemExit("Must run inside a Slurm job")

    out_dir = run / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    prev = {r["relative_path"]: r["mapping"] for r in json.loads(PREV_V3_SUMMARY.read_text())["circuits"]}

    rows = []
    items = [(rel, str(out_dir / "work")) for rel in TARGET_PANEL]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for res in pool.map(eval_circuit, items):
            rel = res["relative_path"]
            mp = {**prev[rel], **res["mapping"]}
            res["mapping"] = mp
            rows.append(res)
            print(
                f"{rel:34s} | orig={mp['orig']['delay_ps']:8.1f} resyn2={mp['abc_resyn2']['delay_ps']:8.1f} "
                f"bal_v3={mp['bal_4B_v3']['delay_ps']:8.1f} struct_v3={mp['struct_4B_v3']['delay_ps']:8.1f} | "
                f"v4={mp['4B_v4_slew']['delay_ps']:8.1f} bal_v4_0={mp['bal_4B_v4_noslew']['delay_ps']:8.1f} "
                f"bal_v4_s={mp['bal_4B_v4_slew']['delay_ps']:8.1f} str_v4_s={mp['struct_4B_v4_slew']['delay_ps']:8.1f} | "
                f"cec={all(v['cec']=='PASS' for v in mp.values())}",
                flush=True,
            )

    print("=== Aggregate across 16-circuit target panel (v3 vs v4) ===", flush=True)
    for m in ("abc_resyn2", "4B_v3", "bal_4B_v3", "struct_4B_v3", "4B_v4_slew", "bal_4B_v4_noslew", "bal_4B_v4_slew", "struct_4B_v4_slew"):
        dr = [r["mapping"][m]["delay_ps"] / r["mapping"]["orig"]["delay_ps"] for r in rows]
        ar = [r["mapping"][m]["area"] / r["mapping"]["orig"]["area"] for r in rows]
        w_orig = sum(1 for x in dr if x < 1.0 - 1e-6)
        w_res = sum(1 for r in rows if r["mapping"][m]["delay_ps"] < r["mapping"]["abc_resyn2"]["delay_ps"] - 1e-6)
        print(f"  {m:18s}: delay={_gm(dr):.4f}x ({(_gm(dr)-1)*100:+6.2f}%), area={_gm(ar):.4f}x ({(_gm(ar)-1)*100:+6.2f}%), wins_orig={w_orig}/16, wins_resyn2={w_res}/16", flush=True)

    (out_dir / "summary.json").write_text(json.dumps({
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "circuits": rows,
    }, indent=2, ensure_ascii=False) + "\n")
    print("DONE step6_v4_panel", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
