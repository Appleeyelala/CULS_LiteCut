#!/usr/bin/env python3
"""Confirm final mainline combo: v3 safe macro gate (non-phase/excess_fo2) + v4_slew core on 16-circuit panel."""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from global_timing.select4b import Select4BConfig
from global_timing.structure import parse_aig
from scripts.step4_pilot_replay import ABC_DUMP, run_abc_dump
from scripts.step5_longchain_panel import (
    DS_ROOT,
    TARGET_PANEL,
    _aig_stats,
    _conditional_post_bal,
    _eval_aig,
    _run_abc,
    _select_best_pre_macro,
)
from scripts.step6_v4_panel import PREV_V3_SUMMARY, _run_v4_rounds

PREV_V4_SUMMARY = Path("/data/wushangge/CULS_LiteCut/runs/20261125T065157Z-step6-v4-panel-a899e192/output/summary.json")


def eval_circuit(item: tuple[str, str]) -> dict:
    rel_path, out_root_str = item
    orig_p = DS_ROOT / rel_path
    slug = rel_path.replace("/", "__").removesuffix(".aig")
    work_dir = Path(out_root_str) / slug
    if work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    orig_aig, orig_dp, orig_fo, orig_ex2 = _aig_stats(orig_p)
    r1_cuts = run_abc_dump(ABC_DUMP, orig_p, work_dir / "orig_dump", work_dir / "orig_dump.log")

    cfg_v4 = Select4BConfig(
        lam_area=0.05, fanout_alpha=0.20, pi_fanout_alpha=0.10,
        fanout_reg=0.0005, fanout_pivot=6, allow_slack_area=False,
        strash_aware=True, per_root_k=2, fanout_budget=True,
    )

    # 1. bal_v3g_v4s: v3 safe pre-balance gate -> 4B_v4_slew (3 rounds) -> v3 safe post-balance gate
    b_dir = work_dir / "bal_v3g_v4s"
    b_dir.mkdir(parents=True, exist_ok=True)
    p_b1 = b_dir / "pre_bal.aig"
    _run_abc(orig_p, "balance", p_b1, b_dir / "pre_bal.log")
    a_b1, dp_b1, fo_b1, ex2_b1 = _aig_stats(p_b1)
    acc_pre_bal = (
        fo_b1 <= max(64, orig_fo)
        and ex2_b1 <= max(1000, int(orig_ex2 * 1.02))
        and dp_b1.depth <= orig_dp.depth - 1.0
    )
    start_p, start_a = (p_b1, a_b1) if acc_pre_bal else (orig_p, orig_aig)
    cuts_b1 = run_abc_dump(ABC_DUMP, start_p, b_dir / "r1_dump", b_dir / "r1_dump.log") if acc_pre_bal else r1_cuts
    rep_b = _run_v4_rounds(slug, start_p, start_a, cuts_b1, b_dir / "core", "bal_v3g_v4s", cfg_v4, slew_beta=0.005)
    p_b_fin, acc_post_bal = _conditional_post_bal(Path(rep_b["final_aig"]), b_dir, "post_bal")

    # 2. struct_v3g_v4s: v3 safe pre-macro {orig, bal, rfz, bal_rfz} -> 4B_v4_slew -> v3 safe post-balance
    s_dir = work_dir / "struct_v3g_v4s"
    s_dir.mkdir(parents=True, exist_ok=True)
    macro_tag, p_s1, a_s1, _, _, _ = _select_best_pre_macro(orig_p, s_dir)
    if macro_tag == "orig":
        cuts_s1 = r1_cuts
    elif macro_tag == "bal" and acc_pre_bal:
        cuts_s1 = cuts_b1
    else:
        cuts_s1 = run_abc_dump(ABC_DUMP, p_s1, s_dir / "r1_dump", s_dir / "r1_dump.log")
    rep_s = _run_v4_rounds(slug, p_s1, a_s1, cuts_s1, s_dir / "core", "struct_v3g_v4s", cfg_v4, slew_beta=0.005)
    p_s_fin, acc_s_post = _conditional_post_bal(Path(rep_s["final_aig"]), s_dir, "post_bal")

    mapped = {
        "bal_v3g_v4s": _eval_aig(orig_p, p_b_fin, work_dir / "map_bal_v3g_v4s"),
        "struct_v3g_v4s": _eval_aig(orig_p, p_s_fin, work_dir / "map_struct_v3g_v4s"),
    }
    shutil.rmtree(work_dir, ignore_errors=True)
    return {
        "relative_path": rel_path,
        "meta": {
            "bal_v3g_v4s": [acc_pre_bal, acc_post_bal],
            "struct_v3g_v4s": [macro_tag, acc_s_post],
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
    prev_v3 = {r["relative_path"]: r["mapping"] for r in json.loads(PREV_V3_SUMMARY.read_text())["circuits"]}
    prev_v4 = {r["relative_path"]: r["mapping"] for r in json.loads(PREV_V4_SUMMARY.read_text())["circuits"]}

    rows = []
    items = [(rel, str(out_dir / "work")) for rel in TARGET_PANEL]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for res in pool.map(eval_circuit, items):
            rel = res["relative_path"]
            mp = {**prev_v3[rel], "4B_v4_slew": prev_v4[rel]["4B_v4_slew"], **res["mapping"]}
            res["mapping"] = mp
            rows.append(res)
            print(
                f"{rel:34s} | orig={mp['orig']['delay_ps']:8.1f} resyn2={mp['abc_resyn2']['delay_ps']:8.1f} "
                f"bal_v3={mp['bal_4B_v3']['delay_ps']:8.1f} str_v3={mp['struct_4B_v3']['delay_ps']:8.1f} | "
                f"v4s={mp['4B_v4_slew']['delay_ps']:8.1f} bal_v3g_v4s={mp['bal_v3g_v4s']['delay_ps']:8.1f} "
                f"str_v3g_v4s={mp['struct_v3g_v4s']['delay_ps']:8.1f} | meta={res['meta']} cec={all(v['cec']=='PASS' for v in mp.values())}",
                flush=True,
            )

    print("=== Aggregate across 16-circuit target panel ===", flush=True)
    for m in ("abc_resyn2", "4B_v4_slew", "bal_4B_v3", "struct_4B_v3", "bal_v3g_v4s", "struct_v3g_v4s"):
        dr = [r["mapping"][m]["delay_ps"] / r["mapping"]["orig"]["delay_ps"] for r in rows]
        ar = [r["mapping"][m]["area"] / r["mapping"]["orig"]["area"] for r in rows]
        w_orig = sum(1 for x in dr if x < 1.0 - 1e-6)
        w_res = sum(1 for r in rows if r["mapping"][m]["delay_ps"] < r["mapping"]["abc_resyn2"]["delay_ps"] - 1e-6)
        print(f"  {m:16s}: delay={_gm(dr):.4f}x ({(_gm(dr)-1)*100:+6.2f}%), area={_gm(ar):.4f}x ({(_gm(ar)-1)*100:+6.2f}%), wins_orig={w_orig}/16, wins_resyn2={w_res}/16", flush=True)

    (out_dir / "summary.json").write_text(json.dumps({
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "circuits": rows,
    }, indent=2, ensure_ascii=False) + "\n")
    print("DONE step6_final_v4", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
