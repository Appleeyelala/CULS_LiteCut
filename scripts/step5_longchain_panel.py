#!/usr/bin/env python3
"""Verify v3 refinements (PI fanout + strict slack guard + excess-fanout-guarded balance/rfz) on 16-circuit panel."""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from global_timing.aig_replay import propagate_fanout_delays
from global_timing.evaluation import Evaluator
from global_timing.level_weights import propagate_levels
from global_timing.structure import parse_aig, references
from scripts.step4_adaptive_tau_pilot import run_adaptive_arm
from scripts.step4_fanout_pilot import ABC_EVAL, LIB_ASAP7
from scripts.step4_pilot_replay import ABC_DUMP, ABC_ORIG, run_abc_dump

DS_ROOT = Path("/data/wushangge/CULS_LiteCut/assets/inputs/paper_audit_905")

TARGET_PANEL = (
    # 4 core pilots
    "OpenABC/c1355_orig.aig",
    "OpenABC/c5315_orig.aig",
    "forgeEDA/intra_pred_PE.aig",
    "forgeEDA/cabac_ucontext.aig",
    # 10 representative long-chain / refactor / PI-fanout gaps
    "forgeEDA/SQ_stage.aig",
    "forgeEDA/sram_64.aig",
    "forgeEDA/reservation_station.aig",
    "forgeEDA/BMD_EP_MEM.aig",
    "forgeEDA/sirv_sim_ram.aig",
    "ACE/des_perf_comb.aig",
    "ACE/vga_lcd_comb.aig",
    "forgeEDA/dmem.aig",
    "forgeEDA/branch_prediction_globle.aig",
    "iwls2022/ex65.aig",
    # 2 high-fanout 4B-win guards where balance/resyn2 regressed
    "forgeEDA/pa_f_spsram_1024x4.aig",
    "OpenABC/picosoc_orig.aig",
)


def _run_abc(in_p: Path, cmd: str, out_p: Path, log_p: Path, timeout: int = 600) -> None:
    out_p.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [str(ABC_ORIG), "-c", f"read_aiger {in_p}; strash; {cmd}; write_aiger {out_p}"],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    log_p.write_text(proc.stdout + proc.stderr)
    if proc.returncode != 0 or not out_p.is_file():
        raise RuntimeError(f"ABC `{cmd}` failed rc={proc.returncode}")


def _aig_stats(p: Path, fanout_alpha: float = 0.20, pi_fanout_alpha: float = 0.10):
    aig = parse_aig(p)
    fos = references(aig)
    dp = propagate_fanout_delays(aig, fos, fanout_alpha=fanout_alpha, pi_fanout_alpha=pi_fanout_alpha)
    max_fo = max((fos[n] for n in range(1, aig.maximum_variable + 1)), default=0)
    excess_fo2 = sum(max(0, fos[n] - 6) ** 2 for n in range(1, aig.maximum_variable + 1))
    return aig, dp, max_fo, excess_fo2


def _select_best_pre_macro(orig_p: Path, work_dir: Path) -> tuple[str, Path, object, object, int, int]:
    """Choose best AIG-level starting structure among `{orig, bal, rfz, bal;rfz}` by `(dp.depth, excess_fo2)`."""
    orig_aig, orig_dp, orig_fo, orig_ex2 = _aig_stats(orig_p)
    candidates = [("orig", orig_p, orig_aig, orig_dp, orig_fo, orig_ex2)]
    for tag, cmd in [("bal", "balance"), ("rfz", "refactor -z"), ("bal_rfz", "balance; refactor -z")]:
        cand_p = work_dir / f"pre_{tag}.aig"
        _run_abc(orig_p, cmd, cand_p, work_dir / f"pre_{tag}.log")
        a, dp, fo, ex2 = _aig_stats(cand_p)
        if fo <= max(64, orig_fo) and ex2 <= max(1000, int(orig_ex2 * 1.02)) and dp.depth <= orig_dp.depth - 0.5:
            candidates.append((tag, cand_p, a, dp, fo, ex2))

    # Prefer strictly lower AIG fanout-aware depth, then lower excess_fo2, then fewer ANDs
    best = min(candidates, key=lambda x: (round(x[3].depth, 3), x[5], x[2].and_count))
    return best


def _conditional_post_bal(cur_p: Path, work_dir: Path, tag: str) -> tuple[Path, bool]:
    cur_a, cur_dp, cur_fo, cur_ex2 = _aig_stats(cur_p)
    cand_p = work_dir / f"{tag}.aig"
    _run_abc(cur_p, "balance", cand_p, work_dir / f"{tag}.log")
    a, dp, fo, ex2 = _aig_stats(cand_p)
    if fo <= max(64, cur_fo) and ex2 <= max(1000, int(cur_ex2 * 1.02)) and dp.depth <= cur_dp.depth - 1.0:
        return cand_p, True
    return cur_p, False


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


def eval_circuit(item: tuple[str, str]) -> dict:
    rel_path, out_root_str = item
    orig_p = DS_ROOT / rel_path
    slug = rel_path.replace("/", "__").removesuffix(".aig")
    work_dir = Path(out_root_str) / slug
    if work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    orig_aig, orig_dp, orig_fo, orig_ex2 = _aig_stats(orig_p)
    r1_cuts = run_abc_dump(ABC_DUMP, orig_p, work_dir / "orig_dump", work_dir / "orig_dump.log")

    # 1. Baseline v1 (previous 4B_fo020_ann_i3)
    rep_v1 = run_adaptive_arm(
        name=slug, orig_aig_path=orig_p, orig_aig=orig_aig, cuts_path=r1_cuts,
        work_dir=work_dir / "4B_v1", arm="4B_v1", fanout_alpha=0.20,
        scales=(1.5, 1.0, 0.5), max_tau=6.0, lam=0.05, fanout_reg=0.0005, verify_each_round=False,
    )

    # 2. Pure 4B_v3 (pi_fanout_alpha=0.10, allow_slack_area=False!)
    rep_v3 = run_adaptive_arm(
        name=slug, orig_aig_path=orig_p, orig_aig=orig_aig, cuts_path=r1_cuts,
        work_dir=work_dir / "4B_v3", arm="4B_v3", fanout_alpha=0.20,
        scales=(1.5, 1.0, 0.5), max_tau=6.0, lam=0.05, fanout_reg=0.0005, verify_each_round=False,
        pi_fanout_alpha=0.10, allow_slack_area=False,
    )

    # 3. bal_4B_v3 (excess-fo-guarded balance -> 4B_v3 -> excess-fo-guarded post balance)
    b_dir = work_dir / "bal_4B_v3"
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
    rep_bal_v3 = run_adaptive_arm(
        name=slug, orig_aig_path=orig_p, orig_aig=start_a, cuts_path=cuts_b1,
        work_dir=b_dir / "core", arm="bal_4B_v3", fanout_alpha=0.20,
        scales=(1.5, 1.0, 0.5), max_tau=6.0, lam=0.05, fanout_reg=0.0005, verify_each_round=False,
        pi_fanout_alpha=0.10, allow_slack_area=False,
    )
    p_bal_final, acc_post_bal = _conditional_post_bal(Path(rep_bal_v3["final_aig"]), b_dir, "post_bal")

    # 4. struct_4B_v3 (best AIG-level macro from {orig, bal, rfz, bal_rfz} -> 4B_v3 -> post balance)
    s_dir = work_dir / "struct_4B_v3"
    s_dir.mkdir(parents=True, exist_ok=True)
    macro_tag, p_s1, a_s1, _, _, _ = _select_best_pre_macro(orig_p, s_dir)
    if macro_tag == "orig":
        cuts_s1 = r1_cuts
    elif macro_tag == "bal" and acc_pre_bal:
        cuts_s1 = cuts_b1
    else:
        cuts_s1 = run_abc_dump(ABC_DUMP, p_s1, s_dir / "r1_dump", s_dir / "r1_dump.log")
    rep_struct_v3 = run_adaptive_arm(
        name=slug, orig_aig_path=orig_p, orig_aig=a_s1, cuts_path=cuts_s1,
        work_dir=s_dir / "core", arm="struct_4B_v3", fanout_alpha=0.20,
        scales=(1.5, 1.0, 0.5), max_tau=6.0, lam=0.05, fanout_reg=0.0005, verify_each_round=False,
        pi_fanout_alpha=0.10, allow_slack_area=False,
    )
    p_struct_final, acc_struct_post_bal = _conditional_post_bal(Path(rep_struct_v3["final_aig"]), s_dir, "post_bal")

    # Native baselines
    bal_aig = work_dir / "abc_balance.aig"
    rw_aig = work_dir / "abc_rewrite.aig"
    res2_aig = work_dir / "abc_resyn2.aig"
    _run_abc(orig_p, "balance", bal_aig, work_dir / "abc_balance.log")
    _run_abc(orig_p, "rewrite", rw_aig, work_dir / "abc_rewrite.log")
    _run_abc(
        orig_p,
        "balance; rewrite; refactor; balance; rewrite; rewrite -z; balance; refactor -z; rewrite -z; balance",
        res2_aig,
        work_dir / "abc_resyn2.log",
    )

    methods = {
        "orig": orig_p,
        "abc_balance": bal_aig,
        "abc_rewrite": rw_aig,
        "abc_resyn2": res2_aig,
        "4B_v1": Path(rep_v1["final_aig"]),
        "4B_v3": Path(rep_v3["final_aig"]),
        "bal_4B_v3": p_bal_final,
        "struct_4B_v3": p_struct_final,
    }
    mapped = {m: _eval_aig(orig_p, mp, work_dir / f"map_{m}") for m, mp in methods.items()}
    shutil.rmtree(work_dir / "orig_dump", ignore_errors=True)
    for d in (b_dir, s_dir, work_dir / "4B_v1", work_dir / "4B_v3"):
        for sub in ("r1_dump", "round2_dump", "round3_dump"):
            shutil.rmtree(d / sub, ignore_errors=True)
            shutil.rmtree(d / "core" / sub, ignore_errors=True)

    return {
        "relative_path": rel_path,
        "orig_ands": orig_aig.and_count,
        "meta": {
            "bal_4B_v3": [acc_pre_bal, acc_post_bal],
            "struct_4B_v3": [macro_tag, acc_struct_post_bal],
        },
        "total_runtime_s": time.perf_counter() - t0,
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
    items = [(rel, str(out_dir / "work")) for rel in TARGET_PANEL]

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for res in pool.map(eval_circuit, items):
            rows.append(res)
            mp = res["mapping"]
            print(
                f"{res['relative_path']:34s} | orig={mp['orig']['delay_ps']:8.1f} "
                f"bal={mp['abc_balance']['delay_ps']:8.1f} resyn2={mp['abc_resyn2']['delay_ps']:8.1f} | "
                f"v1={mp['4B_v1']['delay_ps']:8.1f} v3={mp['4B_v3']['delay_ps']:8.1f} "
                f"bal_v3={mp['bal_4B_v3']['delay_ps']:8.1f} struct_v3={mp['struct_4B_v3']['delay_ps']:8.1f} | "
                f"meta={res['meta']} cec_all={all(v['cec']=='PASS' for v in mp.values())}",
                flush=True,
            )

    print("=== Aggregate across 16-circuit target panel ===", flush=True)
    for m in ("abc_balance", "abc_rewrite", "abc_resyn2", "4B_v1", "4B_v3", "bal_4B_v3", "struct_4B_v3"):
        dr = [r["mapping"][m]["delay_ps"] / r["mapping"]["orig"]["delay_ps"] for r in rows]
        ar = [r["mapping"][m]["area"] / r["mapping"]["orig"]["area"] for r in rows]
        w_res = sum(1 for r in rows if r["mapping"][m]["delay_ps"] < r["mapping"]["abc_resyn2"]["delay_ps"] - 1e-6)
        w_orig = sum(1 for x in dr if x < 1.0 - 1e-6)
        print(f"  {m:14s}: delay={_gm(dr):.4f}x ({(_gm(dr)-1)*100:+6.2f}%), area={_gm(ar):.4f}x ({(_gm(ar)-1)*100:+6.2f}%), wins_vs_orig={w_orig}/16, wins_vs_resyn2={w_res}/16", flush=True)

    (out_dir / "summary.json").write_text(json.dumps({
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "circuits": rows,
    }, indent=2, ensure_ascii=False) + "\n")
    print("DONE step5_longchain_panel_v3", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
