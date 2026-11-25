#!/usr/bin/env python3
"""Evaluate targeted 16-circuit panel for long-chain + refactor + PI-fanout upgrades."""
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
    return aig, dp, max_fo


def _conditional_abc_step(
    cur_p: Path,
    cur_aig,
    cur_dp,
    cur_max_fo: int,
    cmd: str,
    mode: str,
    work_dir: Path,
    tag: str,
) -> tuple[Path, object, object, int, bool]:
    """Apply `cmd` only if AIG-level fanout-aware delay improves safely without fanout spikes."""
    cand_p = work_dir / f"{tag}.aig"
    _run_abc(cur_p, cmd, cand_p, work_dir / f"{tag}.log")
    cand_aig, cand_dp, cand_max_fo = _aig_stats(cand_p)
    fo_ok = cand_max_fo <= max(64, cur_max_fo)

    if mode == "balance":
        accept = fo_ok and (cand_dp.depth <= cur_dp.depth - 1.0)
    elif mode == "rfz":
        accept = fo_ok and (
            cand_dp.depth < cur_dp.depth - 1e-3
            or (cand_dp.depth <= cur_dp.depth + 1e-6 and cand_aig.and_count < cur_aig.and_count)
        )
    else:
        raise ValueError(mode)

    if accept:
        return cand_p, cand_aig, cand_dp, cand_max_fo, True
    return cur_p, cur_aig, cur_dp, cur_max_fo, False


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
    orig_aig, orig_dp, orig_max_fo = _aig_stats(orig_p)
    r1_cuts = run_abc_dump(ABC_DUMP, orig_p, work_dir / "orig_dump", work_dir / "orig_dump.log")

    # 1. Baseline 4B_fo020_ann_i3 (v1)
    rep_v1 = run_adaptive_arm(
        name=slug, orig_aig_path=orig_p, orig_aig=orig_aig, cuts_path=r1_cuts,
        work_dir=work_dir / "4B_v1", arm="4B_v1", fanout_alpha=0.20,
        scales=(1.5, 1.0, 0.5), max_tau=6.0, lam=0.05, fanout_reg=0.0005, verify_each_round=False,
    )

    # 2. Pure 4B_v2 (adds pi_fanout_alpha=0.10 + allow_slack_area=True)
    rep_v2 = run_adaptive_arm(
        name=slug, orig_aig_path=orig_p, orig_aig=orig_aig, cuts_path=r1_cuts,
        work_dir=work_dir / "4B_v2", arm="4B_v2", fanout_alpha=0.20,
        scales=(1.5, 1.0, 0.5), max_tau=6.0, lam=0.05, fanout_reg=0.0005, verify_each_round=False,
        pi_fanout_alpha=0.10, allow_slack_area=True,
    )

    # 3. rfz_4B_v2 (conditional refactor -z -> 4B_v2 -> conditional refactor -z)
    rfz_dir = work_dir / "rfz_4B_v2"
    rfz_dir.mkdir(parents=True, exist_ok=True)
    p_rf1, a_rf1, dp_rf1, fo_rf1, acc_rf1 = _conditional_abc_step(
        orig_p, orig_aig, orig_dp, orig_max_fo, "refactor -z", "rfz", rfz_dir, "pre_rfz"
    )
    cuts_rf1 = run_abc_dump(ABC_DUMP, p_rf1, rfz_dir / "r1_dump", rfz_dir / "r1_dump.log") if acc_rf1 else r1_cuts
    rep_rfz_v2 = run_adaptive_arm(
        name=slug, orig_aig_path=orig_p, orig_aig=a_rf1, cuts_path=cuts_rf1,
        work_dir=rfz_dir / "core", arm="rfz_4B_v2", fanout_alpha=0.20,
        scales=(1.5, 1.0, 0.5), max_tau=6.0, lam=0.05, fanout_reg=0.0005, verify_each_round=False,
        pi_fanout_alpha=0.10, allow_slack_area=True,
    )
    cur_p = Path(rep_rfz_v2["final_aig"])
    cur_a, cur_dp, cur_fo = _aig_stats(cur_p)
    p_rfz_final, _, _, _, acc_rf2 = _conditional_abc_step(
        cur_p, cur_a, cur_dp, cur_fo, "refactor -z", "rfz", rfz_dir, "post_rfz"
    )

    # 4. bal_rfz_4B_v2 (conditional balance -> conditional rfz -> 4B_v2 -> conditional balance -> conditional rfz)
    br_dir = work_dir / "bal_rfz_4B_v2"
    br_dir.mkdir(parents=True, exist_ok=True)
    p_b1, a_b1, dp_b1, fo_b1, acc_b1 = _conditional_abc_step(
        orig_p, orig_aig, orig_dp, orig_max_fo, "balance", "balance", br_dir, "pre_bal"
    )
    p_br1, a_br1, dp_br1, fo_br1, acc_br1 = _conditional_abc_step(
        p_b1, a_b1, dp_b1, fo_b1, "refactor -z", "rfz", br_dir, "pre_rfz"
    )
    cuts_br1 = run_abc_dump(ABC_DUMP, p_br1, br_dir / "r1_dump", br_dir / "r1_dump.log") if (acc_b1 or acc_br1) else r1_cuts
    rep_br_v2 = run_adaptive_arm(
        name=slug, orig_aig_path=orig_p, orig_aig=a_br1, cuts_path=cuts_br1,
        work_dir=br_dir / "core", arm="bal_rfz_4B_v2", fanout_alpha=0.20,
        scales=(1.5, 1.0, 0.5), max_tau=6.0, lam=0.05, fanout_reg=0.0005, verify_each_round=False,
        pi_fanout_alpha=0.10, allow_slack_area=True,
    )
    p_mid = Path(rep_br_v2["final_aig"])
    a_mid, dp_mid, fo_mid = _aig_stats(p_mid)
    p_b2, a_b2, dp_b2, fo_b2, acc_b2 = _conditional_abc_step(
        p_mid, a_mid, dp_mid, fo_mid, "balance", "balance", br_dir, "post_bal"
    )
    p_br_final, _, _, _, acc_br2 = _conditional_abc_step(
        p_b2, a_b2, dp_b2, fo_b2, "refactor -z", "rfz", br_dir, "post_rfz"
    )

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
        "4B_v2": Path(rep_v2["final_aig"]),
        "rfz_4B_v2": p_rfz_final,
        "bal_rfz_4B_v2": p_br_final,
    }
    mapped = {m: _eval_aig(orig_p, mp, work_dir / f"map_{m}") for m, mp in methods.items()}
    shutil.rmtree(work_dir / "orig_dump", ignore_errors=True)
    for d in (rfz_dir, br_dir, work_dir / "4B_v1", work_dir / "4B_v2"):
        for sub in ("r1_dump", "round2_dump", "round3_dump"):
            shutil.rmtree(d / sub, ignore_errors=True)
            shutil.rmtree(d / "core" / sub, ignore_errors=True)

    return {
        "relative_path": rel_path,
        "orig_ands": orig_aig.and_count,
        "accepted_steps": {
            "rfz_4B_v2": [acc_rf1, acc_rf2],
            "bal_rfz_4B_v2": [acc_b1, acc_br1, acc_b2, acc_br2],
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
                f"v1={mp['4B_v1']['delay_ps']:8.1f} v2={mp['4B_v2']['delay_ps']:8.1f} "
                f"rfz_v2={mp['rfz_4B_v2']['delay_ps']:8.1f} bal_rfz_v2={mp['bal_rfz_4B_v2']['delay_ps']:8.1f} | "
                f"steps={res['accepted_steps']} cec_all={all(v['cec']=='PASS' for v in mp.values())}",
                flush=True,
            )

    print("=== Aggregate across 16-circuit target panel ===", flush=True)
    for m in ("abc_balance", "abc_rewrite", "abc_resyn2", "4B_v1", "4B_v2", "rfz_4B_v2", "bal_rfz_4B_v2"):
        dr = [r["mapping"][m]["delay_ps"] / r["mapping"]["orig"]["delay_ps"] for r in rows]
        ar = [r["mapping"][m]["area"] / r["mapping"]["orig"]["area"] for r in rows]
        w_res = sum(1 for r in rows if r["mapping"][m]["delay_ps"] < r["mapping"]["abc_resyn2"]["delay_ps"] - 1e-6)
        w_orig = sum(1 for x in dr if x < 1.0 - 1e-6)
        print(f"  {m:16s}: delay={_gm(dr):.4f}x ({(_gm(dr)-1)*100:+6.2f}%), area={_gm(ar):.4f}x ({(_gm(ar)-1)*100:+6.2f}%), wins_vs_orig={w_orig}/16, wins_vs_resyn2={w_res}/16", flush=True)

    (out_dir / "summary.json").write_text(json.dumps({
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "circuits": rows,
    }, indent=2, ensure_ascii=False) + "\n")
    print("DONE step5_longchain_panel", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
