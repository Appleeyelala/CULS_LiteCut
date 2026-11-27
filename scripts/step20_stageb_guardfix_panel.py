#!/usr/bin/env python3
"""Fast 16-circuit verification of batch-net-fanout guard fix:
Compares fixed Stage B (fo-inc-lam=0.0 vs 0.15) against cached 66425 baselines (abc_ifg, abc_strong_delay, Stage A, Stage B 66415).
"""
from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from scripts.step5_longchain_panel import DS_ROOT, TARGET_PANEL
from scripts.step8_gpu_struct_full905 import STEP4_FULL905_SUMMARY
from scripts.step19_stageb_foinc_panel import _eval_aig_task, _run_gpuls_diag

CACHED_66425 = Path(
    "/data/wushangge/CULS_LiteCut/runs/20261127T060550Z-step24-stageb-foinc-panel-0f815ef3/output/panel_summary.json"
)


def _gm(xs: list[float]) -> float:
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--gpuls", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    run = args.run.resolve()
    gpuls_bin = args.gpuls.resolve()
    work = run / "output/work"
    work.mkdir(parents=True, exist_ok=True)

    base_map = {c["relative_path"]: c.get("mapping", {}) for c in json.loads(STEP4_FULL905_SUMMARY.read_text())["circuits"]}
    cached = json.loads(CACHED_66425.read_text())
    cached_eval = cached["eval"]
    cached_sum = cached["summary"]

    configs = [
        ("stageB_fix_fo00", "-r 3 --joint-grad --steps 25 --lr 0.20 --disc-lam 0.30 --fo-inc-lam 0.0 -v 1"),
        ("stageB_fix_fo15", "-r 3 --joint-grad --steps 25 --lr 0.20 --disc-lam 0.30 --fo-inc-lam 0.15 --fo-inc-th 6.0 -v 1"),
    ]

    eval_items = []
    ginfo = {}
    for tag, flags in configs:
        for rel in TARGET_PANEL:
            slug = rel.replace("/", "__").removesuffix(".aig")
            cdir = work / f"{tag}_{slug}"
            cdir.mkdir(parents=True, exist_ok=True)
            out_aig = cdir / "out.aig"
            gi = _run_gpuls_diag(gpuls_bin, DS_ROOT / rel, out_aig, flags, timeout_s=180)
            ginfo[(tag, rel)] = gi
            if gi["status"] == "OK":
                eval_items.append((tag, rel, str(out_aig), str(cdir)))
            else:
                print(f"  WARN {tag} {rel}: {gi}", flush=True)

    ev = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for tag, rel, res in pool.map(_eval_aig_task, eval_items):
            ev[(tag, rel)] = res

    print("=== 1. PER-CIRCUIT 16-CIRCUIT PANEL (ASAP7 delay_ps) ===", flush=True)
    print(f"  {'Circuit':34s} | {'orig':>8s} | {'abc_strg':>8s} | {'StageA':>8s} | {'B_66415':>8s} | {'B_fix00':>8s} | {'B_fix15':>8s}", flush=True)
    for rel in TARGET_PANEL:
        od = base_map[rel]["orig"]["delay_ps"]
        s_strg = cached_eval.get(f"abc_strong_delay::{rel}", {}).get("delay_ps", -1)
        s_a = cached_eval.get(f"stageA::{rel}", {}).get("delay_ps", -1)
        s_old = cached_eval.get(f"stageB_old::{rel}", {}).get("delay_ps", -1)
        r0 = ev.get(("stageB_fix_fo00", rel), {})
        r15 = ev.get(("stageB_fix_fo15", rel), {})
        d0 = f"{r0.get('delay_ps', -1):8.1f}" if r0.get("cec") == "PASS" else f"{'ERR':>8s}"
        d15 = f"{r15.get('delay_ps', -1):8.1f}" if r15.get("cec") == "PASS" else f"{'ERR':>8s}"
        print(f"  {rel:34s} | {od:8.1f} | {s_strg:8.1f} | {s_a:8.1f} | {s_old:8.1f} | {d0} | {d15}", flush=True)

    print("\n=== 2. AGGREGATE 16-CIRCUIT PANEL SUMMARY ===", flush=True)
    for cname in ("abc_ifg", "abc_strong_delay", "stageA", "stageB_old"):
        cs = cached_sum[cname]
        print(
            f"  {cname:18s} | delay={cs['delay_gm']:.4f}x ({(cs['delay_gm']-1)*100:+6.2f}%) | "
            f"area={cs['area_gm']:.4f}x ({(cs['area_gm']-1)*100:+6.2f}%) | ADP={cs['adp_gm']:.4f}x ({(cs['adp_gm']-1)*100:+6.2f}%) | "
            f"w_orig={cs['wins_orig']:2d}/16 cec={cs['cec_ok']:2d}/16 | time_ms={cs['core_ms']:8.1f} (cached 66425)",
            flush=True,
        )
    for tag, _ in configs:
        dr, ar, adpr = [], [], []
        w_orig, cec_ok = 0, 0
        for rel in TARGET_PANEL:
            r = ev.get((tag, rel), {})
            if r.get("cec") == "PASS":
                cec_ok += 1
                d_ratio = r["delay_ps"] / base_map[rel]["orig"]["delay_ps"]
                a_ratio = r["area"] / base_map[rel]["orig"]["area"]
                dr.append(d_ratio); ar.append(a_ratio); adpr.append(d_ratio * a_ratio)
                if d_ratio < 1.0 - 1e-6:
                    w_orig += 1
        tot_ms = sum(ginfo[(tag, rel)].get("rw4b_total_s", 0.0) * 1000.0 for rel in TARGET_PANEL)
        print(
            f"  {tag:18s} | delay={_gm(dr):.4f}x ({(_gm(dr)-1)*100:+6.2f}%) | "
            f"area={_gm(ar):.4f}x ({(_gm(ar)-1)*100:+6.2f}%) | ADP={_gm(adpr):.4f}x ({(_gm(adpr)-1)*100:+6.2f}%) | "
            f"w_orig={w_orig:2d}/16 cec={cec_ok:2d}/16 | time_ms={tot_ms:8.1f}",
            flush=True,
        )

    print("\n=== 3. DIAGNOSTIC A: GUARD REJECTION RATE ===", flush=True)
    for tag, _ in configs:
        tot_raw = sum(sum(x["raw"] for x in ginfo[(tag, rel)].get("rounds", [])) for rel in TARGET_PANEL)
        tot_app = sum(sum(x["applied"] for x in ginfo[(tag, rel)].get("rounds", [])) for rel in TARGET_PANEL)
        tot_blk = tot_raw - tot_app
        res_rds = ginfo[(tag, "forgeEDA/reservation_station.aig")].get("rounds", [])
        res_raw = sum(x["raw"] for x in res_rds)
        res_app = sum(x["applied"] for x in res_rds)
        res_blk = res_raw - res_app
        print(
            f"  {tag:16s} | ALL 16: raw={tot_raw:6d} app={tot_app:6d} blk={tot_blk:5d} ({100.0*tot_blk/max(1,tot_raw):5.2f}%) | "
            f"reservation_station: raw={res_raw:5d} app={res_app:5d} blk={res_blk:4d} ({100.0*res_blk/max(1,res_raw):5.2f}%)",
            flush=True,
        )

    print("\n=== 4. DIAGNOSTIC B: PER-ROUND LOGS FOR sirv_sim_ram & branch_prediction_globle ===", flush=True)
    for rel in ("forgeEDA/sirv_sim_ram.aig", "forgeEDA/branch_prediction_globle.aig", "forgeEDA/reservation_station.aig"):
        print(f"  --- {rel} ---", flush=True)
        for tag, _ in configs:
            r_eval = ev.get((tag, rel), {})
            print(f"    [{tag}] ASAP7 delay={r_eval.get('delay_ps', -1):.1f} ps, area={r_eval.get('area', -1):.1f}", flush=True)
            for rd in ginfo[(tag, rel)].get("rounds", []):
                blk = rd["raw"] - rd["applied"]
                print(
                    f"      R{rd['round']}: M={rd['M']:5d} raw={rd['raw']:5d} applied={rd['applied']:5d} blocked={blk:4d} "
                    f"theta={rd['theta']:.2f} avgMaxP={rd['avgMaxP']:.3f} discD={rd['discD']:.2f}(D0={rd['D0']:.2f}) dAND={rd['dAND']:+d}",
                    flush=True,
                )

    print("DONE step20_stageb_guardfix_panel", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
