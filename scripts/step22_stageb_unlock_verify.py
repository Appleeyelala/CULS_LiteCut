#!/usr/bin/env python3
"""Verify Move 1 (z0=1.2 + late Gini magnet) & Move 2 (GWMIN slimming for off-critical nodes)
on the 3 top under-applied 905 circuits (FFTStoreBit, aes_xcrypt_orig, Memory) + 16-circuit target panel.
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

SPOT_905 = [
    "forgeEDA/FFTStoreBit.aig",
    "OpenABC/aes_xcrypt_orig.aig",
    "forgeEDA/Memory.aig",
]

CACHED_66425 = Path(
    "/data/wushangge/CULS_LiteCut/runs/20261127T060550Z-step24-stageb-foinc-panel-0f815ef3/output/panel_summary.json"
)
STAGEA_905_JSONL = Path(
    "/data/wushangge/CULS_LiteCut/runs/20261126T080537Z-step17-claude-final-full905-270d8735/cap0/output/results.jsonl"
)
STAGEB_905_OLD_JSONL = Path(
    "/data/wushangge/CULS_LiteCut/runs/20261127T064134Z-step26-stageb-full905-e101acb0/output/results.jsonl"
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
    stageA_905 = {json.loads(l)["relative_path"]: json.loads(l) for l in STAGEA_905_JSONL.open()}
    stageB_old = {json.loads(l)["relative_path"]: json.loads(l) for l in STAGEB_905_OLD_JSONL.open()}

    all_circuits = list(dict.fromkeys(SPOT_905 + list(TARGET_PANEL)))
    flags = "-r 3 --joint-grad --steps 25 --lr 0.20 --disc-lam 0.30 --fo-inc-lam 0.0 -v 1"

    eval_items = []
    ginfo = {}
    for rel in all_circuits:
        slug = rel.replace("/", "__").removesuffix(".aig")
        cdir = work / f"b_v2_{slug}"
        cdir.mkdir(parents=True, exist_ok=True)
        out_aig = cdir / "out.aig"
        gi = _run_gpuls_diag(gpuls_bin, DS_ROOT / rel, out_aig, flags, timeout_s=180)
        ginfo[rel] = gi
        if gi["status"] == "OK":
            eval_items.append(("b_v2", rel, str(out_aig), str(cdir)))
        else:
            print(f"  WARN {rel}: {gi}", flush=True)

    ev = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for tag, rel, res in pool.map(_eval_aig_task, eval_items):
            ev[rel] = res

    print("=== 1. TOP-3 UNDER-APPLIED 905 SPOT CIRCUITS ===", flush=True)
    for rel in SPOT_905:
        od = base_map[rel]["orig"]["delay_ps"]
        da = stageA_905[rel]["gpu_struct_rw4b"]["delay_ps"]
        aa = stageA_905[rel]["gpu_info"]["applied"]
        db_old = stageB_old[rel]["gpu_struct_rw4b"]["delay_ps"]
        ab_old = stageB_old[rel]["gpu_info"]["applied"]
        r_new = ev.get(rel, {})
        g_new = ginfo[rel]
        print(
            f"  {rel:28s} | orig={od:9.1f} | StageA={da:8.1f}(app={aa:5d}) | "
            f"B_old={db_old:9.1f}(app={ab_old:4d}) | B_v2={r_new.get('delay_ps',-1):8.1f}"
            f"(app={g_new.get('applied',0):5d}, cec={r_new.get('cec','ERR')}, {g_new.get('rw4b_total_s',0)*1000:5.0f}ms)",
            flush=True,
        )
        for rd in g_new.get("rounds", []):
            print(
                f"      R{rd['round']}: M_crit={rd['M']:4d} app={rd['applied']:5d}(crit={rd['crit']:4d}/raw={rd['raw']:4d}, nc={rd['nc']:5d}) "
                f"z0={rd['z0']:.1f} theta={rd['theta']:.2f} avgMaxP={rd['avgMaxP']:.3f} loss={rd['initLoss']:.2f}->{rd['finalLoss']:.2f} "
                f"discD={rd['discD']:.1f}(D0={rd['D0']:.1f}) grad={rd['grad_ms']:.1f}ms",
                flush=True,
            )

    print("\n=== 2. 16-CIRCUIT TARGET PANEL (ASAP7 delay_ps) ===", flush=True)
    print(f"  {'Circuit':34s} | {'orig':>8s} | {'StageA':>8s} | {'B_66415':>8s} | {'B_v3':>8s} | {'app':>6s} | z0(initLoss) per round", flush=True)
    for rel in TARGET_PANEL:
        od = base_map[rel]["orig"]["delay_ps"]
        s_a = cached_eval.get(f"stageA::{rel}", {}).get("delay_ps", -1)
        s_old = cached_eval.get(f"stageB_old::{rel}", {}).get("delay_ps", -1)
        rn = ev.get(rel, {})
        dn = f"{rn.get('delay_ps', -1):8.1f}" if rn.get("cec") == "PASS" else f"{'ERR':>8s}"
        z_str = " ".join(f"R{rd['round']}:{rd['z0']:.1f}({rd['initLoss']:.2f})" for rd in ginfo[rel].get("rounds", []))
        print(f"  {rel:34s} | {od:8.1f} | {s_a:8.1f} | {s_old:8.1f} | {dn} | {ginfo[rel].get('applied',0):6d} | {z_str}", flush=True)

    print("\n=== 3. AGGREGATE 16-CIRCUIT PANEL SUMMARY ===", flush=True)
    for cname in ("abc_strong_delay", "stageA", "stageB_old"):
        cs = cached_sum[cname]
        print(
            f"  {cname:18s} | delay={cs['delay_gm']:.4f}x ({(cs['delay_gm']-1)*100:+6.2f}%) | "
            f"area={cs['area_gm']:.4f}x ({(cs['area_gm']-1)*100:+6.2f}%) | ADP={cs['adp_gm']:.4f}x ({(cs['adp_gm']-1)*100:+6.2f}%) | "
            f"w_orig={cs['wins_orig']:2d}/16 cec={cs['cec_ok']:2d}/16 | time_ms={cs['core_ms']:8.1f}",
            flush=True,
        )
    dr, ar, adpr = [], [], []
    w_orig, cec_ok = 0, 0
    for rel in TARGET_PANEL:
        r = ev.get(rel, {})
        if r.get("cec") == "PASS":
            cec_ok += 1
            d_ratio = r["delay_ps"] / base_map[rel]["orig"]["delay_ps"]
            a_ratio = r["area"] / base_map[rel]["orig"]["area"]
            dr.append(d_ratio); ar.append(a_ratio); adpr.append(d_ratio * a_ratio)
            if d_ratio < 1.0 - 1e-6:
                w_orig += 1
    tot_ms = sum(ginfo[rel].get("rw4b_total_s", 0.0) * 1000.0 for rel in TARGET_PANEL)
    print(
        f"  {'stageB_v2':18s} | delay={_gm(dr):.4f}x ({(_gm(dr)-1)*100:+6.2f}%) | "
        f"area={_gm(ar):.4f}x ({(_gm(ar)-1)*100:+6.2f}%) | ADP={_gm(adpr):.4f}x ({(_gm(adpr)-1)*100:+6.2f}%) | "
        f"w_orig={w_orig:2d}/16 cec={cec_ok:2d}/16 | time_ms={tot_ms:8.1f}",
        flush=True,
    )
    print("DONE step22_stageb_unlock_verify", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
