#!/usr/bin/env python3
"""Verify GPU rw4b fixes on: (1) 28 previously-crashing circuits, (2) Top-10 regressed circuits, (3) 16-circuit panel."""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from scripts.step5_longchain_panel import DS_ROOT, TARGET_PANEL, _eval_aig
from scripts.step8_gpu_struct_full905 import STEP4_FULL905_SUMMARY, _run_gpuls

PREV_905_SUMMARY = Path(
    "/data/wushangge/CULS_LiteCut/runs/20261125T085355Z-step8-gpu-struct-full905-5fca4fc0/output/summary.json"
)
TOP10_REGRESSIONS = [
    "forgeEDA/srpt_grant_pkts.aig",
    "forgeEDA/pa_f_spsram_512x16.aig",
    "forgeEDA/pa_f_spsram_32x46.aig",
    "forgeEDA/pa_f_spsram_512x4.aig",
    "forgeEDA/pa_f_spsram_32x47.aig",
    "forgeEDA/cache.aig",
    "forgeEDA/cache_datapath.aig",
]


def _eval_one(item: tuple[str, str, str]) -> tuple[str, dict]:
    rel, out_str, w_str = item
    w = Path(w_str)
    try:
        res = _eval_aig(DS_ROOT / rel, Path(out_str), w / "map")
    except Exception as exc:
        res = {"status": "ERROR", "cec": "ERROR", "error": f"{type(exc).__name__}: {exc}"}
    finally:
        shutil.rmtree(w, ignore_errors=True)
    return rel, res


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

    prev905 = json.loads(PREV_905_SUMMARY.read_text())
    crash_list = [f["relative_path"] for f in prev905["failures"]]
    base_map = {c["relative_path"]: c.get("mapping", {}) for c in json.loads(STEP4_FULL905_SUMMARY.read_text())["circuits"]}

    # 1. Test all 28 previously-crashing circuits
    print(f"=== 1. Testing {len(crash_list)} previously-crashing circuits ===", flush=True)
    c_ok = 0
    c_fail = []
    for rel in crash_list:
        slug = rel.replace("/", "__").removesuffix(".aig")
        cdir = work / f"crash_{slug}"
        cdir.mkdir(parents=True, exist_ok=True)
        out_aig = cdir / "out.aig"
        gi = _run_gpuls(gpuls_bin, DS_ROOT / rel, out_aig, "-m -r 3 -v 1", timeout_s=120)
        if gi["status"] == "OK":
            c_ok += 1
        else:
            c_fail.append((rel, gi["status"], gi.get("error_tail", "")[-120:]))
        shutil.rmtree(cdir, ignore_errors=True)
    print(f"Crash set result: {c_ok}/{len(crash_list)} OK!", flush=True)
    for rel, st, tail in c_fail:
        print(f"  STILL FAIL: {rel} | {st} | {tail}", flush=True)

    # 2. Test Top-7 regression circuits + 16-circuit panel (deduplicated)
    eval_list = list(dict.fromkeys(list(TOP10_REGRESSIONS) + list(TARGET_PANEL)))
    items = []
    ginfo_map = {}
    for rel in eval_list:
        slug = rel.replace("/", "__").removesuffix(".aig")
        cdir = work / f"eval_{slug}"
        cdir.mkdir(parents=True, exist_ok=True)
        out_aig = cdir / "out.aig"
        gi = _run_gpuls(gpuls_bin, DS_ROOT / rel, out_aig, "-m -r 3 -v 1", timeout_s=120)
        ginfo_map[rel] = gi
        if gi["status"] == "OK":
            items.append((rel, str(out_aig), str(cdir)))

    ev_map = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for rel, res in pool.map(_eval_one, items):
            ev_map[rel] = res

    print("\n=== 2. Top-7 Previously-Regressed SRAM/Cache Circuits ===", flush=True)
    for rel in TOP10_REGRESSIONS:
        od = base_map[rel]["orig"]["delay_ps"]
        cd = base_map[rel]["4B_fo020_ann_i3"]["delay_ps"]
        gd = ev_map[rel]["delay_ps"]
        cec = ev_map[rel]["cec"]
        pm = ginfo_map[rel].get("pre_macro", "?")
        print(f"  {rel:34s} | orig={od:8.1f} cpu4b={cd:8.1f} NEW_gpu_str={gd:8.1f} ({gd/od:5.2f}x) | pre={pm} cec={cec}", flush=True)

    print("\n=== 3. 16-Circuit Target Panel ===", flush=True)
    dr, ar = [], []
    w_orig = w_res = cec_ok = 0
    for rel in TARGET_PANEL:
        od = base_map[rel]["orig"]["delay_ps"]
        oa = base_map[rel]["orig"]["area"]
        rd = base_map[rel]["abc_resyn2"]["delay_ps"]
        gd = ev_map[rel]["delay_ps"]
        ga = ev_map[rel]["area"]
        cec = ev_map[rel]["cec"]
        pm = ginfo_map[rel].get("pre_macro", "?")
        dr.append(gd / od)
        ar.append(ga / oa)
        if gd < od - 1e-6: w_orig += 1
        if gd < rd - 1e-6: w_res += 1
        if cec == "PASS": cec_ok += 1
        print(f"  {rel:34s} | orig={od:8.1f} resyn2={rd:8.1f} NEW_gpu_str={gd:8.1f} | pre={pm:7s} cec={cec}", flush=True)
    print(
        f"  NEW gpu_struct_rw4b (16-panel): delay={_gm(dr):.4f}x ({(_gm(dr)-1)*100:+6.2f}%), "
        f"area={_gm(ar):.4f}x ({(_gm(ar)-1)*100:+6.2f}%), wins_orig={w_orig}/16, wins_resyn2={w_res}/16, cec={cec_ok}/16",
        flush=True,
    )
    print("DONE step9_verify_gpu_fixes", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
