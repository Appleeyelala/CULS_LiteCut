#!/usr/bin/env python3
"""Quick verification of Fanout-Capped Strash + coneIds slack_other on FFTStoreBit, y_quantizer, generic_mem, and 16-circuit panel."""
from __future__ import annotations

import argparse
import json
import math
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from scripts.step5_longchain_panel import DS_ROOT, TARGET_PANEL, _eval_aig
from scripts.step8_gpu_struct_full905 import STEP4_FULL905_SUMMARY, _run_gpuls

SPOT_CIRCUITS = [
    "forgeEDA/y_quantizer.aig",
    "forgeEDA/FFTStoreBit.aig",
    "forgeEDA/generic_mem.aig",
    "forgeEDA/LZ77_Encoder.aig",
]


def _eval_one(item: tuple[str, str, str, str]) -> tuple[str, str, dict]:
    tag, rel, out_str, w_str = item
    w = Path(w_str)
    try:
        res = _eval_aig(DS_ROOT / rel, Path(out_str), w / "map")
    except Exception as exc:
        res = {"status": "ERROR", "cec": "ERROR", "error": f"{type(exc).__name__}: {exc}"}
    finally:
        shutil.rmtree(w, ignore_errors=True)
    return tag, rel, res


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

    eval_items = []
    ginfo = {}

    # 1. Spot check fo_cap=8 vs fo_cap=16 on SPOT_CIRCUITS
    for cap in (8, 16):
        tag = f"cap{cap}"
        for rel in SPOT_CIRCUITS:
            slug = rel.replace("/", "__").removesuffix(".aig")
            cdir = work / f"{tag}_{slug}"
            cdir.mkdir(parents=True, exist_ok=True)
            out_aig = cdir / "out.aig"
            gi = _run_gpuls(gpuls_bin, DS_ROOT / rel, out_aig, f"-r 3 --fo-cap {cap} -v 1", timeout_s=120)
            ginfo[(tag, rel)] = gi
            if gi["status"] == "OK":
                eval_items.append((tag, rel, str(out_aig), str(cdir)))

    # 2. Full 16-circuit target panel with default fo_cap=16
    for rel in TARGET_PANEL:
        slug = rel.replace("/", "__").removesuffix(".aig")
        cdir = work / f"p16_{slug}"
        cdir.mkdir(parents=True, exist_ok=True)
        out_aig = cdir / "out.aig"
        gi = _run_gpuls(gpuls_bin, DS_ROOT / rel, out_aig, "-r 3 --fo-cap 16 -v 1", timeout_s=120)
        ginfo[("p16", rel)] = gi
        if gi["status"] == "OK":
            eval_items.append(("p16", rel, str(out_aig), str(cdir)))

    ev = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for tag, rel, res in pool.map(_eval_one, eval_items):
            ev[(tag, rel)] = res

    print("=== 1. SPOT CHECK: fo_cap=8 vs fo_cap=16 ===", flush=True)
    for rel in SPOT_CIRCUITS:
        od = base_map[rel]["orig"]["delay_ps"]
        cd = base_map[rel]["4B_fo020_ann_i3"]["delay_ps"]
        r8 = ev[("cap8", rel)]; g8 = ginfo[("cap8", rel)]
        r16 = ev[("cap16", rel)]; g16 = ginfo[("cap16", rel)]
        print(
            f"  {rel:28s} | orig={od:9.1f} cpu4b={cd:8.1f} | "
            f"cap8={r8['delay_ps']:8.1f}(app={g8.get('applied',0):5d},cec={r8['cec']}) | "
            f"cap16={r16['delay_ps']:8.1f}(app={g16.get('applied',0):5d},cec={r16['cec']})",
            flush=True,
        )

    print("\n=== 2. 16-CIRCUIT TARGET PANEL (pure gpu_rw4b -r 3 --fo-cap 16) ===", flush=True)
    dr, ar = [], []
    w_orig = w_res = cec_ok = 0
    for rel in TARGET_PANEL:
        od = base_map[rel]["orig"]["delay_ps"]
        oa = base_map[rel]["orig"]["area"]
        rd = base_map[rel]["abc_resyn2"]["delay_ps"]
        r = ev[("p16", rel)]; gi = ginfo[("p16", rel)]
        dr.append(r["delay_ps"] / od)
        ar.append(r["area"] / oa)
        if r["delay_ps"] < od - 1e-6: w_orig += 1
        if r["delay_ps"] < rd - 1e-6: w_res += 1
        if r["cec"] == "PASS": cec_ok += 1
        print(
            f"  {rel:34s} | orig={od:8.1f} resyn2={rd:8.1f} gpu_v4={r['delay_ps']:8.1f} "
            f"(app={gi.get('applied',0):5d}, {gi.get('rw4b_total_s',0)*1000:5.1f}ms) cec={r['cec']}",
            flush=True,
        )
    print(
        f"  AGG 16-panel pure gpu_rw4b: delay={_gm(dr):.4f}x ({(_gm(dr)-1)*100:+6.2f}%), "
        f"area={_gm(ar):.4f}x ({(_gm(ar)-1)*100:+6.2f}%), wins_orig={w_orig}/16, wins_resyn2={w_res}/16, cec={cec_ok}/16",
        flush=True,
    )
    print("DONE step14_verify_focap_panel", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
