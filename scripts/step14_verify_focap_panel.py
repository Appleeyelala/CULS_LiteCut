#!/usr/bin/env python3
"""Verify Claude Step 1 pure bug fixes (P0-1, P0-4, P1-1..P1-4) comparing --fo-cap 0 vs --fo-cap 16 on 4 spot + 16 panel circuits."""
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

    all_circuits = list(dict.fromkeys(SPOT_CIRCUITS + list(TARGET_PANEL)))
    eval_items = []
    ginfo = {}

    for cap in (0, 16):
        tag = f"cap{cap}"
        for rel in all_circuits:
            slug = rel.replace("/", "__").removesuffix(".aig")
            cdir = work / f"{tag}_{slug}"
            cdir.mkdir(parents=True, exist_ok=True)
            out_aig = cdir / "out.aig"
            gi = _run_gpuls(gpuls_bin, DS_ROOT / rel, out_aig, f"-r 3 --fo-cap {cap} -v 1", timeout_s=120)
            ginfo[(tag, rel)] = gi
            if gi["status"] == "OK":
                eval_items.append((tag, rel, str(out_aig), str(cdir)))

    ev = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for tag, rel, res in pool.map(_eval_one, eval_items):
            ev[(tag, rel)] = res

    print("=== 1. SPOT CHECK: --fo-cap 0 vs --fo-cap 16 (after P0-1/P0-4/P1-1..P1-4 fixes) ===", flush=True)
    for rel in SPOT_CIRCUITS:
        od = base_map[rel]["orig"]["delay_ps"]
        cd = base_map[rel]["4B_fo020_ann_i3"]["delay_ps"]
        r0 = ev[("cap0", rel)]; g0 = ginfo[("cap0", rel)]
        r16 = ev[("cap16", rel)]; g16 = ginfo[("cap16", rel)]
        print(
            f"  {rel:28s} | orig={od:9.1f} cpu4b={cd:8.1f} | "
            f"cap0={r0['delay_ps']:8.1f}(app={g0.get('applied',0):5d},cec={r0['cec']}) | "
            f"cap16={r16['delay_ps']:8.1f}(app={g16.get('applied',0):5d},cec={r16['cec']})",
            flush=True,
        )

    print("\n=== 2. 16-CIRCUIT TARGET PANEL: --fo-cap 0 vs --fo-cap 16 ===", flush=True)
    for rel in TARGET_PANEL:
        od = base_map[rel]["orig"]["delay_ps"]
        rd = base_map[rel]["abc_resyn2"]["delay_ps"]
        r0 = ev[("cap0", rel)]; r16 = ev[("cap16", rel)]
        print(
            f"  {rel:34s} | orig={od:8.1f} resyn2={rd:8.1f} | "
            f"cap0={r0['delay_ps']:8.1f} ({r0['cec']}) | cap16={r16['delay_ps']:8.1f} ({r16['cec']})",
            flush=True,
        )

    print("\n=== 3. AGGREGATE ON 16-CIRCUIT PANEL ===", flush=True)
    for cap in (0, 16):
        tag = f"cap{cap}"
        dr = [ev[(tag, rel)]["delay_ps"] / base_map[rel]["orig"]["delay_ps"] for rel in TARGET_PANEL]
        ar = [ev[(tag, rel)]["area"] / base_map[rel]["orig"]["area"] for rel in TARGET_PANEL]
        adpr = [d * a for d, a in zip(dr, ar)]
        w_orig = sum(1 for x in dr if x < 1.0 - 1e-6)
        w_res = sum(1 for rel in TARGET_PANEL if ev[(tag, rel)]["delay_ps"] < base_map[rel]["abc_resyn2"]["delay_ps"] - 1e-6)
        cec_ok = sum(1 for rel in TARGET_PANEL if ev[(tag, rel)]["cec"] == "PASS")
        print(
            f"  gpu_rw4b --fo-cap {cap:2d}: delay={_gm(dr):.4f}x ({(_gm(dr)-1)*100:+6.2f}%), "
            f"area={_gm(ar):.4f}x ({(_gm(ar)-1)*100:+6.2f}%), ADP={_gm(adpr):.4f}x ({(_gm(adpr)-1)*100:+6.2f}%), "
            f"wins_orig={w_orig}/16, wins_resyn2={w_res}/16, cec={cec_ok}/16",
            flush=True,
        )
    print("DONE step14_verify_focap_panel", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
