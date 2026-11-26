#!/usr/bin/env python3
"""Sweep Claude Step 2 (P0-2 foCap gating + P0-3 scale-invariant ADP scoring) (w, kappa) on 16-circuit panel."""
from __future__ import annotations

import argparse
import json
import math
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from scripts.step5_longchain_panel import DS_ROOT, TARGET_PANEL, _eval_aig
from scripts.step8_gpu_struct_full905 import STEP4_FULL905_SUMMARY, _run_gpuls

CONFIGS = [
    ("legacy_c16",    "-r 3 --no-adp --fo-cap 16 -v 1"),
    ("adp_w1.0_k1.0", "-r 3 --adp-w 1.0 --kappa 1.0 --fo-cap 16 -v 1"),
    ("adp_w1.5_k1.0", "-r 3 --adp-w 1.5 --kappa 1.0 --fo-cap 16 -v 1"),
    ("adp_w2.0_k1.0", "-r 3 --adp-w 2.0 --kappa 1.0 --fo-cap 16 -v 1"),
    ("adp_w1.5_k2.0", "-r 3 --adp-w 1.5 --kappa 2.0 --fo-cap 16 -v 1"),
    ("adp_w2.0_k2.0", "-r 3 --adp-w 2.0 --kappa 2.0 --fo-cap 16 -v 1"),
    ("adp_w3.0_k2.0", "-r 3 --adp-w 3.0 --kappa 2.0 --fo-cap 16 -v 1"),
]

EXTRA_SPOT = ["forgeEDA/FFTStoreBit.aig", "forgeEDA/LZ77_Encoder.aig"]


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

    all_circuits = list(dict.fromkeys(list(TARGET_PANEL) + EXTRA_SPOT))
    eval_items = []
    ginfo = {}

    for tag, flags in CONFIGS:
        for rel in all_circuits:
            slug = rel.replace("/", "__").removesuffix(".aig")
            cdir = work / f"{tag}_{slug}"
            cdir.mkdir(parents=True, exist_ok=True)
            out_aig = cdir / "out.aig"
            gi = _run_gpuls(gpuls_bin, DS_ROOT / rel, out_aig, flags, timeout_s=120)
            ginfo[(tag, rel)] = gi
            if gi["status"] == "OK":
                eval_items.append((tag, rel, str(out_aig), str(cdir)))

    ev = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for tag, rel, res in pool.map(_eval_one, eval_items):
            ev[(tag, rel)] = res

    print("=== 16-CIRCUIT TARGET PANEL SWEEP (P0-2 + P0-3 ADP SCORING) ===", flush=True)
    summary = {}
    for tag, flags in CONFIGS:
        dr = [ev[(tag, rel)]["delay_ps"] / base_map[rel]["orig"]["delay_ps"] for rel in TARGET_PANEL]
        ar = [ev[(tag, rel)]["area"] / base_map[rel]["orig"]["area"] for rel in TARGET_PANEL]
        adpr = [d * a for d, a in zip(dr, ar)]
        w_orig = sum(1 for x in dr if x < 1.0 - 1e-6)
        w_res = sum(1 for rel in TARGET_PANEL if ev[(tag, rel)]["delay_ps"] < base_map[rel]["abc_resyn2"]["delay_ps"] - 1e-6)
        cec_ok = sum(1 for rel in all_circuits if ev[(tag, rel)]["cec"] == "PASS")
        tot_ms = sum(ginfo[(tag, rel)].get("alg_ms", 0.0) for rel in TARGET_PANEL)
        fft_d = ev[(tag, "forgeEDA/FFTStoreBit.aig")]["delay_ps"]
        lz_d = ev[(tag, "forgeEDA/LZ77_Encoder.aig")]["delay_ps"]
        summary[tag] = {
            "flags": flags,
            "delay_gm": _gm(dr),
            "area_gm": _gm(ar),
            "adp_gm": _gm(adpr),
            "wins_orig": w_orig,
            "wins_resyn2": w_res,
            "cec_ok": cec_ok,
            "total_circuits": len(all_circuits),
            "alg_ms_panel": tot_ms,
            "fft_ps": fft_d,
            "lz77_ps": lz_d,
        }
        print(
            f"  {tag:14s} | delay={_gm(dr):.4f}x ({(_gm(dr)-1)*100:+6.2f}%) | "
            f"area={_gm(ar):.4f}x ({(_gm(ar)-1)*100:+6.2f}%) | ADP={_gm(adpr):.4f}x ({(_gm(adpr)-1)*100:+6.2f}%) | "
            f"w_orig={w_orig:2d}/16 w_res={w_res:2d}/16 cec={cec_ok}/{len(all_circuits)} | "
            f"FFT={fft_d:6.1f} LZ77={lz_d:8.1f} ms={tot_ms:6.0f}",
            flush=True,
        )

    out_json = run / "output/sweep_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print("DONE step16_claude_step2_sweep", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
