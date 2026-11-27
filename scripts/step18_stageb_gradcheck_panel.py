#!/usr/bin/env python3
"""Stage B-1 verification: Hard Gate 1 (<1e-4 finite-difference grad check) + Hard Gate 2 (16-circuit panel A/B vs Stage A)."""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from scripts.step5_longchain_panel import DS_ROOT, TARGET_PANEL, _eval_aig
from scripts.step8_gpu_struct_full905 import STEP4_FULL905_SUMMARY, _run_gpuls

GRAD_CHECK_CIRCUITS = [
    "OpenABC/c1355_orig.aig",
    "OpenABC/c5315_orig.aig",
    "forgeEDA/intra_pred_PE.aig",
]

CHECK_RE = re.compile(r"rw4b-grad CHECK: M=(\d+) tested=(\d+) max_rel_err=([0-9.eE+-]+) max_abs_err=([0-9.eE+-]+) \(([^)]+)\) \[([^\]]+)\]")
ROUND_RE = re.compile(r"rw4b-grad round (\d+):.*")


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

    print("=== HARD GATE 1: FINITE-DIFFERENCE GRADIENT CHECK (< 1e-4) ===", flush=True)
    all_grad_ok = True
    for rel in GRAD_CHECK_CIRCUITS:
        out_aig = work / f"gchk_{rel.replace('/', '__')}"
        cmd = f"read {DS_ROOT / rel}; rw4b --joint-grad --grad-check -r 1 --steps 10 -v 1; write {out_aig}"
        proc = subprocess.run([str(gpuls_bin), "-c", cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120)
        out = proc.stdout or ""
        m = CHECK_RE.search(out)
        if not m:
            print(f"  {rel:28s} | ERROR: no CHECK line! rc={proc.returncode}\n{out[-600:]}", flush=True)
            all_grad_ok = False
            continue
        M, tested, rel_err, abs_err, worst_str, tag = int(m.group(1)), int(m.group(2)), float(m.group(3)), float(m.group(4)), m.group(5), m.group(6)
        print(f"  {rel:28s} | M={M:5d} tested={tested:3d} max_rel_err={rel_err:.3e} max_abs_err={abs_err:.3e} ({worst_str}) [{tag}]", flush=True)
        for rm in ROUND_RE.finditer(out):
            print(f"    {rm.group(0)}", flush=True)
        if rel_err >= 1e-4:
            all_grad_ok = False
        if out_aig.exists():
            out_aig.unlink()

    if not all_grad_ok:
        print("HARD GATE 1 FAILED: max_rel_err >= 1e-4 (aborting before panel benchmark).", flush=True)
        return 2
    print("HARD GATE 1 PASSED: all circuits max_rel_err < 1e-4!\n", flush=True)

    print("=== HARD GATE 2: 16-CIRCUIT TARGET PANEL (Stage A vs Stage B --joint-grad) ===", flush=True)
    base_map = {c["relative_path"]: c.get("mapping", {}) for c in json.loads(STEP4_FULL905_SUMMARY.read_text())["circuits"]}
    configs = [
        ("stageA_default",  "-r 3 -v 1"),
        ("stageB_lr08_d20", "-r 3 --joint-grad --steps 25 --lr 0.08 --disc-lam 0.20 -v 1"),
        ("stageB_lr10_d30", "-r 3 --joint-grad --steps 25 --lr 0.10 --disc-lam 0.30 -v 1"),
        ("stageB_lr15_d30", "-r 3 --joint-grad --steps 25 --lr 0.15 --disc-lam 0.30 -v 1"),
        ("stageB_w30_lr10", "-r 3 --joint-grad --steps 25 --adp-w 3.0 --lr 0.10 --disc-lam 0.30 -v 1"),
    ]
    eval_items = []
    ginfo = {}
    for tag, flags in configs:
        for rel in TARGET_PANEL:
            slug = rel.replace("/", "__").removesuffix(".aig")
            cdir = work / f"{tag}_{slug}"
            cdir.mkdir(parents=True, exist_ok=True)
            out_aig = cdir / "out.aig"
            gi = _run_gpuls(gpuls_bin, DS_ROOT / rel, out_aig, flags, timeout_s=180)
            ginfo[(tag, rel)] = gi
            if gi["status"] == "OK":
                eval_items.append((tag, rel, str(out_aig), str(cdir)))
            else:
                print(f"  WARN {tag} {rel}: {gi}", flush=True)

    ev = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for tag, rel, res in pool.map(_eval_one, eval_items):
            ev[(tag, rel)] = res

    for rel in TARGET_PANEL:
        od = base_map[rel]["orig"]["delay_ps"]
        ra = ev.get(("stageA_default", rel), {})
        rb = ev.get(("stageB_lr10_d30", rel), {})
        ga = ginfo[("stageA_default", rel)]
        gb = ginfo[("stageB_lr10_d30", rel)]
        print(
            f"  {rel:34s} | orig={od:8.1f} | "
            f"A={ra.get('delay_ps', -1):8.1f}({ra.get('cec','ERR')}, {ga.get('rw4b_total_s',0)*1000:5.0f}ms) | "
            f"B(lr10)={rb.get('delay_ps', -1):8.1f}({rb.get('cec','ERR')}, {gb.get('rw4b_total_s',0)*1000:5.0f}ms)",
            flush=True,
        )

    print("\n=== PANEL AGGREGATE ===", flush=True)
    for tag, flags in configs:
        dr = [ev[(tag, rel)]["delay_ps"] / base_map[rel]["orig"]["delay_ps"] for rel in TARGET_PANEL if (tag, rel) in ev and ev[(tag, rel)].get("cec") == "PASS"]
        ar = [ev[(tag, rel)]["area"] / base_map[rel]["orig"]["area"] for rel in TARGET_PANEL if (tag, rel) in ev and ev[(tag, rel)].get("cec") == "PASS"]
        adpr = [d * a for d, a in zip(dr, ar)]
        w_orig = sum(1 for x in dr if x < 1.0 - 1e-6)
        w_res = sum(1 for rel in TARGET_PANEL if (tag, rel) in ev and ev[(tag, rel)].get("cec") == "PASS" and ev[(tag, rel)]["delay_ps"] < base_map[rel]["abc_resyn2"]["delay_ps"] - 1e-6)
        cec_ok = sum(1 for rel in TARGET_PANEL if (tag, rel) in ev and ev[(tag, rel)].get("cec") == "PASS")
        tot_ms = sum(ginfo[(tag, rel)].get("rw4b_total_s", 0.0) * 1000.0 for rel in TARGET_PANEL)
        print(
            f"  {tag:16s} | delay={_gm(dr):.4f}x ({(_gm(dr)-1)*100:+6.2f}%) | "
            f"area={_gm(ar):.4f}x ({(_gm(ar)-1)*100:+6.2f}%) | ADP={_gm(adpr):.4f}x ({(_gm(adpr)-1)*100:+6.2f}%) | "
            f"w_orig={w_orig:2d}/16 w_res={w_res:2d}/16 cec={cec_ok}/16 | core_ms={tot_ms:7.1f}",
            flush=True,
        )

    print("DONE step18_stageb_gradcheck_panel", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
