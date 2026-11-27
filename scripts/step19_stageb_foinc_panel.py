#!/usr/bin/env python3
"""Stage B-1 L_fo_inc verification:
1) Hard Gate 1: fp64 finite-difference grad-check (<1e-4) with --fo-inc-lam 0.5 --fo-inc-th 6.0;
2) 16-circuit target panel comparing orig, abc_ifg, abc_strong_delay, Stage A, Stage B-1 (66415 old), Stage B-1 (new);
3) Fanout-guard rejection rate & per-round diagnostics for sirv_sim_ram and branch_prediction_globle.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from global_timing.evaluation import cec_passed, parse_timing
from scripts.step4_fanout_pilot import ABC_EVAL, LIB_ASAP7
from scripts.step5_longchain_panel import DS_ROOT, TARGET_PANEL, _eval_aig
from scripts.step8_gpu_struct_full905 import STEP4_FULL905_SUMMARY, TOTAL_RE

GRAD_CHECK_CIRCUITS = [
    "OpenABC/c1355_orig.aig",
    "OpenABC/c5315_orig.aig",
    "forgeEDA/intra_pred_PE.aig",
]

CHECK_RE = re.compile(
    r"rw4b-grad CHECK: M=(\d+) tested=(\d+) max_rel_err=([0-9.eE+-]+) max_abs_err=([0-9.eE+-]+) \(([^)]+)\) \[([^\]]+)\]"
)
GRAD_ROUND_RE = re.compile(
    r"rw4b-grad round (\d+): M=(\d+) applied=(\d+)\(raw=(\d+)\)(?: z0=([0-9.]+))? theta=([0-9.-]+) avgMaxP=([0-9.]+) "
    r"loss=([0-9.eE+-]+)->([0-9.eE+-]+) discD=([0-9.]+)\(D0=([0-9.]+)\) dAND=([0-9-]+) grad=([0-9.]+)ms disc=([0-9.]+)ms"
)


def _run_gpuls_diag(gpuls_bin: Path, in_aig: Path, out_aig: Path, flags: str, timeout_s: int = 180) -> dict:
    cmd = f"read {in_aig}; rw4b {flags}; write {out_aig}"
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            [str(gpuls_bin), "-c", cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"status": "TIMEOUT", "wall_s": time.perf_counter() - t0, "rounds": []}
    wall_s = time.perf_counter() - t0
    out = proc.stdout or ""
    if proc.returncode != 0 or not out_aig.exists() or "not recognized" in out:
        return {"status": f"ERROR_RC_{proc.returncode}", "wall_s": wall_s, "error_tail": out[-400:], "rounds": []}
    info: dict = {"status": "OK", "wall_s": wall_s, "rounds": []}
    m = TOTAL_RE.search(out)
    if m:
        info.update({
            "ands": int(m.group(1)),
            "level": int(m.group(2)),
            "applied": int(m.group(3)),
            "rw4b_total_s": float(m.group(4)) / 1000.0,
        })
    for rm in GRAD_ROUND_RE.finditer(out):
        info["rounds"].append({
            "round": int(rm.group(1)),
            "M": int(rm.group(2)),
            "applied": int(rm.group(3)),
            "raw": int(rm.group(4)),
            "z0": float(rm.group(5)) if rm.group(5) is not None else -1.0,
            "theta": float(rm.group(6)),
            "avgMaxP": float(rm.group(7)),
            "initLoss": float(rm.group(8)),
            "finalLoss": float(rm.group(9)),
            "discD": float(rm.group(10)),
            "D0": float(rm.group(11)),
            "dAND": int(rm.group(12)),
            "grad_ms": float(rm.group(13)),
            "disc_ms": float(rm.group(14)),
        })
    return info


def _eval_aig_task(item: tuple[str, str, str, str]) -> tuple[str, str, dict]:
    tag, rel, out_str, w_str = item
    w = Path(w_str)
    try:
        res = _eval_aig(DS_ROOT / rel, Path(out_str), w / "map")
    except Exception as exc:
        res = {"status": "ERROR", "cec": "ERROR", "error": f"{type(exc).__name__}: {exc}"}
    finally:
        shutil.rmtree(w, ignore_errors=True)
    return tag, rel, res


def _run_abc_strong_task(item: tuple[str, str, str, str]) -> tuple[str, str, dict]:
    """Run abc_ifg or abc_strong_delay (with in-session dch structural choices preserved during ASAP7 map) + cec -n."""
    tag, rel, abc_flow, w_str = item
    orig_p = DS_ROOT / rel
    w = Path(w_str)
    w.mkdir(parents=True, exist_ok=True)
    out_aig = w / "out.aig"
    try:
        # 1) Run optimization + save AIG for cec -n and measure pure optimization wall time
        t0 = time.perf_counter()
        p_opt = subprocess.run(
            [str(ABC_EVAL), "-c", f"read_aiger {orig_p}; {abc_flow}; write_aiger {out_aig}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=600,
        )
        opt_s = time.perf_counter() - t0
        if p_opt.returncode != 0 or not out_aig.is_file():
            return tag, rel, {"status": "ERROR", "cec": "ERROR", "error": (p_opt.stdout or "")[-300:]}

        # 2) Strict cec -n equivalence check
        p_cec = subprocess.run(
            [str(ABC_EVAL), "-c", f"cec -n {orig_p} {out_aig}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=600,
        )
        if not cec_passed(p_cec.stdout or ""):
            return tag, rel, {"status": "FAIL", "cec": "FAIL", "error": (p_cec.stdout or "")[-300:]}

        # 3) Exact same ASAP7 map; topo; stime -p (in-session for dch so structural choices are preserved)
        env = dict(os.environ, CULS_GLOBAL_TIMING_REPORT="1")
        if "dch" in abc_flow:
            map_cmd = f"read_lib {LIB_ASAP7}; read_aiger {orig_p}; {abc_flow}; map; topo; stime -p"
        else:
            map_cmd = f"read_lib {LIB_ASAP7}; read_aiger {out_aig}; map; topo; stime -p"
        p_map = subprocess.run(
            [str(ABC_EVAL), "-c", map_cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=600,
            env=env,
        )
        parsed = parse_timing(p_map.stdout or "")
        return tag, rel, {
            "status": "PASS",
            "cec": "PASS",
            "delay_ps": parsed["delay_ps"],
            "area": parsed["area"],
            "adp": parsed["adp"],
            "opt_s": opt_s,
        }
    except Exception as exc:
        return tag, rel, {"status": "ERROR", "cec": "ERROR", "error": f"{type(exc).__name__}: {exc}"}
    finally:
        shutil.rmtree(w, ignore_errors=True)


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

    print("=== HARD GATE 1: FINITE-DIFFERENCE GRADIENT CHECK WITH L_fo_inc (< 1e-4) ===", flush=True)
    all_grad_ok = True
    for rel in GRAD_CHECK_CIRCUITS:
        out_aig = work / f"gchk_{rel.replace('/', '__')}"
        cmd = f"read {DS_ROOT / rel}; rw4b --joint-grad --grad-check --fo-inc-lam 0.5 --fo-inc-th 6.0 -r 1 --steps 10 -v 1; write {out_aig}"
        proc = subprocess.run([str(gpuls_bin), "-c", cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120)
        out = proc.stdout or ""
        m = CHECK_RE.search(out)
        if not m:
            print(f"  {rel:28s} | ERROR: no CHECK line! rc={proc.returncode}\n{out[-600:]}", flush=True)
            all_grad_ok = False
            continue
        M, tested, rel_err, abs_err, worst_str, tag = int(m.group(1)), int(m.group(2)), float(m.group(3)), float(m.group(4)), m.group(5), m.group(6)
        print(f"  {rel:28s} | M={M:5d} tested={tested:3d} max_rel_err={rel_err:.3e} max_abs_err={abs_err:.3e} ({worst_str}) [{tag}]", flush=True)
        if rel_err >= 1e-4:
            all_grad_ok = False
        if out_aig.exists():
            out_aig.unlink()

    if not all_grad_ok:
        print("HARD GATE 1 FAILED: max_rel_err >= 1e-4 (aborting).", flush=True)
        return 2
    print("HARD GATE 1 PASSED: all circuits max_rel_err < 1e-4!\n", flush=True)

    base_map = {c["relative_path"]: c.get("mapping", {}) for c in json.loads(STEP4_FULL905_SUMMARY.read_text())["circuits"]}

    gpu_configs = [
        ("stageA",      "-r 3 -v 1"),
        ("stageB_old",  "-r 3 --joint-grad --steps 25 --lr 0.20 --disc-lam 0.30 --fo-inc-lam 0.0 -v 1"),
        ("stageB_new",  "-r 3 --joint-grad --steps 25 --lr 0.20 --disc-lam 0.30 --fo-inc-lam 0.5 --fo-inc-th 6.0 -v 1"),
    ]
    abc_configs = [
        ("abc_ifg",          "strash; if -g -K 6 -C 8"),
        ("abc_strong_delay", "strash; if -g -K 6 -C 8; strash; dch"),
    ]

    eval_items = []
    ginfo = {}
    for tag, flags in gpu_configs:
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

    abc_items = []
    for tag, flow in abc_configs:
        for rel in TARGET_PANEL:
            slug = rel.replace("/", "__").removesuffix(".aig")
            abc_items.append((tag, rel, flow, str(work / f"{tag}_{slug}")))

    ev = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        f_gpu = [pool.submit(_eval_aig_task, it) for it in eval_items]
        f_abc = [pool.submit(_run_abc_strong_task, it) for it in abc_items]
        for f in f_gpu + f_abc:
            tag, rel, res = f.result()
            ev[(tag, rel)] = res

    print("=== 1. PER-CIRCUIT 16-CIRCUIT TARGET PANEL (ASAP7 delay_ps) ===", flush=True)
    print(f"  {'Circuit':34s} | {'orig':>8s} | {'abc_ifg':>8s} | {'abc_strg':>8s} | {'StageA':>8s} | {'B_66415':>8s} | {'B_foInc':>8s}", flush=True)
    for rel in TARGET_PANEL:
        od = base_map[rel]["orig"]["delay_ps"]
        vals = []
        for tag in ("abc_ifg", "abc_strong_delay", "stageA", "stageB_old", "stageB_new"):
            r = ev.get((tag, rel), {})
            vals.append(f"{r.get('delay_ps', -1):8.1f}" if r.get("cec") == "PASS" else f"{'ERR':>8s}")
        print(f"  {rel:34s} | {od:8.1f} | " + " | ".join(vals), flush=True)

    print("\n=== 2. AGGREGATE 16-CIRCUIT PANEL SUMMARY ===", flush=True)
    all_tags = ["abc_ifg", "abc_strong_delay", "stageA", "stageB_old", "stageB_new"]
    summary = {}
    for tag in all_tags:
        dr, ar, adpr = [], [], []
        w_orig = 0
        cec_ok = 0
        for rel in TARGET_PANEL:
            r = ev.get((tag, rel), {})
            if r.get("cec") == "PASS":
                cec_ok += 1
                d_ratio = r["delay_ps"] / base_map[rel]["orig"]["delay_ps"]
                a_ratio = r["area"] / base_map[rel]["orig"]["area"]
                dr.append(d_ratio)
                ar.append(a_ratio)
                adpr.append(d_ratio * a_ratio)
                if d_ratio < 1.0 - 1e-6:
                    w_orig += 1
        if tag.startswith("abc_"):
            tot_ms = sum(ev.get((tag, rel), {}).get("opt_s", 0.0) * 1000.0 for rel in TARGET_PANEL)
        else:
            tot_ms = sum(ginfo[(tag, rel)].get("rw4b_total_s", 0.0) * 1000.0 for rel in TARGET_PANEL)
        summary[tag] = {
            "delay_gm": _gm(dr),
            "area_gm": _gm(ar),
            "adp_gm": _gm(adpr),
            "wins_orig": w_orig,
            "cec_ok": cec_ok,
            "core_ms": tot_ms,
        }
        print(
            f"  {tag:18s} | delay={_gm(dr):.4f}x ({(_gm(dr)-1)*100:+6.2f}%) | "
            f"area={_gm(ar):.4f}x ({(_gm(ar)-1)*100:+6.2f}%) | ADP={_gm(adpr):.4f}x ({(_gm(adpr)-1)*100:+6.2f}%) | "
            f"w_orig={w_orig:2d}/16 cec={cec_ok:2d}/16 | time_ms={tot_ms:8.1f}",
            flush=True,
        )

    print("\n=== 3. DIAGNOSTIC A: POST-DISCRETIZATION FANOUT-GUARD REJECTION RATE (66415 old vs new L_fo_inc) ===", flush=True)
    for tag in ("stageB_old", "stageB_new"):
        tot_raw, tot_app = 0, 0
        res_raw, res_app = 0, 0
        for rel in TARGET_PANEL:
            rds = ginfo[(tag, rel)].get("rounds", [])
            r_raw = sum(x["raw"] for x in rds)
            r_app = sum(x["applied"] for x in rds)
            tot_raw += r_raw
            tot_app += r_app
            if rel == "forgeEDA/reservation_station.aig":
                res_raw, res_app = r_raw, r_app
        tot_blk = tot_raw - tot_app
        res_blk = res_raw - res_app
        print(
            f"  {tag:12s} | ALL 16: raw={tot_raw:6d} applied={tot_app:6d} blocked={tot_blk:5d} ({(100.0*tot_blk/max(1,tot_raw)):5.2f}%) | "
            f"reservation_station: raw={res_raw:5d} applied={res_app:5d} blocked={res_blk:4d} ({(100.0*res_blk/max(1,res_raw)):5.2f}%)",
            flush=True,
        )

    print("\n=== 4. DIAGNOSTIC B: PER-ROUND LOGS FOR sirv_sim_ram & branch_prediction_globle ===", flush=True)
    for rel in ("forgeEDA/sirv_sim_ram.aig", "forgeEDA/branch_prediction_globle.aig", "forgeEDA/reservation_station.aig"):
        print(f"  --- {rel} ---", flush=True)
        for tag in ("stageB_old", "stageB_new"):
            r_eval = ev.get((tag, rel), {})
            print(f"    [{tag}] ASAP7 delay={r_eval.get('delay_ps', -1):.1f} ps, area={r_eval.get('area', -1):.1f}", flush=True)
            for rd in ginfo[(tag, rel)].get("rounds", []):
                blk = rd["raw"] - rd["applied"]
                print(
                    f"      R{rd['round']}: M={rd['M']:5d} raw={rd['raw']:5d} applied={rd['applied']:5d} blocked={blk:4d} "
                    f"theta={rd['theta']:.2f} avgMaxP={rd['avgMaxP']:.3f} discD={rd['discD']:.2f}(D0={rd['D0']:.2f}) dAND={rd['dAND']:+d}",
                    flush=True,
                )

    (run / "output/panel_summary.json").write_text(json.dumps({"summary": summary, "eval": {f"{k[0]}::{k[1]}": v for k, v in ev.items()}}, indent=2))
    print("DONE step19_stageb_foinc_panel", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
