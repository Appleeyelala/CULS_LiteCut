#!/usr/bin/env python3
"""Stage B-3 One-Shot Blind Test on the locked 24-circuit held-out panel (panel.json).

Evaluates all 24 unseen circuits on ASAP7 (`map; topo; stime -p`) with strict `cec -n`:
  - orig
  - abc_resyn2
  - abc_ifg          (strash; if -g -K 6 -C 8)
  - abc_strong_delay (strash; if -g -K 6 -C 8; strash; dch, in-session map)
  - stageA           (rw4b -r 3 --adp-w 2.0 --kappa 2.0 --fo-cap 0)
  - stageB_no_fograd (Stage B locked w/o Path 2 fanout adjoint)
  - stageB_no_gini   (Stage B locked w/o Gini discretization magnet)
  - stageB_full      (Stage B locked pure 66415/66427 champion)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from global_timing.evaluation import cec_passed, parse_timing
from scripts.step4_fanout_pilot import ABC_EVAL, LIB_ASAP7
from scripts.step5_longchain_panel import _eval_aig
from scripts.step19_stageb_foinc_panel import _run_gpuls_diag

BLIND_PANEL_JSON = Path(
    "/data/wushangge/CULS_LiteCut/runs/20261119T070904Z-paper-split24-399b690c/output/data/panel.json"
)

ABC_FLOWS = {
    "abc_resyn2": "resyn2",
    "abc_ifg": "strash; if -g -K 6 -C 8",
    "abc_strong_delay": "strash; if -g -K 6 -C 8; strash; dch",
}

GPU_FLOWS = {
    "stageA": "-r 3 --adp-w 2.0 --kappa 2.0 --fo-cap 0 -v 1",
    "stageB_no_fograd": "-r 3 --joint-grad --steps 25 --lr 0.20 --disc-lam 0.30 --fo-inc-lam 0.0 --no-fo-grad -v 1",
    "stageB_no_gini": "-r 3 --joint-grad --steps 25 --lr 0.20 --disc-lam 0.0 --fo-inc-lam 0.0 -v 1",
    "stageB_full": "-r 3 --joint-grad --steps 25 --lr 0.20 --disc-lam 0.30 --fo-inc-lam 0.0 -v 1",
}


def _gm(xs: list[float]) -> float:
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def _eval_orig_task(item: tuple[str, str, str]) -> tuple[str, str, dict]:
    cid, aig_str, w_str = item
    w = Path(w_str)
    w.mkdir(parents=True, exist_ok=True)
    try:
        env = dict(os.environ, CULS_GLOBAL_TIMING_REPORT="1")
        p_map = subprocess.run(
            [str(ABC_EVAL), "-c", f"read_lib {LIB_ASAP7}; read_aiger {aig_str}; map; topo; stime -p"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=600, env=env,
        )
        parsed = parse_timing(p_map.stdout or "")
        return "orig", cid, {"status": "PASS", "cec": "PASS", **parsed}
    except Exception as exc:
        return "orig", cid, {"status": "ERROR", "cec": "ERROR", "error": f"{type(exc).__name__}: {exc}"}
    finally:
        shutil.rmtree(w, ignore_errors=True)


def _eval_abc_task(item: tuple[str, str, str, str, str]) -> tuple[str, str, dict]:
    tag, cid, orig_str, flow, w_str = item
    w = Path(w_str)
    w.mkdir(parents=True, exist_ok=True)
    out_aig = w / "out.aig"
    try:
        t0 = time.perf_counter()
        p_opt = subprocess.run(
            [str(ABC_EVAL), "-c", f"read_aiger {orig_str}; {flow}; write_aiger {out_aig}"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=600,
        )
        opt_s = time.perf_counter() - t0
        if p_opt.returncode != 0 or not out_aig.is_file():
            return tag, cid, {"status": "ERROR", "cec": "ERROR", "error": (p_opt.stdout or "")[-300:]}
        p_cec = subprocess.run(
            [str(ABC_EVAL), "-c", f"cec -n {orig_str} {out_aig}"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=600,
        )
        if not cec_passed(p_cec.stdout or ""):
            return tag, cid, {"status": "FAIL", "cec": "FAIL", "error": (p_cec.stdout or "")[-300:]}
        env = dict(os.environ, CULS_GLOBAL_TIMING_REPORT="1")
        if "dch" in flow:
            map_cmd = f"read_lib {LIB_ASAP7}; read_aiger {orig_str}; {flow}; map; topo; stime -p"
        else:
            map_cmd = f"read_lib {LIB_ASAP7}; read_aiger {out_aig}; map; topo; stime -p"
        p_map = subprocess.run(
            [str(ABC_EVAL), "-c", map_cmd],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=600, env=env,
        )
        parsed = parse_timing(p_map.stdout or "")
        return tag, cid, {"status": "PASS", "cec": "PASS", "opt_s": opt_s, **parsed}
    except Exception as exc:
        return tag, cid, {"status": "ERROR", "cec": "ERROR", "error": f"{type(exc).__name__}: {exc}"}
    finally:
        shutil.rmtree(w, ignore_errors=True)


def _eval_gpu_aig_task(item: tuple[str, str, str, str, str]) -> tuple[str, str, dict]:
    tag, cid, orig_str, out_str, w_str = item
    w = Path(w_str)
    try:
        res = _eval_aig(Path(orig_str), Path(out_str), w / "map")
    except Exception as exc:
        res = {"status": "ERROR", "cec": "ERROR", "error": f"{type(exc).__name__}: {exc}"}
    finally:
        shutil.rmtree(w, ignore_errors=True)
    return tag, cid, res


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--gpuls", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()

    run = args.run.resolve()
    gpuls_bin = args.gpuls.resolve()
    out_dir = run / "output"
    work = out_dir / "work"
    work.mkdir(parents=True, exist_ok=True)

    panel = json.loads(BLIND_PANEL_JSON.read_text())
    rows = panel["rows"]
    print(f"=== Stage B-3 One-Shot Blind Test: N={len(rows)} circuits ===", flush=True)

    circuits = []
    for idx, r in enumerate(rows, 1):
        name = Path(r["source"]).stem
        cid = f"{idx:02d}_{name}"
        circuits.append((cid, name, r["aig"], r["ands"], r["source"]))

    eval_res: dict[tuple[str, str], dict] = {}
    ginfo: dict[tuple[str, str], dict] = {}

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = []
        for cid, name, aig_str, ands, src in circuits:
            futs.append(pool.submit(_eval_orig_task, (cid, aig_str, str(work / f"orig__{cid}"))))
            for tag, flow in ABC_FLOWS.items():
                futs.append(pool.submit(_eval_abc_task, (tag, cid, aig_str, flow, str(work / f"{tag}__{cid}"))))

        for tag, flags in GPU_FLOWS.items():
            for cid, name, aig_str, ands, src in circuits:
                cdir = work / f"{tag}__{cid}"
                cdir.mkdir(parents=True, exist_ok=True)
                out_aig = cdir / "out.aig"
                gi = _run_gpuls_diag(gpuls_bin, Path(aig_str), out_aig, flags, timeout_s=180)
                ginfo[(tag, cid)] = gi
                if gi["status"] == "OK":
                    futs.append(pool.submit(_eval_gpu_aig_task, (tag, cid, aig_str, str(out_aig), str(cdir))))
                else:
                    eval_res[(tag, cid)] = {"status": gi["status"], "cec": "ERROR"}
            print(f"  [GPU {tag}] 24/24 done", flush=True)

        for fut in as_completed(futs):
            tag, cid, res = fut.result()
            eval_res[(tag, cid)] = res

    # Save results.jsonl
    with (out_dir / "results.jsonl").open("w") as wf:
        for cid, name, aig_str, ands, src in circuits:
            rec = {"id": cid, "name": name, "ands": ands, "source": src, "aig": aig_str}
            rec["orig"] = eval_res.get(("orig", cid), {})
            for tag in list(ABC_FLOWS) + list(GPU_FLOWS):
                rec[tag] = eval_res.get((tag, cid), {})
                if tag in GPU_FLOWS:
                    rec[f"{tag}_gpu_s"] = ginfo.get((tag, cid), {}).get("rw4b_total_s", 0.0)
                    rec[f"{tag}_applied"] = ginfo.get((tag, cid), {}).get("applied", 0)
            wf.write(json.dumps(rec) + "\n")

    # Print per-circuit delay table
    print("\n=== PER-CIRCUIT ASAP7 DELAY (ps) ===", flush=True)
    print(f"  {'Circuit':26s} | {'ANDs':>6s} | {'orig':>8s} | {'resyn2':>8s} | {'abc_ifg':>8s} | {'abc_dch':>8s} | {'StageA':>8s} | {'B_noFo':>8s} | {'B_Full':>8s}", flush=True)
    for cid, name, aig_str, ands, src in circuits:
        def _d(t: str) -> str:
            r = eval_res.get((t, cid), {})
            return f"{r['delay_ps']:8.1f}" if r.get("cec") == "PASS" and r.get("delay_ps") else f"{'ERR':>8s}"
        print(f"  {cid:26s} | {ands:6d} | {_d('orig')} | {_d('abc_resyn2')} | {_d('abc_ifg')} | {_d('abc_strong_delay')} | {_d('stageA')} | {_d('stageB_no_fograd')} | {_d('stageB_full')}", flush=True)

    # Aggregate summary
    print("\n=== STAGE B-3 BLIND 24-CIRCUIT PANEL SUMMARY (GM vs orig) ===", flush=True)
    methods = ["abc_resyn2", "abc_ifg", "abc_strong_delay", "stageA", "stageB_no_fograd", "stageB_no_gini", "stageB_full"]
    summary = {}
    for m in methods:
        dr, ar, adpr = [], [], []
        ok_cnt, w_orig, tot_s = 0, 0, 0.0
        for cid, name, aig_str, ands, src in circuits:
            orig = eval_res.get(("orig", cid), {})
            r = eval_res.get((m, cid), {})
            if orig.get("delay_ps") and r.get("cec") == "PASS" and r.get("delay_ps") and r.get("area"):
                ok_cnt += 1
                d_rat = r["delay_ps"] / orig["delay_ps"]
                a_rat = r["area"] / orig["area"]
                dr.append(d_rat); ar.append(a_rat); adpr.append(d_rat * a_rat)
                if d_rat < 1.0 - 1e-6:
                    w_orig += 1
                tot_s += r.get("opt_s", 0.0) if m in ABC_FLOWS else ginfo.get((m, cid), {}).get("rw4b_total_s", 0.0)
        dgm, agm, adpgm = _gm(dr), _gm(ar), _gm(adpr)
        summary[m] = {
            "n_ok": ok_cnt, "n_total": len(circuits), "wins_orig": w_orig,
            "delay_gm": dgm, "area_gm": agm, "adp_gm": adpgm, "core_s": tot_s,
        }
        print(
            f"  {m:18s} | ok={ok_cnt:2d}/24 | D={dgm:.4f}x ({(dgm-1)*100:+6.2f}%) | "
            f"A={agm:.4f}x ({(agm-1)*100:+6.2f}%) | ADP={adpgm:.4f}x ({(adpgm-1)*100:+6.2f}%) | "
            f"w_orig={w_orig:2d}/24 | core={tot_s:7.2f}s",
            flush=True,
        )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\nDONE step24_stageb3_blind24", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
