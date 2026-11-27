#!/usr/bin/env python3
"""Paper Evaluation & Ablation Suite across all 5 datasets (ACE, OpenABC, iwls2022, iwls2024, forgeEDA).

Runs:
  - abc_ifg:          strash; if -g -K 6 -C 8
  - abc_strong_delay: strash; if -g -K 6 -C 8; strash; dch (in-session map preserving choices)
  - stageB_no_fograd: pure 66415 Stage B with --no-fo-grad (Path 2 g_fo = 0)
  - stageB_no_gini:   pure 66415 Stage B with --disc-lam 0.0 (w/o Gini discretization magnet)

Merges with frozen verified full-905 baselines:
  - orig, resyn2, cpu_4b (step4)
  - stageA (step17 cap0, Slurm 66395)
  - stageB_full (step26 pure 66415, Slurm 66427)
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from scripts.step5_longchain_panel import DS_ROOT, TARGET_PANEL
from scripts.step8_gpu_struct_full905 import STEP4_FULL905_SUMMARY
from scripts.step19_stageb_foinc_panel import _eval_aig_task, _run_abc_strong_task, _run_gpuls_diag

STAGEA_905_JSONL = Path(
    "/data/wushangge/CULS_LiteCut/runs/20261126T080537Z-step17-claude-final-full905-270d8735/cap0/output/results.jsonl"
)
STAGEB_905_JSONL = Path(
    "/data/wushangge/CULS_LiteCut/runs/20261127T064134Z-step26-stageb-full905-e101acb0/output/results.jsonl"
)

ABC_FLOWS = {
    "abc_ifg": "strash; if -g -K 6 -C 8",
    "abc_strong_delay": "strash; if -g -K 6 -C 8; strash; dch",
}
GPU_ABLATIONS = {
    "stageB_no_fograd": "-r 3 --joint-grad --steps 25 --lr 0.20 --disc-lam 0.30 --fo-inc-lam 0.0 --no-fo-grad -v 1",
    "stageB_no_gini": "-r 3 --joint-grad --steps 25 --lr 0.20 --disc-lam 0.0 --fo-inc-lam 0.0 -v 1",
}


def _gm(xs: list[float]) -> float:
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--gpuls", required=True, type=Path)
    parser.add_argument("--suites", required=True, help="Comma-separated suite names, e.g. ACE,OpenABC,iwls2022,iwls2024")
    parser.add_argument("--include-panel", action="store_true", help="Also include all 16 TARGET_PANEL circuits")
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    run = args.run.resolve()
    gpuls_bin = args.gpuls.resolve()
    out_dir = run / "output"
    work = out_dir / "work"
    work.mkdir(parents=True, exist_ok=True)

    wanted_suites = {s.strip() for s in args.suites.split(",") if s.strip()}
    step4_data = json.loads(STEP4_FULL905_SUMMARY.read_text())["circuits"]
    base_map = {c["relative_path"]: c.get("mapping", {}) for c in step4_data}

    stageA_map = {json.loads(l)["relative_path"]: json.loads(l) for l in STAGEA_905_JSONL.open()}
    stageB_map = {json.loads(l)["relative_path"]: json.loads(l) for l in STAGEB_905_JSONL.open()}

    circuits = sorted(
        rel for rel in base_map
        if rel.split("/")[0] in wanted_suites and base_map[rel].get("orig", {}).get("delay_ps")
    )
    if args.include_panel:
        for rel in TARGET_PANEL:
            if rel not in circuits and base_map.get(rel, {}).get("orig", {}).get("delay_ps"):
                circuits.append(rel)

    print(f"=== Paper Eval Suite: {sorted(wanted_suites)} (include_panel={args.include_panel}) | N={len(circuits)} ===", flush=True)

    # 1) Launch ABC strong delay baselines + GPU ablations concurrently!
    # Submit all ABC tasks to the 32-worker CPU pool FIRST while the main thread runs GPU sequentially on RTX 4090.
    eval_res: dict[tuple[str, str], dict] = {}
    ginfo: dict[tuple[str, str], dict] = {}
    t_all0 = time.perf_counter()

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = []
        for tag, flow in ABC_FLOWS.items():
            for rel in circuits:
                slug = rel.replace("/", "__").removesuffix(".aig")
                cdir = work / f"{tag}__{slug}"
                futs.append(pool.submit(_run_abc_strong_task, (tag, rel, flow, str(cdir))))
        print(f"Dispatched {len(futs)} ABC baseline tasks to {args.workers} CPU workers.", flush=True)

        # Run GPU ablations sequentially on RTX 4090, dispatching each finished AIG immediately to CPU pool
        for tag, flags in GPU_ABLATIONS.items():
            t_g0 = time.perf_counter()
            ok_cnt = 0
            for idx, rel in enumerate(circuits, 1):
                slug = rel.replace("/", "__").removesuffix(".aig")
                cdir = work / f"{tag}__{slug}"
                cdir.mkdir(parents=True, exist_ok=True)
                out_aig = cdir / "out.aig"
                gi = _run_gpuls_diag(gpuls_bin, DS_ROOT / rel, out_aig, flags, timeout_s=180)
                ginfo[(tag, rel)] = gi
                if gi["status"] == "OK":
                    ok_cnt += 1
                    futs.append(pool.submit(_eval_aig_task, (tag, rel, str(out_aig), str(cdir))))
                if idx % 50 == 0 or idx == len(circuits):
                    print(f"  [GPU {tag}] {idx}/{len(circuits)} done (ok={ok_cnt}, elapsed={time.perf_counter()-t_g0:.1f}s)", flush=True)

        done_cnt = 0
        total_futs = len(futs)
        for fut in as_completed(futs):
            tag, rel, res = fut.result()
            eval_res[(tag, rel)] = res
            done_cnt += 1
            if done_cnt % 100 == 0 or done_cnt == total_futs:
                print(f"  [CPU eval] {done_cnt}/{total_futs} finished (total elapsed={time.perf_counter()-t_all0:.1f}s)", flush=True)

    # 2) Write per-circuit merged results.jsonl
    jsonl_path = out_dir / "results.jsonl"
    with jsonl_path.open("w") as wf:
        for rel in circuits:
            suite = rel.split("/")[0]
            bm = base_map[rel]
            row = {
                "relative_path": rel,
                "suite": suite,
                "in_target_panel": rel in TARGET_PANEL,
                "orig": bm.get("orig", {}),
                "resyn2": bm.get("resyn2", {}),
                "cpu_4b": bm.get("cpu_4b", {}),
                "stageA": stageA_map.get(rel, {}).get("gpu_struct_rw4b", {}),
                "stageA_gpu_s": stageA_map.get(rel, {}).get("gpu_info", {}).get("rw4b_total_s", 0.0),
                "stageB_full": stageB_map.get(rel, {}).get("gpu_struct_rw4b", {}),
                "stageB_full_gpu_s": stageB_map.get(rel, {}).get("gpu_info", {}).get("rw4b_total_s", 0.0),
                "abc_ifg": eval_res.get(("abc_ifg", rel), {}),
                "abc_strong_delay": eval_res.get(("abc_strong_delay", rel), {}),
                "stageB_no_fograd": eval_res.get(("stageB_no_fograd", rel), {}),
                "stageB_no_fograd_gpu_s": ginfo.get(("stageB_no_fograd", rel), {}).get("rw4b_total_s", 0.0),
                "stageB_no_gini": eval_res.get(("stageB_no_gini", rel), {}),
                "stageB_no_gini_gpu_s": ginfo.get(("stageB_no_gini", rel), {}).get("rw4b_total_s", 0.0),
            }
            wf.write(json.dumps(row) + "\n")

    # 3) Summarize per suite + TARGET_PANEL (if included)
    methods = ["resyn2", "abc_ifg", "abc_strong_delay", "stageA", "stageB_no_fograd", "stageB_no_gini", "stageB_full"]
    groups: dict[str, list[str]] = defaultdict(list)
    for rel in circuits:
        if rel.split("/")[0] in wanted_suites:
            groups[rel.split("/")[0]].append(rel)
    if len(wanted_suites) > 1:
        groups["ALL_WANTED"] = [rel for rel in circuits if rel.split("/")[0] in wanted_suites]
    if args.include_panel:
        groups["TARGET_PANEL_16"] = [rel for rel in TARGET_PANEL if rel in circuits]

    summary: dict = {"suites": sorted(wanted_suites), "groups": {}}
    for gname, grels in groups.items():
        print(f"\n=== GROUP: {gname} (N={len(grels)}) ===", flush=True)
        gsum = {}
        for m in methods:
            dr, ar, adpr = [], [], []
            cec_ok, w_orig, tot_s = 0, 0, 0.0
            for rel in grels:
                od = base_map[rel]["orig"]["delay_ps"]
                oa = base_map[rel]["orig"]["area"]
                if m == "resyn2":
                    r = base_map[rel].get("resyn2", {})
                    ok = bool(r.get("delay_ps") and r.get("area"))
                elif m == "stageA":
                    r = stageA_map.get(rel, {}).get("gpu_struct_rw4b", {})
                    ok = (r.get("cec") == "PASS" and r.get("delay_ps"))
                    tot_s += stageA_map.get(rel, {}).get("gpu_info", {}).get("rw4b_total_s", 0.0)
                elif m == "stageB_full":
                    r = stageB_map.get(rel, {}).get("gpu_struct_rw4b", {})
                    ok = (r.get("cec") == "PASS" and r.get("delay_ps"))
                    tot_s += stageB_map.get(rel, {}).get("gpu_info", {}).get("rw4b_total_s", 0.0)
                else:
                    r = eval_res.get((m, rel), {})
                    ok = (r.get("cec") == "PASS" and r.get("delay_ps"))
                    if m in GPU_ABLATIONS:
                        tot_s += ginfo.get((m, rel), {}).get("rw4b_total_s", 0.0)
                    else:
                        tot_s += r.get("opt_s", 0.0)
                if ok:
                    cec_ok += 1
                    d_rat = r["delay_ps"] / od
                    a_rat = r["area"] / oa
                    dr.append(d_rat); ar.append(a_rat); adpr.append(d_rat * a_rat)
                    if d_rat < 1.0 - 1e-6:
                        w_orig += 1
            dgm, agm, adpgm = _gm(dr), _gm(ar), _gm(adpr)
            gsum[m] = {
                "n_ok": cec_ok, "n_total": len(grels), "wins_orig": w_orig,
                "delay_gm": dgm, "area_gm": agm, "adp_gm": adpgm, "core_s": tot_s,
            }
            print(
                f"  {m:18s} | ok={cec_ok:3d}/{len(grels):3d} | D={dgm:.4f}x ({(dgm-1)*100:+6.2f}%) | "
                f"A={agm:.4f}x ({(agm-1)*100:+6.2f}%) | ADP={adpgm:.4f}x ({(adpgm-1)*100:+6.2f}%) | "
                f"w_orig={w_orig:3d} | core={tot_s:7.2f}s",
                flush=True,
            )
        summary["groups"][gname] = gsum

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\nDONE step23_paper_eval_suite", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
