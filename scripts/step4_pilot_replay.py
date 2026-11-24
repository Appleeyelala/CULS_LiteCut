#!/usr/bin/env python3
"""Step 4 pilot: conflict-free AIG joint installation comparing 4A (local) vs 4B (global PI->PO sensitivity) vs ABC native."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import time
from pathlib import Path

from global_timing.aig_replay import (
    ReplacementOp,
    apply_and_write_aig,
    bounded_mffc_nodes,
    candidate_output_complement,
    global_po_sensitivity,
    original_cone_truth,
    output_tns_metrics,
    select_nonconflicting,
    verify_cec,
)
from global_timing.level_weights import original_leaf_depths, propagate_levels
from global_timing.structure import parse_aig, references

PILOTS = ("c1355", "c5315", "intra_pred_PE", "cabac_ucontext")
ABC_ORIG = Path("/data/wushangge/CULS_LiteCut/runs/20261123T041932Z-rwr-dump-build2-3c2894f0/build/abc_orig/abc")
ABC_DUMP = Path("/data/wushangge/CULS_LiteCut/runs/20261123T041932Z-rwr-dump-build2-3c2894f0/build/abc_dump/abc")
EXPORT3 = Path("/data/wushangge/CULS_LiteCut/runs/20261123T090318Z-level-step3v3-export3-b5c9bc64/output")


def load_pilot_inputs() -> dict[str, tuple[Path, Path]]:
    cohort = json.loads((EXPORT3 / "cohort.json").read_text())
    manifest = json.loads((EXPORT3 / "export_manifest.json").read_text())
    aig_map = {r["name"]: Path(r["path"]) for r in cohort["pilots"]}
    exp_map = {r["circuit"]: Path(r["export"]) / "cuts.jsonl" for r in manifest["results"] if r["group"] == "pilot"}
    return {name: (aig_map[name], exp_map[name]) for name in PILOTS}


def load_native_baselines(repo_root: Path) -> dict[str, dict[str, dict]]:
    tsv = repo_root / "analysis/level_weighted/abc_native/rows.tsv"
    res: dict[str, dict[str, dict]] = {}
    with tsv.open() as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if r["circuit"] in PILOTS:
                res.setdefault(r["circuit"], {})[r["command"]] = {
                    "depth": int(r["depth"]),
                    "delta_depth": int(r["delta_depth"]),
                    "ands": int(r["ands"]),
                    "delta_ands": int(r["delta_ands"]),
                    "tns1": int(r["tns1"]),
                    "tns2": int(r["tns2"]),
                    "tns3": int(r["tns3"]),
                    "runtime_s": float(r["runtime_s"]),
                }
    return res


def run_abc_dump(abc_dump: Path, aig_path: Path, dump_dir: Path, log_path: Path) -> Path:
    dump_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CULS_RWR_DUMP=str(dump_dir))
    cmd = f"read_aiger {aig_path}; rewrite"
    proc = subprocess.run([str(abc_dump), "-c", cmd], env=env, capture_output=True, text=True, timeout=1800)
    log_path.write_text(proc.stdout + proc.stderr)
    if proc.returncode != 0 or not (dump_dir / "cuts.jsonl").is_file():
        raise RuntimeError(f"abc_dump failed on {aig_path}: rc={proc.returncode}")
    return dump_dir / "cuts.jsonl"


def _lse(vals: list[float], tau: float) -> float:
    m = max(vals)
    return m + tau * math.log(sum(math.exp((v - m) / tau) for v in vals))


def select_4a(aig, levels, fanouts, cuts_path: Path, scheme: str, slack_gate: int | None) -> list[ReplacementOp]:
    """4A: fixed-bug local per-cut winner (explicit no-op included), greedy non-conflicting install."""
    tau = 1.0
    offset = aig.po_count
    to_aig = lambda nid: nid if nid <= aig.pi_count else nid - offset

    best_per_root: dict[int, tuple[tuple, dict]] = {}
    with cuts_path.open() as fh:
        for line in fh:
            rec = json.loads(line)
            cands = rec["candidates"]
            if not cands:
                continue
            root = to_aig(rec["node"])
            leaves = [to_aig(lf["id"]) for lf in rec["leaves"]]
            leaves_compl = [int(lf["compl"]) for lf in rec["leaves"]]
            req = levels.required[root]
            if req is None:
                continue
            node_lv = levels.level[root]
            node_slack = req - node_lv
            if slack_gate is not None and node_slack > slack_gate:
                # Off critical region: only allow non-area-increasing rewrites
                allow_area_up = False
            else:
                allow_area_up = True

            depths, _ = original_leaf_depths(aig, root, leaves, levels.level)
            slacks = [req - (levels.level[lf] + dp) for lf, dp in zip(leaves, depths)]

            if scheme == "uniform":
                ws = [1.0] * 4
                tol = 0.0
            else:
                m = min(slacks)
                ws = [math.exp(-(s - m) / tau) for s in slacks]
                tol = 1e-9

            # Explicit no-op option: original cone depths
            orig_J = sum(w * d for w, d in zip(ws, depths))
            options = [(orig_J, 0, -1, None)]  # (J, true_delta_and, subgraph, cand)
            saved = rec["n_nodes_saved"]
            for c in cands:
                if c["self"]:
                    continue
                if c["root_level"] > req:
                    continue
                da = c["nodes_added"] - saved
                if not allow_area_up and da > 0:
                    continue
                j_val = sum(w * d for w, d in zip(ws, c["leaf_depths"]) if d >= 0)
                options.append((j_val, da, c["subgraph"], c))

            best_J = min(o[0] for o in options)
            limit = tol * max(1.0, abs(best_J))
            tied = [o for o in options if abs(o[0] - best_J) <= limit]
            chosen = min(tied, key=lambda o: (o[1], o[2]))
            if chosen[3] is None:
                continue  # keep original
            c = chosen[3]
            da = chosen[1]
            rl_gain = node_lv - c["root_level"]
            # Priority across cuts/nodes: most urgent slack first, then larger level gain, then smaller area
            prio = (-node_slack, rl_gain, orig_J - chosen[0], -da, -root, -c["subgraph"])
            prev = best_per_root.get(root)
            if prev is None or prio > prev[0]:
                best_per_root[root] = (prio, dict(root=root, leaves=leaves, leaves_compl=leaves_compl, cand=c))

    # Materialize ReplacementOps only for winning root candidates (verifying truth & bounded MFFC)
    queue: list[tuple[tuple, ReplacementOp]] = []
    for root, (prio, info) in best_per_root.items():
        leaves = info["leaves"]
        cand = info["cand"]
        rem_all, rem_strict = bounded_mffc_nodes(root, leaves, aig, fanouts)
        tt = original_cone_truth(aig, root, leaves)
        flip = candidate_output_complement(tt, info["leaves_compl"], cand)
        op = ReplacementOp(root, tuple(leaves), tuple(info["leaves_compl"]), cand, rem_all, rem_strict, flip)
        queue.append((prio, op))
    return select_nonconflicting(queue)


def select_4b(aig, levels, fanouts, sens: list[float], cuts_path: Path,
              lam_area: float, tau: float = 1.0) -> list[ReplacementOp]:
    """4B: Global PI->PO sensitivity * convex smooth arrival reduction - lambda * area_increase."""
    offset = aig.po_count
    to_aig = lambda nid: nid if nid <= aig.pi_count else nid - offset

    best_per_root: dict[int, tuple[tuple, dict]] = {}
    with cuts_path.open() as fh:
        for line in fh:
            rec = json.loads(line)
            cands = rec["candidates"]
            if not cands:
                continue
            root = to_aig(rec["node"])
            req = levels.required[root]
            if req is None:
                continue
            s_root = sens[root]
            leaves = [to_aig(lf["id"]) for lf in rec["leaves"]]
            leaves_compl = [int(lf["compl"]) for lf in rec["leaves"]]
            node_lv = levels.level[root]

            depths, _ = original_leaf_depths(aig, root, leaves, levels.level)
            orig_arr_rel = [levels.level[lf] + dp - node_lv for lf, dp in zip(leaves, depths)]
            j_orig = _lse(orig_arr_rel, tau)

            saved = rec["n_nodes_saved"]
            best_score = 0.0
            best_cand = None
            best_da = 0
            for c in cands:
                if c["self"]:
                    continue
                if c["root_level"] > req:
                    continue
                da = c["nodes_added"] - saved
                arr_rel = [levels.level[lf] + dp - node_lv for lf, dp in zip(leaves, c["leaf_depths"]) if dp >= 0]
                if not arr_rel:
                    # Constant node: arrival effectively 0 (rel = -node_lv)
                    j_cand = float(-node_lv)
                else:
                    j_cand = _lse(arr_rel, tau)
                delta_j = j_orig - j_cand
                if da <= 0:
                    # Area-saving or area-neutral: require non-degrading convex arrival
                    if delta_j < -1e-9:
                        continue
                    score = s_root * delta_j + 1e-5 * (-da) + 1e-8
                else:
                    score = s_root * delta_j - lam_area * da
                if score > best_score:
                    best_score = score
                    best_cand = c
                    best_da = da

            if best_cand is not None:
                prio = (best_score, -best_da, -root, -best_cand["subgraph"])
                prev = best_per_root.get(root)
                if prev is None or prio > prev[0]:
                    best_per_root[root] = (prio, dict(root=root, leaves=leaves, leaves_compl=leaves_compl, cand=best_cand))

    queue: list[tuple[tuple, ReplacementOp]] = []
    for root, (prio, info) in best_per_root.items():
        leaves = info["leaves"]
        cand = info["cand"]
        rem_all, rem_strict = bounded_mffc_nodes(root, leaves, aig, fanouts)
        tt = original_cone_truth(aig, root, leaves)
        flip = candidate_output_complement(tt, info["leaves_compl"], cand)
        op = ReplacementOp(root, tuple(leaves), tuple(info["leaves_compl"]), cand, rem_all, rem_strict, flip)
        queue.append((prio, op))
    return select_nonconflicting(queue)


def run_single_arm(name: str, aig_orig_path: Path, orig_aig, orig_levels, ref_depth: int,
                   cuts_path: Path, work_dir: Path, arm: str) -> dict:
    t0 = time.perf_counter()
    work_dir.mkdir(parents=True, exist_ok=True)
    if arm in ("4A_tau1_all", "4A_tau1_gated", "4A_uniform_gated"):
        scheme = "uniform" if arm == "4A_uniform_gated" else "tau1"
        gate = None if arm == "4A_tau1_all" else 2
        fanouts = references(orig_aig)
        ops = select_4a(orig_aig, orig_levels, fanouts, cuts_path, scheme, gate)
        out_aig_path = work_dir / f"{name}.{arm}.aig"
        if ops:
            new_aig = apply_and_write_aig(orig_aig, ops, out_aig_path)
            cec_ok = verify_cec(ABC_ORIG, aig_orig_path, out_aig_path, work_dir / f"{arm}.cec.log")
            new_lv = propagate_levels(new_aig)
            m = output_tns_metrics(new_aig, new_lv, ref_depth)
        else:
            cec_ok = True
            m = output_tns_metrics(orig_aig, orig_levels, ref_depth)
        return {
            "arm": arm,
            "rounds": [{"round": 1, "applied_ops": len(ops), **m, "cec": cec_ok}],
            "applied_total": len(ops),
            "cec": cec_ok,
            "runtime_s": time.perf_counter() - t0,
            **m,
            "delta_depth": m["depth"] - orig_levels.depth,
            "delta_ands": m["ands"] - orig_aig.and_count,
        }

    # 4B variants
    rounds = 1 if arm.startswith("4B_r1") else 3
    lam = 0.001 if arm.endswith("_free") else 0.05
    cur_aig = orig_aig
    cur_lv = orig_levels
    cur_aig_path = aig_orig_path
    cur_cuts = cuts_path
    round_logs = []
    applied_total = 0
    cec_all = True

    for rnd in range(1, rounds + 1):
        if rnd > 1:
            cur_cuts = run_abc_dump(
                ABC_DUMP, cur_aig_path, work_dir / f"round{rnd}_dump", work_dir / f"round{rnd}_dump.log"
            )
        fanouts = references(cur_aig)
        _, sens = global_po_sensitivity(cur_aig, cur_lv, tau=1.0)
        ops = select_4b(cur_aig, cur_lv, fanouts, sens, cur_cuts, lam_area=lam, tau=1.0)
        if not ops:
            m = output_tns_metrics(cur_aig, cur_lv, ref_depth)
            round_logs.append({"round": rnd, "applied_ops": 0, **m, "cec": True})
            break
        next_aig_path = work_dir / f"{name}.{arm}.r{rnd}.aig"
        next_aig = apply_and_write_aig(cur_aig, ops, next_aig_path)
        cec_ok = verify_cec(ABC_ORIG, aig_orig_path, next_aig_path, work_dir / f"{arm}.r{rnd}.cec.log")
        cec_all = cec_all and cec_ok
        next_lv = propagate_levels(next_aig)
        m = output_tns_metrics(next_aig, next_lv, ref_depth)
        applied_total += len(ops)
        round_logs.append({"round": rnd, "applied_ops": len(ops), **m, "cec": cec_ok})
        cur_aig, cur_lv, cur_aig_path = next_aig, next_lv, next_aig_path
        if not cec_ok:
            break

    final_m = output_tns_metrics(cur_aig, cur_lv, ref_depth)
    return {
        "arm": arm,
        "rounds": round_logs,
        "applied_total": applied_total,
        "cec": cec_all,
        "runtime_s": time.perf_counter() - t0,
        **final_m,
        "delta_depth": final_m["depth"] - orig_levels.depth,
        "delta_ands": final_m["ands"] - orig_aig.and_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    if not os.getenv("SLURM_JOB_ID"):
        raise SystemExit("Must run inside a Slurm job")

    repo_root = run / "src"
    out_dir = run / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    pilots = load_pilot_inputs()
    native = load_native_baselines(repo_root)
    arms = [
        "4A_tau1_all",
        "4A_tau1_gated",
        "4A_uniform_gated",
        "4B_r1_free",
        "4B_r1_area",
        "4B_iter3_free",
        "4B_iter3_area",
    ]

    summary = {"slurm_job_id": os.getenv("SLURM_JOB_ID"), "pilots": {}}
    for name in PILOTS:
        aig_path, cuts_path = pilots[name]
        orig_aig = parse_aig(aig_path)
        orig_lv = propagate_levels(orig_aig)
        orig_m = output_tns_metrics(orig_aig, orig_lv, orig_lv.depth)
        print(f"=== {name}: orig D={orig_lv.depth}, AND={orig_aig.and_count}, TNS={orig_m['tns']} ===", flush=True)
        c_res = {"orig": orig_m, "native": native[name], "arms": {}}
        for arm in arms:
            res = run_single_arm(name, aig_path, orig_aig, orig_lv, orig_lv.depth, cuts_path, out_dir / name / arm, arm)
            c_res["arms"][arm] = res
            print(
                f"  {arm:18s}: D={res['depth']:3d} (dD={res['delta_depth']:+3d}), "
                f"AND={res['ands']:5d} (dA={res['delta_ands']:+6d}), "
                f"TNS1/2/3={res['tns'][1]}/{res['tns'][2]}/{res['tns'][3]}, "
                f"ops={res['applied_total']:5d}, cec={res['cec']}, t={res['runtime_s']:.1f}s",
                flush=True,
            )
        summary["pilots"][name] = c_res

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print("DONE step4_pilot_replay", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
