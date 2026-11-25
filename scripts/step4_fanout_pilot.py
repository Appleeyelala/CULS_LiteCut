#!/usr/bin/env python3
"""Pilot experiment for fanout-aware 4B global PI->PO AIG rewriting and ASAP7 mapping evaluation."""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from global_timing.aig_replay import (
    DelayProfile,
    ReplacementOp,
    apply_and_write_aig,
    bounded_mffc_nodes,
    candidate_output_complement,
    global_po_sensitivity,
    original_cone_timing_and_refs,
    original_cone_truth,
    output_tns_metrics,
    propagate_fanout_delays,
    verify_cec,
)
from global_timing.evaluation import Evaluator
from global_timing.level_weights import propagate_levels
from global_timing.structure import parse_aig, references
from scripts.step4_pilot_replay import (
    ABC_DUMP,
    ABC_ORIG,
    PILOTS,
    _lse,
    load_pilot_inputs,
    run_abc_dump,
)

ABC_EVAL = Path("/data/wushangge/CULS_LiteCut/runs/20261111T173355Z-acyclic-replay-tool-build-64cf764e/build/framework_tools/abc/abc")
LIB_ASAP7 = Path("/data/wushangge/CULS_LiteCut/assets/libraries/asap7.lib")
PREV_MAP_SUMMARY = Path("/data/wushangge/CULS_LiteCut/runs/20261124T085044Z-step4-pilot-map-109c4238/output/summary.json")


def _cand_timing(
    delay: DelayProfile,
    root: int,
    leaves: list[int],
    cand: dict,
    leaf_delta_delay: list[float],
    pi_count: int,
    fanout_alpha: float,
    tau: float,
) -> tuple[float, float]:
    """Return `(max_abs_arrival_at_root, smooth_rel_arrival_at_root)` for `cand`."""
    node_lv = delay.level[root]
    edges = cand["edges"]
    if cand["n_size"] == 0:
        return 0.0, float(-node_lv)
    if not edges:
        rn = cand["root_node"]
        arr_abs = delay.level[leaves[rn]] + leaf_delta_delay[rn]
        return arr_abs, arr_abs - node_lv

    k = len(edges)
    e_fo = [0] * (4 + k)
    for e0n, _, e1n, _ in edges:
        e_fo[e0n] += 1
        e_fo[e1n] += 1

    cv_max = [delay.level[lf] + leaf_delta_delay[i] for i, lf in enumerate(leaves)]
    cv_lse = [a - node_lv for a in cv_max]
    rn = cand["root_node"]
    for idx_gate, (e0n, _, e1n, _) in enumerate(edges, start=4):
        if idx_gate == rn:
            gd = delay.gate_delay[root]
        else:
            gd = 1.0 + fanout_alpha * max(0, e_fo[idx_gate] - 1)
        a0, a1 = cv_max[e0n], cv_max[e1n]
        cv_max.append(gd + (a0 if a0 >= a1 else a1))
        cv_lse.append(gd + _lse([cv_lse[e0n], cv_lse[e1n]], tau))
    return cv_max[rn], cv_lse[rn]


def select_4b_fanout(
    aig,
    delay: DelayProfile,
    sens: list[float],
    cuts_path: Path,
    lam_area: float,
    fanout_alpha: float,
    fanout_reg: float = 0.0005,
    fanout_pivot: int = 6,
    tau: float = 1.0,
) -> list[ReplacementOp]:
    """Fanout-aware 4B: PI->PO global sensitivity + leaf fanout reaction + wire-dump guard."""
    from global_timing.aig_replay import select_nonconflicting

    offset = aig.po_count
    to_aig = lambda nid: nid if nid <= aig.pi_count else nid - offset

    best_per_root: dict[int, tuple[tuple, dict]] = {}
    with cuts_path.open() as fh:
        for line in fh:
            rec = json.loads(line)
            cands = rec["candidates"]
            if not cands or not any(not c["self"] for c in cands):
                continue
            root = to_aig(rec["node"])
            req = delay.required[root]
            if req is None:
                continue
            s_root = sens[root]
            leaves = [to_aig(lf["id"]) for lf in rec["leaves"]]
            leaves_compl = [int(lf["compl"]) for lf in rec["leaves"]]
            node_lv = delay.level[root]

            rem_all, rem_strict = bounded_mffc_nodes(root, leaves, aig, delay.fanouts)
            leaf_delays, orig_refs, sens_other = original_cone_timing_and_refs(
                aig, root, leaves, delay, rem_all, sens, tau=tau
            )
            orig_arr_rel = [delay.level[lf] + dp - node_lv for lf, dp in zip(leaves, leaf_delays) if dp is not None]
            j_orig = _lse(orig_arr_rel, tau) if orig_arr_rel else float(-node_lv)

            saved = rec["n_nodes_saved"]
            best_score = 0.0
            best_cand = None
            best_da = 0
            root_fo = delay.fanouts[root]

            for c in cands:
                if c["self"]:
                    continue
                if c["n_size"] == 0:
                    delta_fo = [-r for r in orig_refs]
                elif not c["edges"]:
                    rn = c["root_node"]
                    delta_fo = [(root_fo if i == rn else 0) - orig_refs[i] for i in range(4)]
                else:
                    cand_refs = [0, 0, 0, 0]
                    for e0n, _, e1n, _ in c["edges"]:
                        if e0n < 4:
                            cand_refs[e0n] += 1
                        if e1n < 4:
                            cand_refs[e1n] += 1
                    delta_fo = [cand_refs[i] - orig_refs[i] for i in range(4)]

                leaf_delta_delay = [0.0, 0.0, 0.0, 0.0]
                fanout_side_cost = 0.0
                fanout_reg_cost = 0.0
                for i, lf in enumerate(leaves):
                    dfo = delta_fo[i]
                    if dfo == 0 or lf <= aig.pi_count:
                        continue
                    old_fo = delay.fanouts[lf]
                    new_fo = max(0, old_fo + dfo)
                    dd = fanout_alpha * (max(0, new_fo - 1) - max(0, old_fo - 1))
                    leaf_delta_delay[i] = dd
                    fanout_side_cost += sens_other[i] * dd
                    if fanout_reg > 0.0:
                        h_new = max(0, new_fo - fanout_pivot)
                        h_old = max(0, old_fo - fanout_pivot)
                        fanout_reg_cost += fanout_reg * (h_new * h_new - h_old * h_old)

                max_arr_abs, j_cand = _cand_timing(
                    delay, root, leaves, c, leaf_delta_delay, aig.pi_count, fanout_alpha, tau
                )
                if max_arr_abs > req + 1e-6:
                    continue

                da = c["nodes_added"] - saved
                delta_j = j_orig - j_cand
                eff_delta = s_root * delta_j - fanout_side_cost - fanout_reg_cost
                if da <= 0:
                    if delta_j < -1e-9 or eff_delta < -1e-9:
                        continue
                    score = eff_delta + 1e-5 * (-da) + 1e-8
                else:
                    score = eff_delta - lam_area * da

                if score > best_score:
                    best_score = score
                    best_cand = c
                    best_da = da

            if best_cand is not None:
                prio = (best_score, -best_da, -root, -best_cand["subgraph"])
                prev = best_per_root.get(root)
                if prev is None or prio > prev[0]:
                    best_per_root[root] = (
                        prio,
                        dict(
                            root=root,
                            leaves=leaves,
                            leaves_compl=leaves_compl,
                            cand=best_cand,
                            rem_all=rem_all,
                            rem_strict=rem_strict,
                        ),
                    )

    queue: list[tuple[tuple, ReplacementOp]] = []
    for root, (prio, info) in best_per_root.items():
        tt = original_cone_truth(aig, root, info["leaves"])
        flip = candidate_output_complement(tt, info["leaves_compl"], info["cand"])
        op = ReplacementOp(
            root,
            tuple(info["leaves"]),
            tuple(info["leaves_compl"]),
            info["cand"],
            info["rem_all"],
            info["rem_strict"],
            flip,
        )
        queue.append((prio, op))
    return select_nonconflicting(queue)


def run_fanout_arm(
    name: str,
    orig_aig_path: Path,
    orig_aig,
    cuts_path: Path,
    work_dir: Path,
    arm: str,
    rounds: int,
    lam: float,
    fanout_alpha: float,
    fanout_reg: float,
) -> dict:
    t0 = time.perf_counter()
    work_dir.mkdir(parents=True, exist_ok=True)
    orig_lv = propagate_levels(orig_aig)

    cur_aig = orig_aig
    cur_aig_path = orig_aig_path
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
        delay = propagate_fanout_delays(cur_aig, fanouts=fanouts, fanout_alpha=fanout_alpha)
        _, sens = global_po_sensitivity(cur_aig, delay, tau=1.0)
        ops = select_4b_fanout(
            cur_aig,
            delay,
            sens,
            cur_cuts,
            lam_area=lam,
            fanout_alpha=fanout_alpha,
            fanout_reg=fanout_reg,
            tau=1.0,
        )
        if not ops:
            cur_lv = propagate_levels(cur_aig)
            m = output_tns_metrics(cur_aig, cur_lv, orig_lv.depth)
            round_logs.append({"round": rnd, "applied_ops": 0, **m, "max_fo": max(fanouts.values(), default=0), "cec": True})
            break
        next_aig_path = work_dir / f"{name}.{arm}.r{rnd}.aig"
        next_aig = apply_and_write_aig(cur_aig, ops, next_aig_path)
        cec_ok = verify_cec(ABC_ORIG, orig_aig_path, next_aig_path, work_dir / f"{arm}.r{rnd}.cec.log")
        cec_all = cec_all and cec_ok
        next_lv = propagate_levels(next_aig)
        next_fo = references(next_aig)
        m = output_tns_metrics(next_aig, next_lv, orig_lv.depth)
        applied_total += len(ops)
        round_logs.append({
            "round": rnd,
            "applied_ops": len(ops),
            **m,
            "max_fo": max((next_fo[n] for n in range(next_aig.pi_count + 1, next_aig.maximum_variable + 1)), default=0),
            "cec": cec_ok,
        })
        cur_aig, cur_aig_path = next_aig, next_aig_path
        if not cec_ok:
            break

    cur_lv = propagate_levels(cur_aig)
    cur_fo = references(cur_aig)
    final_m = output_tns_metrics(cur_aig, cur_lv, orig_lv.depth)
    return {
        "arm": arm,
        "rounds": round_logs,
        "applied_total": applied_total,
        "cec": cec_all,
        "final_aig": str(cur_aig_path),
        "max_and_fo": max((cur_fo[n] for n in range(cur_aig.pi_count + 1, cur_aig.maximum_variable + 1)), default=0),
        "replay_runtime_s": time.perf_counter() - t0,
        **final_m,
    }


def map_one(task: tuple[str, str, Path, Path, Path]) -> tuple[str, str, dict]:
    circuit, arm, orig_path, cand_path, work_dir = task
    ev = Evaluator(ABC_EVAL, LIB_ASAP7, orig_path, timeout=600)
    res = ev.evaluate(cand_path, work_dir)
    aig = parse_aig(cand_path)
    lv = propagate_levels(aig)
    return circuit, arm, {
        "status": res["status"],
        "cec": res["cec"],
        "aig_depth": lv.depth,
        "aig_ands": aig.and_count,
        "delay_ps": res["delay_ps"],
        "area": res["area"],
        "mapped_gates": res["mapped_gates"],
        "adp": res["adp"],
        "map_runtime_s": res["runtime_s"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    run = args.run.resolve()
    if not os.getenv("SLURM_JOB_ID"):
        raise SystemExit("Must run inside a Slurm job")

    out_dir = run / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    arm_specs = [
        ("4B_fo015_r1_area", 1, 0.05, 0.15, 0.0005),
        ("4B_fo015_iter3_area", 3, 0.05, 0.15, 0.0005),
        ("4B_fo025_r1_area", 1, 0.05, 0.25, 0.0005),
        ("4B_fo025_iter3_area", 3, 0.05, 0.25, 0.0005),
        ("4B_fo025_iter3_free", 3, 0.001, 0.25, 0.0005),
    ]

    pilots = load_pilot_inputs()
    prev_map = json.loads(PREV_MAP_SUMMARY.read_text())["pilots"]
    summary = {"slurm_job_id": os.getenv("SLURM_JOB_ID"), "pilots": {n: {"arms": {}} for n in PILOTS}}

    map_tasks = []
    for name in PILOTS:
        orig_p, cuts_p = pilots[name]
        orig_aig = parse_aig(orig_p)
        print(f"=== Replay {name} (orig AND={orig_aig.and_count}) ===", flush=True)
        for arm, rounds, lam, alpha, reg in arm_specs:
            r_res = run_fanout_arm(name, orig_p, orig_aig, cuts_p, out_dir / name / arm, arm, rounds, lam, alpha, reg)
            summary["pilots"][name]["arms"][arm] = r_res
            print(
                f"  {arm:22s}: D={r_res['depth']:3d} AND={r_res['ands']:5d} maxFO={r_res['max_and_fo']:4d} "
                f"ops={r_res['applied_total']:5d} cec={r_res['cec']} t={r_res['replay_runtime_s']:.1f}s",
                flush=True,
            )
            map_tasks.append((name, arm, orig_p, Path(r_res["final_aig"]), out_dir / name / arm / "map"))

    print("=== ASAP7 Mapping ===", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for name, arm, m_res in pool.map(map_one, map_tasks):
            summary["pilots"][name]["arms"][arm]["mapping"] = m_res
            orig_delay = prev_map[name]["orig"]["delay_ps"]
            resyn_delay = prev_map[name]["abc_resyn2"]["delay_ps"]
            old_r1 = prev_map[name]["4B_r1_area"]["delay_ps"]
            old_i3 = prev_map[name]["4B_iter3_area"]["delay_ps"]
            print(
                f"  {name:14s} {arm:22s}: delay={m_res['delay_ps']:9.3f} ps "
                f"(orig={orig_delay:.1f}, resyn2={resyn_delay:.1f}, old_r1={old_r1:.1f}, old_i3={old_i3:.1f}) "
                f"area={m_res['area']:9.1f} cec={m_res['cec']}",
                flush=True,
            )

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print("DONE step4_fanout_pilot", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
