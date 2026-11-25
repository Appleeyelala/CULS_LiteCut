"""Review-optimized 4B selector (v4).

Feature switches (with every switch off and a legacy `DelayProfile` this reproduces
`scripts/step4_fanout_pilot.select_4b_fanout` bit-for-bit):

- `strash_aware`: candidate gates whose literal pair hashes onto an existing AIG node (as the
  strashing replay builder will do) reuse that node -- its real arrival and its fanout increase are
  priced -- and only truly new gates add leaf references.  Removing the bounded MFFC also releases
  references on shared (non-MFFC) interior nodes.
- PI phase-aware inverter load and slew-coupled gate delay come from the `DelayProfile`
  (`propagate_fanout_delays(..., slew_beta, pi_phase_aware)`); the selector prices the slew
  cross-term of a node's fanout change on its downstream gates.
- `per_root_k`: keep the best option of up to `k` distinct cuts per root for conflict resolution.
- `fanout_budget`: (a) a candidate may not raise a touched node's delay beyond the slack of that
  node's *other* consumers, and (b) the greedy batch tracks live fanout deltas of accepted ops:
  cumulative increases must stay within those slacks, the root arrival check includes the other
  ops' leaf delay increases, and the quadratic fanout regularizer is re-priced on live fanouts.
"""
from __future__ import annotations

import heapq
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .aig_replay import (
    DelayProfile,
    ReplacementOp,
    bounded_mffc_nodes,
    candidate_output_complement,
    original_cone_context,
    original_cone_truth,
    select_nonconflicting,
    slew_coupling_coefficients,
)


@dataclass(frozen=True)
class Select4BConfig:
    lam_area: float = 0.05
    fanout_alpha: float = 0.20
    pi_fanout_alpha: float = 0.0
    fanout_reg: float = 0.0005
    fanout_pivot: int = 6
    allow_slack_area: bool = False
    strash_aware: bool = False
    per_root_k: int = 1
    fanout_budget: bool = False
    budget_tol: float = 0.0


@dataclass
class _Option:
    score: float
    da: int
    cand: dict
    max_arr: float
    req: float
    touched: tuple  # ((key, dfo, coef, base_count, slack_other), ...)
    inputs: tuple  # ((key, coef, base_count), ...) cone inputs whose arrival feeds the candidate


def _lse(vals: list[float], tau: float) -> float:
    m = max(vals)
    return m + tau * math.log(sum(math.exp((v - m) / tau) for v in vals))


def _load(x: int) -> int:
    return x - 1 if x > 1 else 0


def build_strash_table(aig) -> dict[tuple[int, int], int]:
    table: dict[tuple[int, int], int] = {}
    for node in range(aig.pi_count + 1, aig.maximum_variable + 1):
        l0, l1 = aig.fanin_literals[node]
        key = (l0, l1) if l0 < l1 else (l1, l0)
        table.setdefault(key, node)
    return table


def _consumer_bounds(aig, delay: DelayProfile, phase: bool) -> dict[tuple[int, int], list[tuple[float, int]]]:
    """Three tightest consumer input-required bounds per (node, phase) key; PO consumers use id -1."""
    pi_count = aig.pi_count
    raw: dict[tuple[int, int], list[tuple[float, int]]] = defaultdict(list)

    def _push(lit: int, bound: float, g: int) -> None:
        n = lit >> 1
        if n == 0:
            return
        if n <= pi_count and phase:
            if not (lit & 1):
                return
            key = (n, 1)
        else:
            key = (n, 0)
        lst = raw[key]
        lst.append((bound, g))
        if len(lst) > 6:
            lst.sort()
            del lst[3:]

    for g in range(pi_count + 1, aig.maximum_variable + 1):
        req = delay.required[g]
        if req is None:
            continue
        b = req - delay.gate_delay[g]
        for lit in aig.fanin_literals[g]:
            _push(lit, b, g)
    for lit in aig.po_literals:
        _push(lit, delay.depth, -1)
    out = {}
    for key, lst in raw.items():
        lst.sort()
        out[key] = lst[:3]
    return out


def select_4b_v4(
    aig,
    delay: DelayProfile,
    sens: list[float],
    cuts_path: Path,
    tau: float,
    cfg: Select4BConfig,
    stats: dict | None = None,
) -> list[ReplacementOp]:
    pi_count = aig.pi_count
    offset = aig.po_count
    to_aig = lambda nid: nid if nid <= pi_count else nid - offset  # noqa: E731

    fo = delay.fanouts
    phase = bool(delay.pi_phase_aware)
    neg_fo = delay.neg_fanouts if phase else None
    beta = delay.slew_beta
    alpha = cfg.fanout_alpha
    pa = cfg.pi_fanout_alpha
    reg_w = cfg.fanout_reg
    pivot = cfg.fanout_pivot
    level = delay.level
    neg_extra = delay.neg_extra
    strash = build_strash_table(aig) if cfg.strash_aware else None
    if beta > 0.0:
        slew_coef, slew_coef_neg = slew_coupling_coefficients(aig, delay, sens)
    else:
        slew_coef = slew_coef_neg = None
    bounds = _consumer_bounds(aig, delay, phase) if cfg.fanout_budget else None
    st = stats if stats is not None else {}
    for k in ("cuts", "options", "strash_hits", "budget_rejects_eval", "budget_rejects_batch", "requeued"):
        st.setdefault(k, 0)

    def _key(n: int, compl: int):
        if n <= pi_count and phase:
            return (n, 1) if compl else None
        return (n, 0)

    def _coef(key) -> float:
        n = key[0]
        if n <= pi_count:
            return pa
        if beta > 0.0:
            return alpha * (1.0 + beta * delay.slew_in[n])
        return alpha

    def _base(key) -> int:
        n, neg = key
        return neg_fo[n] if neg else fo[n]

    def _slack_other(key, exclude: frozenset[int]) -> float | None:
        lst = bounds.get(key)
        if not lst:
            return None
        n, neg = key
        arr = level[n] + (neg_extra[n] if (neg and neg_extra is not None) else 0.0)
        for b, g in lst:
            if g not in exclude:
                return b - arr
        if len(lst) >= 3:
            return lst[0][0] - arr  # conservative fallback
        return None

    per_root: dict[int, list] = {}
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
            st["cuts"] += 1
            s_root = sens[root]
            leaves = [to_aig(lf["id"]) for lf in rec["leaves"]]
            leaves_compl = [int(lf["compl"]) for lf in rec["leaves"]]
            node_lv = level[root]

            rem_all, rem_strict = bounded_mffc_nodes(root, leaves, aig, fo)
            ctx = original_cone_context(aig, root, leaves, delay, rem_all, sens, tau=tau)
            orig_arr_rel = [level[lf] + dp - node_lv for lf, dp in zip(leaves, ctx.leaf_delays) if dp is not None]
            j_orig = _lse(orig_arr_rel, tau) if orig_arr_rel else float(-node_lv)
            saved = rec["n_nodes_saved"]
            root_fo = fo[root]
            leaf_idx = {lf: i for i, lf in enumerate(leaves)}

            base_touch: dict = defaultdict(int)
            for i, lf in enumerate(leaves):
                if phase and lf <= pi_count:
                    if ctx.orig_refs_neg[i]:
                        base_touch[(lf, 1)] -= ctx.orig_refs_neg[i]
                elif ctx.orig_refs[i]:
                    base_touch[(lf, 0)] -= ctx.orig_refs[i]
            if cfg.strash_aware:
                for node in rem_all:
                    for lit in aig.fanin_literals[node]:
                        src = lit >> 1
                        if src in leaf_idx or src in rem_all or src == 0:
                            continue
                        key = _key(src, lit & 1)
                        if key is not None:
                            base_touch[key] -= 1
            rem_slew: dict = defaultdict(float)
            if slew_coef is not None:
                ab = alpha * beta
                for g in rem_all:
                    arg = delay.slew_arg[g]
                    if arg < 0:
                        continue
                    lit = aig.fanin_literals[g][arg]
                    key = _key(lit >> 1, lit & 1) if (lit >> 1) > 0 else None
                    if key is not None:
                        rem_slew[key] += sens[g] * ab * _load(fo[g])

            def _sens_other(n: int) -> float:
                idx = leaf_idx.get(n)
                if idx is not None:
                    return ctx.sens_other[idx]
                return max(0.0, sens[n] - ctx.local_sens.get(n, 0.0))

            orig_tt = None

            def _flip(c) -> int:
                nonlocal orig_tt
                if orig_tt is None:
                    orig_tt = original_cone_truth(aig, root, leaves)
                return candidate_output_complement(orig_tt, leaves_compl, c)

            def _merge_root(touched: dict, lit: int, c) -> None:
                """Root consumers move onto literal `lit` (wire, alias or strash merge)."""
                n = lit >> 1
                if n == 0:
                    return
                if phase and n <= pi_count:
                    p = (lit & 1) ^ c["root_compl"] ^ _flip(c)
                    rneg = neg_fo[root]
                    rpos = root_fo - rneg
                    moved = rneg if p == 0 else rpos
                    if moved:
                        touched[(n, 1)] += moved
                else:
                    touched[(n, 0)] += root_fo

            best_opt = None
            best_score = 0.0
            for c in cands:
                if c["self"]:
                    continue
                touched = defaultdict(int, base_touch)
                edges = c["edges"]
                rn = c["root_node"]
                gate_lit: dict[int, int | None] = {}
                hit_node: dict[int, int] = {}
                gate_in: dict[int, tuple[int, int]] = {}
                if c["n_size"] == 0:
                    kind = "const"
                elif not edges:
                    kind = "wire"
                    _merge_root(touched, (leaves[rn] << 1) ^ leaves_compl[rn], c)
                else:
                    kind = "net"
                    e_fo = [0] * (4 + len(edges))
                    lits = [(leaves[i] << 1) ^ leaves_compl[i] for i in range(4)]

                    def _lit_of(en: int, ec: int):
                        if en < 4:
                            return lits[en] ^ ec
                        gl = gate_lit.get(en)
                        return None if gl is None else gl ^ ec

                    for idx, (e0n, e0c, e1n, e1c) in enumerate(edges, start=4):
                        reused = None
                        if strash is not None:
                            in0, in1 = _lit_of(e0n, e0c), _lit_of(e1n, e1c)
                            if in0 is not None and in1 is not None:
                                if in0 == in1:
                                    reused = in0
                                elif in0 == (in1 ^ 1) or in0 == 0 or in1 == 0:
                                    reused = 0
                                elif in0 == 1:
                                    reused = in1
                                elif in1 == 1:
                                    reused = in0
                                else:
                                    u = strash.get((in0, in1) if in0 < in1 else (in1, in0))
                                    if u is not None and u != root and u not in rem_strict:
                                        reused = u << 1
                                        hit_node[idx] = u
                                        gate_in[idx] = (in0, in1)
                                        st["strash_hits"] += 1
                        if reused is not None:
                            gate_lit[idx] = reused
                            continue
                        gate_lit[idx] = None
                        for en, ec in ((e0n, e0c), (e1n, e1c)):
                            if en < 4:
                                key = _key(leaves[en], leaves_compl[en] ^ ec)
                                if key is not None:
                                    touched[key] += 1
                            elif gate_lit.get(en) is not None:
                                gl = gate_lit[en] ^ ec
                                key = _key(gl >> 1, gl & 1) if (gl >> 1) > 0 else None
                                if key is not None:
                                    touched[key] += 1
                            else:
                                e_fo[en] += 1
                    if gate_lit.get(rn) is not None:
                        _merge_root(touched, gate_lit[rn], c)

                # ---- fanout change costs -------------------------------------------------
                side = 0.0
                reg = 0.0
                slew_cost = 0.0
                dd_map: dict = {}
                dl_map: dict = {}
                touched_list = []
                budget_ok = True
                for key, d in touched.items():
                    if d == 0:
                        continue
                    coef = _coef(key)
                    if coef <= 0.0:
                        continue
                    base_cnt = _base(key)
                    new_cnt = max(0, base_cnt + d)
                    dl = _load(new_cnt) - _load(base_cnt)
                    dd = coef * dl
                    dd_map[key] = dd
                    dl_map[key] = dl
                    so = _sens_other(key[0])
                    side += so * dd
                    if reg_w > 0.0:
                        hn = max(0, new_cnt - pivot)
                        ho = max(0, base_cnt - pivot)
                        reg += reg_w * (hn * hn - ho * ho)
                    if slew_coef is not None and dl:
                        sc = (slew_coef_neg[key[0]] if key[1] else slew_coef[key[0]]) - rem_slew.get(key, 0.0)
                        if sc > 0.0:
                            slew_cost += sc * dl
                    slack_o = None
                    if bounds is not None:
                        slack_o = _slack_other(key, rem_all)
                        if dd > 1e-12 and slack_o is not None and dd > max(0.0, slack_o) + cfg.budget_tol:
                            budget_ok = False
                            break
                    touched_list.append((key, d, coef, base_cnt, slack_o))
                if not budget_ok:
                    st["budget_rejects_eval"] += 1
                    continue

                # ---- candidate timing ----------------------------------------------------
                def _arr_node(n: int, compl: int) -> float:
                    if n == 0:
                        return 0.0
                    if n <= pi_count:
                        if phase:
                            return level[n] + ((neg_extra[n] + dd_map.get((n, 1), 0.0)) if compl else 0.0)
                        return level[n] + dd_map.get((n, 0), 0.0)
                    return level[n] + dd_map.get((n, 0), 0.0)

                def _slew_node(n: int, compl: int) -> float:
                    if beta <= 0.0 or n == 0:
                        return 0.0
                    if n <= pi_count:
                        if phase and compl:
                            return delay.neg_slew[n] + dl_map.get((n, 1), 0)
                        return 0.0
                    return delay.out_slew[n] + dl_map.get((n, 0), 0)

                inputs = []
                if kind == "const":
                    max_arr_abs, j_cand = 0.0, float(-node_lv)
                elif kind == "wire":
                    lf = leaves[rn]
                    p = 0
                    if phase and lf <= pi_count:
                        p = leaves_compl[rn] ^ c["root_compl"] ^ _flip(c)
                    max_arr_abs = _arr_node(lf, p)
                    j_cand = max_arr_abs - node_lv
                    k_in = _key(lf, p)
                    if k_in is not None:
                        inputs.append((k_in, _coef(k_in), _base(k_in)))
                else:
                    cv_max: dict[int, float] = {}
                    cv_lse: dict[int, float] = {}
                    cv_slew: dict[int, float] = {}

                    def _in(en: int, ec: int):
                        if en < 4:
                            lf = leaves[en]
                            ph = leaves_compl[en] ^ ec
                            a = _arr_node(lf, ph)
                            return a, a - node_lv, _slew_node(lf, ph)
                        gl = gate_lit.get(en)
                        if gl is not None and en not in hit_node:
                            gl ^= ec
                            a = _arr_node(gl >> 1, gl & 1)
                            return a, a - node_lv, _slew_node(gl >> 1, gl & 1)
                        return cv_max[en], cv_lse[en], cv_slew[en]

                    seen_inputs: set = set()
                    for idx, (e0n, e0c, e1n, e1c) in enumerate(edges, start=4):
                        gl = gate_lit.get(idx)
                        if gl is not None and idx not in hit_node:
                            continue  # alias / constant: resolved at use time
                        for en, ec in ((e0n, e0c), (e1n, e1c)):
                            if en < 4:
                                k_in = _key(leaves[en], leaves_compl[en] ^ ec)
                                if k_in is not None and k_in[0] > 0 and k_in not in seen_inputs:
                                    seen_inputs.add(k_in)
                                    inputs.append((k_in, _coef(k_in), _base(k_in)))
                            else:
                                g_en = gate_lit.get(en)
                                if g_en is not None and en not in hit_node:
                                    g_en ^= ec
                                    k_in = _key(g_en >> 1, g_en & 1) if (g_en >> 1) > 0 else None
                                    if k_in is not None and k_in not in seen_inputs:
                                        seen_inputs.add(k_in)
                                        inputs.append((k_in, _coef(k_in), _base(k_in)))
                        a0, r0, s0 = _in(e0n, e0c)
                        a1, r1, s1 = _in(e1n, e1c)
                        s_in = s0 if s0 >= s1 else s1
                        if idx in hit_node:
                            u = hit_node[idx]
                            k_u = (u, 0)
                            dl_u = dl_map.get(k_u, 0)
                            if beta > 0.0:
                                gd = 1.0 + alpha * (_load(fo[u]) + dl_u) * (1.0 + beta * s_in)
                            else:
                                gd = delay.gate_delay[u] + dd_map.get(k_u, 0.0)
                            sl = float(_load(fo[u]) + dl_u)
                            if k_u not in seen_inputs:
                                seen_inputs.add(k_u)
                                inputs.append((k_u, _coef(k_u), _base(k_u)))
                        elif idx == rn:
                            load = _load(root_fo)
                            gd = 1.0 + alpha * load * (1.0 + beta * s_in) if beta > 0.0 else delay.gate_delay[root]
                            sl = float(load)
                        else:
                            load = _load(e_fo[idx])
                            gd = 1.0 + alpha * load * (1.0 + beta * s_in) if beta > 0.0 else 1.0 + alpha * load
                            sl = float(load)
                        cv_max[idx] = gd + (a0 if a0 >= a1 else a1)
                        cv_lse[idx] = gd + _lse([r0, r1], tau)
                        cv_slew[idx] = sl
                    if rn in cv_max:
                        max_arr_abs, j_cand = cv_max[rn], cv_lse[rn]
                    else:
                        gl = gate_lit[rn]
                        n_gl = gl >> 1
                        p_gl = (gl & 1) ^ c["root_compl"] ^ _flip(c) if (phase and 0 < n_gl <= pi_count) else (gl & 1)
                        max_arr_abs = _arr_node(n_gl, p_gl)
                        j_cand = max_arr_abs - node_lv
                        k_in = _key(n_gl, p_gl) if n_gl > 0 else None
                        if k_in is not None and k_in not in seen_inputs:
                            seen_inputs.add(k_in)
                            inputs.append((k_in, _coef(k_in), _base(k_in)))
                if max_arr_abs > req + 1e-6:
                    continue

                da = c["nodes_added"] - saved
                delta_j = j_orig - j_cand
                eff_delta = s_root * delta_j - side - reg - slew_cost
                if cfg.allow_slack_area:
                    if da < 0:
                        score = eff_delta + cfg.lam_area * (-da)
                    elif da == 0:
                        if eff_delta <= 1e-9:
                            continue
                        score = eff_delta
                    else:
                        score = eff_delta - cfg.lam_area * da
                elif da <= 0:
                    if delta_j < -1e-9 or eff_delta < -1e-9:
                        continue
                    score = eff_delta + 1e-5 * (-da) + 1e-8
                else:
                    score = eff_delta - cfg.lam_area * da
                if score > best_score:
                    best_score = score
                    best_opt = _Option(score, da, c, max_arr_abs, req, tuple(touched_list), tuple(inputs))

            if best_opt is not None:
                st["options"] += 1
                prio = (best_opt.score, -best_opt.da, -root, -best_opt.cand["subgraph"])
                lst = per_root.setdefault(root, [])
                lst.append((prio, best_opt, leaves, leaves_compl, rem_all, rem_strict, orig_tt))
                if len(lst) > cfg.per_root_k:
                    lst.sort(key=lambda x: x[0], reverse=True)
                    del lst[cfg.per_root_k:]

    entries = []
    for root, lst in per_root.items():
        for prio, opt, leaves, leaves_compl, rem_all, rem_strict, tt in lst:
            if tt is None:
                tt = original_cone_truth(aig, root, leaves)
            flip = candidate_output_complement(tt, leaves_compl, opt.cand)
            op = ReplacementOp(root, tuple(leaves), tuple(leaves_compl), opt.cand, rem_all, rem_strict, flip)
            entries.append((prio, op, opt))
    if not cfg.fanout_budget:
        return select_nonconflicting([(prio, op) for prio, op, _ in entries])
    return _select_with_budget(entries, cfg, st)


def _select_with_budget(entries, cfg: Select4BConfig, st: dict) -> list[ReplacementOp]:
    """Lazy greedy non-conflicting selection with live fanout accounting (see module docstring)."""
    heap = []
    for idx, (prio, _op, _opt) in enumerate(entries):
        heap.append((-prio[0], -prio[1], -prio[2], -prio[3], idx))
    heapq.heapify(heap)
    acc: dict = defaultdict(int)
    used_roots: set[int] = set()
    deleted_strict: set[int] = set()
    protected: set[int] = set()
    selected: list[ReplacementOp] = []
    requeue_cnt: dict[int, int] = defaultdict(int)
    reg_w, pivot = cfg.fanout_reg, cfg.fanout_pivot

    def _hq(x: int) -> int:
        h = x - pivot
        return h * h if h > 0 else 0

    while heap:
        neg_score, a, b, c, idx = heapq.heappop(heap)
        prio, op, opt = entries[idx]
        if op.root in used_roots or op.root in deleted_strict:
            continue
        if op.rem_strict & protected or op.rem_all & deleted_strict or set(op.leaves) & deleted_strict:
            continue
        ok = True
        extra_reg = 0.0
        for key, d, coef, base, slack_o in opt.touched:
            a_ = acc.get(key, 0)
            if a_ and reg_w > 0.0:
                extra_reg += reg_w * ((_hq(base + a_ + d) - _hq(base + a_)) - (_hq(base + d) - _hq(base)))
            if d > 0 and slack_o is not None:
                inc = coef * (_load(max(0, base + a_ + d)) - _load(base))
                if inc > 1e-12 and inc > max(0.0, slack_o) + cfg.budget_tol:
                    ok = False
                    break
        if ok:
            arr_inc = 0.0
            for key, coef, base in opt.inputs:
                a_ = acc.get(key, 0)
                if a_ > 0:
                    inc = coef * (_load(base + a_) - _load(base))
                    if inc > arr_inc:
                        arr_inc = inc
            if opt.max_arr + arr_inc > opt.req + 1e-6:
                ok = False
        if not ok:
            st["budget_rejects_batch"] += 1
            continue
        live_score = prio[0] - extra_reg
        if live_score <= 0.0:
            st["budget_rejects_batch"] += 1
            continue
        if requeue_cnt[idx] < 4 and heap and live_score < -heap[0][0] - 1e-12:
            requeue_cnt[idx] += 1
            st["requeued"] += 1
            heapq.heappush(heap, (-live_score, a, b, c, idx))
            continue
        selected.append(op)
        used_roots.add(op.root)
        deleted_strict.update(op.rem_strict)
        protected.update(op.rem_all)
        protected.update(op.leaves)
        for key, d, _coef, _base, _s in opt.touched:
            acc[key] += d
    return selected
