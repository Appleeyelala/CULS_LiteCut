import math
import tempfile
import unittest
from pathlib import Path

from global_timing.aig_replay import (
    ReplacementOp,
    apply_and_write_aig,
    bounded_mffc_nodes,
    candidate_output_complement,
    compute_adaptive_tau,
    edge_arrival,
    global_po_sensitivity,
    original_cone_timing_and_refs,
    original_cone_truth,
    propagate_fanout_delays,
    select_nonconflicting,
    slew_coupling_coefficients,
    write_binary_aig,
)
from global_timing.select4b import Select4BConfig, select_4b_v4
from global_timing.level_weights import propagate_levels
from global_timing.structure import aig_structure_digest, parse_aig, references
from test_structure import fixture


def eval_aig_all_inputs(aig):
    """Exact truth table vector over all 2^pi_count input assignments for small test AIGs."""
    n_pi = aig.pi_count
    table = []
    for mask in range(1 << n_pi):
        vals = {0: 0}
        for i in range(n_pi):
            vals[i + 1] = (mask >> i) & 1
        for node in range(n_pi + 1, aig.maximum_variable + 1):
            l0, l1 = aig.fanin_literals[node]
            vals[node] = (vals[l0 >> 1] ^ (l0 & 1)) & (vals[l1 >> 1] ^ (l1 & 1))
        table.append(tuple(vals[lit >> 1] ^ (lit & 1) for lit in aig.po_literals))
    return table


class AigReplayTests(unittest.TestCase):
    def test_binary_aig_roundtrip(self):
        aig = fixture(4, [(2, 4), (6, 8), (10, 3)], [12, 13, 0, 1, 2])
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "rt.aig"
            write_binary_aig(p, aig.pi_count, list(aig.po_literals), dict(aig.fanin_literals))
            aig2 = parse_aig(p)
            self.assertEqual(aig2.pi_count, aig.pi_count)
            self.assertEqual(aig2.po_literals, aig.po_literals)
            self.assertEqual(aig2.fanin_literals, {k: tuple(sorted(v, reverse=True)) for k, v in aig.fanin_literals.items()})

    def test_cascaded_replacements_fcompl_and_wire_constant(self):
        # Node 5 = x1 & x2, Node 6 = x3 & x4, Node 7 = node5 & node6
        # Node 8 = node7 & x1, Node 9 = node7 & ~x1
        # Node 10 = ~node8 & ~node9 = ~( (node7&x1) | (node7&~x1) ) = ~node7
        aig = fixture(4, [(2, 4), (6, 8), (10, 12), (14, 2), (14, 3), (17, 19)], [20, 14, 15])
        orig_tt = eval_aig_all_inputs(aig)
        refs = references(aig)

        cand1 = {
            "subgraph": 0,
            "n_leaves": 4,
            "n_size": 7,
            "root_node": 6,
            "root_compl": 0,
            "edges": [[0, 0, 2, 0], [1, 0, 3, 0], [4, 0, 5, 0]],
        }
        leaves1 = [1, 2, 3, 4]
        rem_all1, rem_str1 = bounded_mffc_nodes(7, leaves1, aig, refs)
        self.assertEqual(rem_all1, {5, 6, 7})
        self.assertEqual(rem_str1, {5, 6})
        tt1 = original_cone_truth(aig, 7, leaves1)
        cand1_inv = dict(cand1, root_compl=1)
        flip1 = candidate_output_complement(tt1, [0, 0, 0, 0], cand1_inv)
        self.assertEqual(flip1, 1)
        op1 = ReplacementOp(7, tuple(leaves1), (0, 0, 0, 0), cand1_inv, rem_all1, rem_str1, flip1)

        # Op2: simplify node 10 (~node7, whose leaf is node 7!) to inverted wire ~node7!
        leaves2 = [7, 1, 2, 3]
        rem_all2, rem_str2 = bounded_mffc_nodes(10, leaves2, aig, refs)
        tt2 = original_cone_truth(aig, 10, leaves2)
        self.assertEqual(tt2, 0xAAAA ^ 0xFFFF)
        cand2 = {
            "subgraph": 1,
            "n_leaves": 4,
            "n_size": 4,
            "root_node": 0,  # wire to leaf 0 (node 7)
            "root_compl": 1,  # ~node7
            "edges": [],
        }
        flip2 = candidate_output_complement(tt2, [0, 0, 0, 0], cand2)
        self.assertEqual(flip2, 0)
        op2 = ReplacementOp(10, tuple(leaves2), (0, 0, 0, 0), cand2, rem_all2, rem_str2, flip2)

        selected = select_nonconflicting([((2,), op1), ((1,), op2)])
        self.assertEqual(len(selected), 2)

        with tempfile.TemporaryDirectory() as tmp:
            new_aig = apply_and_write_aig(aig, selected, Path(tmp) / "new.aig")
            self.assertEqual(eval_aig_all_inputs(new_aig), orig_tt)
            self.assertEqual(new_aig.and_count, 3)  # node 7 rebuilt with 3 ANDs, nodes 8/9/10 swept!

    def test_conflict_blocks_leaf_deletion(self):
        # Node 5 = x1 & x2, Node 6 = node5 & x3, Node 7 = node6 & x4
        # Op A at root 6 with leaves [1,2,3,4] has rem_strict={5}.
        # Op B at root 7 with leaves [5,3,4,1] needs leaf 5 -> must conflict!
        aig = fixture(4, [(2, 4), (10, 6), (12, 8)], [14])
        refs = references(aig)
        ra, sa = bounded_mffc_nodes(6, [1, 2, 3, 4], aig, refs)
        rb, sb = bounded_mffc_nodes(7, [5, 3, 4, 1], aig, refs)
        dummy_cand = {"subgraph": 0, "n_leaves": 4, "n_size": 4, "root_node": 0, "root_compl": 0, "edges": []}
        op_a = ReplacementOp(6, (1, 2, 3, 4), (0, 0, 0, 0), dummy_cand, ra, sa, 0)
        op_b = ReplacementOp(7, (5, 3, 4, 1), (0, 0, 0, 0), dummy_cand, rb, sb, 0)
        sel = select_nonconflicting([((10.0,), op_a), ((5.0,), op_b)])
        self.assertEqual([op.root for op in sel], [6])

    def test_global_po_sensitivity_aggregates_multiple_pos(self):
        # x1, x2, x3, x4 -> node 5=(x1&x2) depth1, node 6=(node5&x3) depth2, node 7=(x3&x4) depth1
        # POs: [node6 (depth 2), node6 (depth 2), node7 (depth 1)]
        aig = fixture(4, [(2, 4), (10, 6), (6, 8)], [12, 12, 14])
        lv = propagate_levels(aig)
        phi, sens = global_po_sensitivity(aig, lv, tau=1.0)
        self.assertAlmostEqual(phi, 2.0 + math.exp(-1.0), places=9)
        self.assertAlmostEqual(sens[6], 2.0, places=9)
        self.assertAlmostEqual(sens[7], math.exp(-1.0), places=9)
        # node 5 has level 1, x3 has level 0 -> split exp(0)/(exp(0)+exp(-1))
        w5 = 1.0 / (1.0 + math.exp(-1.0))
        self.assertAlmostEqual(sens[5], 2.0 * w5, places=9)

    def test_propagate_fanout_delays_zero_alpha_matches_levels(self):
        aig = fixture(4, [(2, 4), (10, 6), (6, 8), (12, 14)], [12, 16])
        lv = propagate_levels(aig)
        dp = propagate_fanout_delays(aig, fanout_alpha=0.0)
        self.assertEqual(dp.depth, float(lv.depth))
        self.assertEqual(dp.level, [float(x) for x in lv.level])
        self.assertEqual(dp.required, [None if r is None else float(r) for r in lv.required])

    def test_propagate_fanout_delays_shifts_sensitivity_to_high_fanout(self):
        # Node 5 = (x1 & x2) fans out to nodes 6, 7, 8 (fanout=3).
        # Node 9 = (x3 & x4), Node 10 = (node9 & x1), Node 11 = (node10 & x2) (fanout=1, structural depth 3).
        # POs: [node6 (struct depth 2, fo-loaded), node11 (struct depth 3, fo=1)].
        aig = fixture(
            4,
            [(2, 4), (10, 6), (10, 8), (10, 2), (6, 8), (18, 2), (20, 4)],
            [12, 14, 16, 22],
        )
        # Check unit timing first: PO node 11 has depth 3 > node 6 depth 2 -> sens[11] > sens[6]
        lv = propagate_levels(aig)
        _, sens_unit = global_po_sensitivity(aig, lv, tau=1.0)
        self.assertGreater(sens_unit[11], sens_unit[6])

        # With fanout_alpha=1.0: node 5 has fanout=3 -> gate_delay[5] = 3.0 -> node 6 arrival = 4.0 > node 11 = 3.0!
        dp = propagate_fanout_delays(aig, fanout_alpha=1.0)
        self.assertEqual(dp.gate_delay[5], 3.0)
        self.assertEqual(dp.depth, 4.0)
        _, sens_fo = global_po_sensitivity(aig, dp, tau=1.0)
        self.assertGreater(sens_fo[6], sens_fo[11])
        expected_w = 1.0 / (1.0 + math.exp(-3.0))
        self.assertAlmostEqual(sens_fo[5], 3.0 * expected_w, places=9)

    def test_original_cone_timing_and_refs_splits_external_sensitivity(self):
        # x1..x4; node 5=(x1&x2) [fanout=2: node 6 & PO], node 6=(node5&x3), node 7=(node6&x4).
        # Cut at root=7 with leaves=[5, 3, 4, 1] (node 5 is both a cut leaf and directly a PO!).
        aig = fixture(4, [(2, 4), (10, 6), (12, 8)], [10, 14])
        dp = propagate_fanout_delays(aig, fanout_alpha=0.5)
        _, sens = global_po_sensitivity(aig, dp, tau=1.0)
        rem_all, _ = bounded_mffc_nodes(7, [5, 3, 4, 1], aig, dp.fanouts)
        leaf_delays, orig_refs, sens_other = original_cone_timing_and_refs(
            aig, 7, [5, 3, 4, 1], dp, rem_all, sens, tau=1.0
        )
        self.assertEqual(rem_all, frozenset({6, 7}))
        self.assertEqual(orig_refs, (1, 1, 1, 0))
        # Path delay from leaf 5 through nodes 6 (gd=1.0) and 7 (gd=1.0) is 2.0.
        self.assertAlmostEqual(leaf_delays[0], 2.0, places=9)
        # PO directly on node 5 gives external sensitivity exp((level[5] - depth) / 1.0) outside rem_all.
        expected_ext_5 = math.exp((dp.level[5] - dp.depth) / 1.0)
        self.assertAlmostEqual(sens_other[0], expected_ext_5, places=9)
        self.assertAlmostEqual(sens_other[1], 0.0, places=9)

    def test_compute_adaptive_tau_scales_with_po_spread(self):
        # Uniform POs (depth 1 & 1) -> clamped to min_tau=1.0
        aig_flat = fixture(2, [(2, 4)], [6, 6])
        dp_flat = propagate_fanout_delays(aig_flat, fanout_alpha=0.2)
        self.assertEqual(compute_adaptive_tau(aig_flat, dp_flat), 1.0)

        # Wide spread: PO0 depth=1, PO1 depth=1, PO2 depth=1, PO3 fanout-loaded -> spread > 0
        # Build deep chain so depth=21 vs q75=1 -> spread=20 -> 0.2*20=4.0
        gates = [(2, 4)]  # node 3 depth 1
        prev = 6
        for idx in range(20):
            gates.append((prev, 2))
            prev = 2 * (4 + idx)
        aig_wide = fixture(2, gates, [6, 6, 6, prev])
        dp_wide = propagate_fanout_delays(aig_wide, fanout_alpha=0.0)
        self.assertAlmostEqual(compute_adaptive_tau(aig_wide, dp_wide), 4.0, places=9)
        self.assertAlmostEqual(compute_adaptive_tau(aig_wide, dp_wide, scale=2.0), 6.0, places=9)

    def test_v4_pi_phase_aware_and_slew_coupling(self):
        # x1 (lit 2/3) used positively twice and negatively 3 times (neg_fo=3 -> l_neg=2)
        # node 3 = (~x1 & x2), node 4 = (~x1 & x2) [diff phase: x1 & x2], node 5 = (node3 & ~x1), node 6 = (node5 & x2)
        aig = fixture(2, [(3, 4), (2, 4), (6, 3), (10, 4)], [3, 8, 12])
        dp = propagate_fanout_delays(aig, fanout_alpha=0.2, pi_fanout_alpha=0.5, slew_beta=0.1, pi_phase_aware=True)
        # Positive x1 has 0 arrival, complemented ~x1 has 0.5 * (3 - 1) = 1.0
        self.assertAlmostEqual(edge_arrival(dp, 2), 0.0, places=9)
        self.assertAlmostEqual(edge_arrival(dp, 3), 1.0, places=9)
        # Node 3 = (~x1 & x2) has load=0 (fo=1), arrival = 1.0 + 1.0 = 2.0
        self.assertAlmostEqual(dp.level[3], 2.0, places=9)
        # Adaptive tau uses edge_arrival for complemented PO lit=3 without error
        tau = compute_adaptive_tau(aig, dp)
        self.assertGreaterEqual(tau, 1.0)
        # Slew coupling coefficients non-negative and zero on unused/load-0 gates
        _, sens = global_po_sensitivity(aig, dp, tau=1.0)
        coef, coef_neg = slew_coupling_coefficients(aig, dp, sens)
        self.assertEqual(len(coef), aig.maximum_variable + 1)
        self.assertEqual(len(coef_neg), aig.maximum_variable + 1)
        self.assertTrue(all(x >= 0.0 for x in coef + coef_neg))

    def test_select_4b_v4_legacy_equivalence_and_budget_guard(self):
        import json
        from scripts.step4_fanout_pilot import select_4b_fanout

        # Build simple AIG: x1..x4; node 5=(x1&x2), node 6=(node5&x3), node 7=(node6&x4), PO=[node7]
        aig = fixture(4, [(2, 4), (10, 6), (12, 8)], [14])
        dp = propagate_fanout_delays(aig, fanout_alpha=0.2)
        _, sens = global_po_sensitivity(aig, dp, tau=1.0)
        # abc_dump IDs: PIs 1..4, PO=5 (offset=1), ANDs 6,7,8 -> root=8 (aig 7) with leaves [1,2,3,4]
        rec = {
            "node": 8,
            "leaves": [{"id": i, "compl": 0} for i in (1, 2, 3, 4)],
            "n_nodes_saved": 3,
            "candidates": [{
                "subgraph": 1,
                "self": False,
                "n_leaves": 4,
                "n_size": 7,
                "nodes_added": 3,
                "root_node": 6,
                "root_compl": 0,
                "leaf_depths": [2, 2, 2, 2],
                "edges": [[0, 0, 1, 0], [2, 0, 3, 0], [4, 0, 5, 0]],
            }],
        }
        with tempfile.TemporaryDirectory() as tmp:
            cuts = Path(tmp) / "cuts.jsonl"
            cuts.write_text(json.dumps(rec) + "\n")
            ops_old = select_4b_fanout(aig, dp, sens, cuts, lam_area=0.05, fanout_alpha=0.2, tau=1.0)
            cfg_legacy = Select4BConfig(lam_area=0.05, fanout_alpha=0.2)
            ops_new = select_4b_v4(aig, dp, sens, cuts, tau=1.0, cfg=cfg_legacy)
            self.assertEqual([(o.root, o.cand["subgraph"], o.flip) for o in ops_old],
                             [(o.root, o.cand["subgraph"], o.flip) for o in ops_new])
            # Also verify full v4 switches run cleanly and produce equivalent AIG
            dp_v4 = propagate_fanout_delays(aig, fanout_alpha=0.2, pi_fanout_alpha=0.1, slew_beta=0.005, pi_phase_aware=True)
            _, sens_v4 = global_po_sensitivity(aig, dp_v4, tau=1.0)
            cfg_v4 = Select4BConfig(
                lam_area=0.05, fanout_alpha=0.2, pi_fanout_alpha=0.1,
                strash_aware=True, per_root_k=2, fanout_budget=True,
            )
            ops_v4 = select_4b_v4(aig, dp_v4, sens_v4, cuts, tau=1.0, cfg=cfg_v4)
            self.assertEqual(len(ops_v4), 1)
            new_aig = apply_and_write_aig(aig, ops_v4, Path(tmp) / "v4.aig")
            self.assertEqual(eval_aig_all_inputs(new_aig), eval_aig_all_inputs(aig))
