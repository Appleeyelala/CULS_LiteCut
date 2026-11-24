import math
import tempfile
import unittest
from pathlib import Path

from global_timing.aig_replay import (
    ReplacementOp,
    apply_and_write_aig,
    bounded_mffc_nodes,
    candidate_output_complement,
    global_po_sensitivity,
    original_cone_truth,
    select_nonconflicting,
    write_binary_aig,
)
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
