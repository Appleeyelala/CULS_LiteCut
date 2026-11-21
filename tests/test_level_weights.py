import unittest
from types import SimpleNamespace

from global_timing.level_weights import candidate_root_level, propagate_levels


def graph(pis, ands, outputs):
    return SimpleNamespace(pi_count=pis, maximum_variable=pis + len(ands),
                           fanin_literals=dict(enumerate(ands, pis + 1)), po_literals=outputs)


class LevelTest(unittest.TestCase):
    def test_chain_inverted_edges_and_loose_output(self):
        result = propagate_levels(graph(2, [(2, 5), (7, 2), (9, 4)], [11, 3]))
        self.assertEqual(result.level, [0, 0, 0, 1, 2, 3])
        self.assertEqual(result.required, [None, 0, 0, 1, 2, 3])
        self.assertEqual(result.depth, 3)

    def test_reconvergence_takes_tightest_requirement(self):
        # x3 feeds x6 directly and via x4, x5.
        result = propagate_levels(graph(2, [(2, 4), (6, 2), (8, 4), (6, 10)], [12]))
        self.assertEqual(result.required, [None, 0, 0, 1, 2, 3, 4])
        self.assertEqual(result.slack(3), 0)

    def test_tied_outputs_and_unreachable_nodes(self):
        result = propagate_levels(graph(3, [(2, 4), (2, 6), (4, 6)], [8, 11, 6]))
        self.assertEqual(result.depth, 1)
        self.assertEqual([result.slack(i) for i in (4, 5, 6)], [0, 0, None])
        self.assertEqual(result.level[6], 1)

    def test_direct_pi_constants_and_shorter_output(self):
        result = propagate_levels(graph(2, [(2, 3)], [0, 1, 5, 6]))
        self.assertEqual(result.required, [1, 0, 1, 1])
        self.assertEqual(result.slack(2), 1)
        self.assertEqual(propagate_levels(graph(1, [], [0, 1, 3])).depth, 0)

    def test_direct_unused_and_unequal_leaf_depths(self):
        self.assertEqual(candidate_root_level([7, 20, 1], [0, None, 4]), 7)
        self.assertEqual(candidate_root_level([], []), 0)
        with self.assertRaises(ValueError):
            candidate_root_level([3], [])
