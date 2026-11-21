import unittest
from types import SimpleNamespace

from global_timing.level_weights import candidate_root_level, original_leaf_depths, propagate_levels


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

    def test_original_longest_reconvergent_path_and_open_boundary(self):
        aig = graph(3, [(2, 4), (8, 6), (8, 10)], [12])
        levels = propagate_levels(aig)
        self.assertEqual(original_leaf_depths(aig, 6, [4, 3, 2], levels.level), ((2, 2, None), None))
        self.assertEqual(original_leaf_depths(aig, 6, [3], levels.level), ((2,), 3))

    def test_native_permuted_library_candidate_by_hand(self):
        from global_timing.level_candidates import LibraryDepths
        library = LibraryDepths()
        decision = dict(root=151, cut_slot=5, cut_leaves=[21, 34, 145, 146], truth=26217, truth_class=219)
        # Slots 0,1 are library inputs 2,3. Their longest paths are
        # input -> v4/v5 -> v6 -> v8/v9 -> v10. Slots 2,3 go via v7, v8/v9.
        self.assertEqual(library.get(decision, 0), (4, 4, 3, 3))
        self.assertEqual(library.scalar_level(decision, 0, [0, 0, 6, 4]), 9)

    def test_direct_library_output_in_a_three_leaf_cut(self):
        from unittest.mock import patch
        from global_timing.level_candidates import LibraryDepths
        truth = 0xcccc
        lib = {'p_phases': {truth: 0}, 'p_perms': {truth: 0}, 'perms4': [[1, 2, 0, 3]],
               'classes': {0: {'roots': [0], 'fanins': []}}}
        decision = dict(root=20, cut_slot=1, cut_leaves=[3, 8, 9], truth=truth, truth_class=0)
        with patch('global_timing.level_candidates.parse_library', return_value=lib):
            library = LibraryDepths()
        self.assertEqual(library.get(decision, 0), (None, 0, None))
        self.assertEqual(library.scalar_level(decision, 0, [10, 2, 7]), 2)
