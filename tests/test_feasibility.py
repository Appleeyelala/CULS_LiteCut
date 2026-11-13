"""Synthetic reuse/cycle constraints; not measurements or Boolean proofs."""
import unittest

from global_timing.candidates import Catalogue
from global_timing.feasibility import CandidateFeasibility
from test_structure import fixture


def library():
    return {'p_phases': [0], 'p_perms': [0], 'perms4': [[0, 1, 2, 3]],
            'classes': {0: {'fanins': [(0, 1, 0, 0)], 'roots': [4, 0]}}}


def decision(root, candidate=0):
    return {'root': root, 'cut_slot': 0, 'truth': 0, 'truth_class': 0,
            'cut_leaves': [1, 2, 3, 4], 'removable_nodes': [root],
            'candidate_count': 1, 'candidate_subgraph_indices': [candidate]}


class ReuseTests(unittest.TestCase):
    def test_cross_decision_original_hash_reuse_rejected_in_both_projection_orders(self):
        aig = fixture(4, [(2, 4), (6, 8), (10, 12)], [14])
        catalogue = Catalogue([decision(7), decision(5, 1)], [], {0: [0, 1]})
        rule = CandidateFeasibility(aig, catalogue, library())
        self.assertEqual(rule.references[0][1], {5})
        self.assertEqual(rule.forbidden[0][1], {1})
        with self.assertRaises(ValueError):
            rule.validate([1, 1])
        self.assertEqual(rule.project([0, 0], [(2, 0, 1), (1, 1, 1)], 2), [1, 0])
        self.assertEqual(rule.project([0, 0], [(1, 0, 1), (2, 1, 1)], 2), [0, 1])

    def test_self_reuse_and_transitive_original_descendant_are_blocked(self):
        aig = fixture(4, [(2, 4), (10, 6), (12, 8)], [14])
        catalogue = Catalogue([decision(5)], [], {0: [0]})
        rule = CandidateFeasibility(aig, catalogue, library())
        self.assertFalse(rule.allowed()[0][1])
        self.assertEqual(rule.project([0], [(3, 0, 1)], 1), [0])
        d = decision(5)
        # Boundary reference to an original descendant gives the transitive case.
        d['cut_leaves'] = [6, 4, 1, 2]
        lib = library()
        lib['classes'][0] = {'fanins': [(0, 1, 0, 0)], 'roots': [4]}
        rule = CandidateFeasibility(aig, Catalogue([d], [], {0: [0]}), lib)
        self.assertEqual(rule.references[0][1], {7})
        self.assertTrue(rule.blocked[0][1])

    def test_multiple_compatible_decisions_in_one_region_are_allowed(self):
        aig = fixture(4, [(2, 4), (6, 8), (10, 12)], [14])
        catalogue = Catalogue([decision(5, 1), decision(6, 1)], [], {0: [0, 1]})
        rule = CandidateFeasibility(aig, catalogue, library())
        self.assertEqual(rule.project([0, 0], [(1, 0, 1), (2, 1, 1)], 2), [1, 1])
        self.assertEqual(rule.project([1, 1], [(1, 0, 0)], 1), [0, 1])

    def test_input_phase_permutation_and_ambiguous_original_hashes(self):
        aig = fixture(4, [(3, 4), (3, 4), (10, 12)], [14])
        catalogue = Catalogue([decision(7), decision(5, 1)], [], {0: [0, 1]})
        lib = library()
        lib['p_phases'][0] = 2
        lib['perms4'][0] = [1, 0, 2, 3]
        rule = CandidateFeasibility(aig, catalogue, lib)
        self.assertEqual(rule.references[0][1], {5, 6})
        self.assertEqual(rule.ambiguous_hash_hits, 1)
        self.assertIn(1, rule.forbidden[0][1])


if __name__ == '__main__':
    unittest.main()
