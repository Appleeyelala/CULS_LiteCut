import unittest

from global_timing.litecut import verify_cut_truth
from global_timing.truth import BDD
from test_structure import fixture


class TruthTests(unittest.TestCase):
    def test_functional_cut_is_valid_without_structural_boundary(self):
        # ((a & x) | (a & !x)) & b & c = a & b & c.
        aig = fixture(4, [(2, 8), (2, 9), (11, 13), (15, 4), (16, 6)], [18])
        decision = {'root': 9, 'cut_leaves': [1, 2, 3], 'truth': 0x8080}
        result = verify_cut_truth(aig, decision)
        self.assertEqual(result['cut_semantics'], 'FUNCTIONAL_SUPPORT_MINIMIZED')
        self.assertEqual(result['eliminated_external_pi_count'], 1)
        with self.assertRaises(ValueError):
            verify_cut_truth(aig, dict(decision, truth=0xaaaa))

    def test_unlisted_functional_input_is_rejected(self):
        aig = fixture(4, [(2, 8), (10, 4), (12, 6)], [14])
        with self.assertRaises(ValueError):
            verify_cut_truth(aig, {'root': 7, 'cut_leaves': [1, 2, 3], 'truth': 0x8080})

    def test_bdd_resource_limit_is_explicit(self):
        bdd = BDD(limit=3)
        bdd.node(0, 0, 1)
        with self.assertRaises(ValueError):
            bdd.node(1, 0, 1)
