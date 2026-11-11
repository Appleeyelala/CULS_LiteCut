import unittest

from global_timing.candidates import Catalogue, conflict_reason


def decision(root, members, leaves):
    return {'root': root, 'cut_slot': 1, 'candidate_count': 2, 'candidate_subgraph_indices': [0, 3],
            'removable_nodes': members, 'cut_leaves': leaves}


class CandidateTests(unittest.TestCase):
    def test_conflict_types(self):
        a = decision(10, [8, 10], [1, 2, 3])
        self.assertEqual(conflict_reason(a, decision(10, [10], [4, 5, 6])), 'SAME_ROOT')
        self.assertEqual(conflict_reason(a, decision(12, [8, 12], [4, 5, 6])), 'REMOVABLE_OVERLAP')
        self.assertEqual(conflict_reason(a, decision(14, [14], [8, 5, 6])), 'CUT_LEAF_DELETION')
        self.assertIsNone(conflict_reason(a, decision(16, [16], [1, 2, 3])))

    def test_noop_and_projection_can_keep_two_decisions_in_one_region(self):
        catalogue = Catalogue([decision(10, [10], [1, 2, 3]), decision(12, [12], [1, 2, 3]),
                               decision(14, [14], [1, 2, 3])], [(0, 2)], {0: [0, 1, 2]})
        self.assertEqual(catalogue.requested([0, 0, 0]), [])
        choices = catalogue.project([[0, 5, 2], [0, 3, 4], [0, 4, 2]])
        self.assertEqual(choices, [1, 2, 0])
        self.assertEqual(catalogue.requested(choices), [(10, 1, 0), (12, 1, 3)])
        with self.assertRaises(ValueError):
            catalogue.validate([1, 0, 1])
        with self.assertRaises(ValueError):
            catalogue.validate([3, 0, 0])
        with self.assertRaises(ValueError):
            catalogue.project([[float('nan'), 1, 2]] * 3)

    def test_empty_catalogue(self):
        catalogue = Catalogue([], [], {})
        self.assertEqual(catalogue.project([]), [])
        self.assertEqual(catalogue.requested([]), [])


if __name__ == '__main__':
    unittest.main()
