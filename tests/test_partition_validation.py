"""Mutation and independent-oracle tests of partition certificates."""
import copy
import random
import unittest

from global_timing.partition_validation import validate_partition
from global_timing.structure import make_partition
from test_structure import fixture, oracle


class PartitionValidationTests(unittest.TestCase):
    def test_certified_partitions_match_all_root_oracle_on_varied_dags(self):
        rng = random.Random(20261112)
        for _ in range(80):
            pi, size = rng.randint(1, 6), rng.randint(0, 30)
            aig = fixture(pi, [(rng.randrange(2*n), rng.randrange(2*n))
                              for n in range(pi+1, pi+1+size)],
                          [rng.randrange(2*(pi+size+1)) for _ in range(5)])
            p = make_partition(aig)
            self.assertEqual(validate_partition(p)['status'], 'PASS')
            self.assertEqual({frozenset(r['internal_nodes']) for r in p.regions}, oracle(aig))

    def test_split_closed_cone_is_rejected_even_when_coverage_and_edges_look_valid(self):
        p = make_partition(fixture(2, [(2, 4), (6, 2)], [8]))
        p.regions = [{'region_id': 0, 'root': 3, 'internal_nodes': [3], 'cone_size': 1},
                     {'region_id': 1, 'root': 4, 'internal_nodes': [4], 'cone_size': 1}]
        p.owner, p.edges, p.order = {3: 0, 4: 1}, [(0, 1)], [0, 1]
        p.transfers = [{1: 1, 2: 1}, {3: 1, 1: 1}]
        with self.assertRaisesRegex(ValueError, 'absorbed'):
            validate_partition(p)

    def test_each_port_distance_is_checked_even_when_global_depth_does_not_change(self):
        p = make_partition(fixture(3, [(2, 4), (8, 6)], [10]))
        self.assertEqual(p.transfers, [{1: 2, 2: 2, 3: 1}])
        p.transfers[0][3] = 2  # Overall longest depth remains two.
        with self.assertRaisesRegex(ValueError, 'reverse traversal'):
            validate_partition(p)

    def test_ownership_dag_and_root_mutations_are_rejected(self):
        original = make_partition(fixture(2, [(2, 4), (6, 2), (6, 4)], [8, 10]))
        for mutation in ['owner', 'edge', 'order', 'duplicate', 'size', 'root']:
            p = copy.deepcopy(original)
            if mutation == 'owner': p.owner.pop(3)
            elif mutation == 'edge': p.edges.pop()
            elif mutation == 'order': p.order.reverse()
            elif mutation == 'duplicate': p.regions[0]['internal_nodes'].append(3)
            elif mutation == 'size': p.regions[0]['cone_size'] += 1
            else: p.regions[0]['root'] = 999
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_partition(p)

    def test_external_fanout_and_disconnected_members_cannot_form_a_cone(self):
        p = make_partition(fixture(2, [(2, 4), (6, 2)], [6, 8]))
        p.regions = [{'region_id': 0, 'root': 4, 'internal_nodes': [3, 4], 'cone_size': 2}]
        p.owner, p.edges, p.order, p.transfers = {3: 0, 4: 0}, [], [0], [{1: 2, 2: 2}]
        with self.assertRaisesRegex(ValueError, 'PO fanout'):
            validate_partition(p)
        p = make_partition(fixture(2, [(2, 4), (2, 4)], [8]))
        p.regions = [{'region_id': 0, 'root': 4, 'internal_nodes': [3, 4], 'cone_size': 2}]
        p.owner, p.edges, p.order, p.transfers = {3: 0, 4: 0}, [], [0], [{1: 1, 2: 1}]
        with self.assertRaisesRegex(ValueError, 'no internal path'):
            validate_partition(p)


if __name__ == '__main__':
    unittest.main()
