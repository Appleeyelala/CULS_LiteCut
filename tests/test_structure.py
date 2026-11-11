"""Independent oracle and structural invariants, including reconvergence."""
import random
from pathlib import Path
import unittest

from global_timing.common import legacy
from global_timing.structure import make_partition, references


def fixture(pi, fanins, outputs):
    return legacy('gate3b_analyze').Aig(Path('unit_fixture.aig'), pi + len(fanins), pi, 0,
                                      len(outputs), len(fanins), tuple(outputs),
                                      {pi + 1 + i: tuple(x) for i, x in enumerate(fanins)})


def oracle(aig):
    n = aig.maximum_variable + 1
    left, right = [0] * n, [0] * n
    for node, pair in aig.fanin_literals.items():
        left[node], right[node] = pair
    refs = references(aig)
    all_cones = [set(legacy('gate3a_analyze').strict_standard_mffc(root, left, right,
                 [refs[i] for i in range(n)], aig.pi_count)) for root in sorted(aig.and_nodes)]
    return {frozenset(cone) for cone in all_cones if not any(cone < other for other in all_cones)}


class StructureTests(unittest.TestCase):
    def check_graph(self, aig):
        partition = make_partition(aig)
        self.assertEqual({frozenset(r['internal_nodes']) for r in partition.regions}, oracle(aig))
        self.assertEqual(set(partition.owner), aig.and_nodes)
        self.assertEqual(sum(r['cone_size'] for r in partition.regions), aig.and_count)
        position = {r: i for i, r in enumerate(partition.order)}
        self.assertTrue(all(position[a] < position[b] for a, b in partition.edges))
        direct = {i: 0 for i in range(aig.pi_count + 1)}
        for node, literals in sorted(aig.fanin_literals.items()):
            direct[node] = 1 + max(direct[x >> 1] for x in literals)
        region_arrival = {}
        for rid in partition.order:
            region_arrival[rid] = max((region_arrival[partition.owner[port]] if port in partition.owner else 0) + length
                                     for port, length in partition.transfers[rid].items())
            self.assertEqual(region_arrival[rid], direct[partition.regions[rid]['root']])

    def test_reconvergence_duplicate_edges_constants_and_dangling(self):
        self.check_graph(fixture(3, [(2, 4), (8, 6), (8, 7), (10, 12), (14, 14), (2, 1)], [16, 4, 0]))

    def test_empty_and_graph(self):
        self.check_graph(fixture(2, [], [2, 5, 0]))

    def test_generated_dags_against_full_oracle(self):
        rng = random.Random(20261111)
        for _ in range(80):
            pi = rng.randint(1, 6)
            size = rng.randint(1, 35)
            fanins = [(rng.randrange(2 * node), rng.randrange(2 * node)) for node in range(pi + 1, pi + 1 + size)]
            outputs = [rng.randrange(2 * (pi + size + 1)) for _ in range(4)]
            self.check_graph(fixture(pi, fanins, outputs))


if __name__ == '__main__':
    unittest.main()
