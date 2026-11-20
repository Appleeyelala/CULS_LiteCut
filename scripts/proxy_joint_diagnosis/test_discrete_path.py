"""Exercise equal critical outputs/arcs, selected covers, polarity and scaled boundaries."""
import unittest

import numpy as np
import torch

from discrete_path import critical_path
from global_timing.liberty_cover import TimingGraph


class TestLibrary:
    device = torch.device('cpu')

    def __init__(self):
        self.cells = {'buffer': dict(area=1, pins={'A': 0}), 'merge': dict(area=1, pins={'A': 0, 'B': 0}),
                      'invert': dict(area=1, pins={'A': 1})}
        self.senses = [1, 2]
        self.caps = torch.ones((2, 2))
        self.host_caps = np.ones((2, 2), dtype=np.float32)

    def index(self, values):
        return torch.tensor(values, dtype=torch.long)

    def lookup(self, ids, slew, load):
        # The inverter's rising output takes three units, all other arcs one.
        return torch.where(ids == 4, 3., 1.)


class TestBank:
    def __init__(self, ratio=1.):
        groups = {'left': (0, ['a', 'b'], [1., 1.]), 'right': (2, ['a', 'b'], [1., 1.]),
                  'other': (4, ['a', 'b'], [1., 1.]), 'cover': (6, ['join', 'short'], [ratio, 1.])}
        gates = {'long0': ('buffer', ['left']), 'long1': ('buffer', ['right']),
                 'join': ('merge', ['long0', 'long1']), 'short': ('buffer', ['other']),
                 'out': ('invert', ['cover']), 'alias': (None, ['out'])}
        self.graph = TimingGraph(TestLibrary(), gates, ['a', 'b'], ['out', 'alias', 'short'], groups)

    def hard(self, vector):
        return torch.tensor(vector, dtype=torch.float32)

    def forward(self, weights):
        return self.graph.forward(weights)


class DiscretePathTest(unittest.TestCase):
    def test_polarity_tied_paths_and_unselected_mux(self):
        bank = TestBank()
        delay, endpoints, marked = critical_path(bank, [1, 0, 0, 1, 1, 0, 1, 0])
        self.assertEqual(delay, 5.)
        self.assertEqual(endpoints, [(0, 0), (1, 0)])
        pos = bank.graph.positions
        for name in ('left', 'right', 'a', 'b'):
            self.assertEqual(marked[pos[name]].tolist(), [False, True])
        self.assertFalse(marked[pos['other']].any())
        self.assertFalse(marked[pos['short']].any())

    def test_cover_change_and_inactive_plain_inputs(self):
        bank = TestBank()
        delay, _, marked = critical_path(bank, [1, 0, 0, 1, 1, 0, 0, 1])
        self.assertEqual(delay, 4.)
        pos = bank.graph.positions
        self.assertEqual(marked[pos['other']].tolist(), [False, True])
        self.assertTrue(marked[pos['a']].any())
        for name in ('left', 'right', 'b'):
            self.assertFalse(marked[pos[name]].any())

    def test_litecut_scaled_wire_has_an_active_predecessor(self):
        bank = TestBank(ratio=2.)
        delay, _, marked = critical_path(bank, [1, 0, 0, 1, 1, 0, 1, 0])
        self.assertEqual(delay, 6.)
        self.assertTrue(marked[bank.graph.positions['join'], 1])


if __name__ == '__main__':
    unittest.main()
