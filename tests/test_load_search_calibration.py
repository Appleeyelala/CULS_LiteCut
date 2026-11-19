import unittest
from types import SimpleNamespace

import torch

from global_timing.load_marginals import marginal_greedy
from global_timing.load_search import optimize, size_for_step


class TwoChoiceBank:
    device = 'cpu'
    states = 2
    groups = [[(-1, 0), (0, 1)]]

    def hard(self, vector):
        return torch.tensor([1 - vector[0], vector[0]], dtype=torch.float32)

    def soft(self, z):
        return torch.softmax(z, dim=0)

    def forward(self, weights, detail=False):
        po = (weights @ weights.new_tensor([[8, 6], [4, 5]])).reshape(1, 2)
        area = weights.new_tensor(1)
        return (po, area, None, None) if detail else (po.amax(), area)


class OneRule:
    catalogue = SimpleNamespace(decisions=[{}])

    def validate(self, v):
        assert v in ([0], [1])

    def project(self, zero, edits, capacity):
        return [int(bool(edits) and capacity > 0)]


class CalibrationTest(unittest.TestCase):
    def test_each_step_uses_same_po_reference(self):
        for observed, expected in [(None,[8,5,5]), ([[2,8]],[8,7,7])]:
            selection, trace = optimize(TwoChoiceBank(), OneRule(), 'gradient', 3, None, 1,
                                        None, 1, observed_po=observed)
            self.assertEqual(selection,[1])
            self.assertEqual([r['delay_ps'] for r in trace['trace']],expected)

    def test_marginal_greedy_accepts_a_globally_faster_move(self):
        vector,trace,_=marginal_greedy(TwoChoiceBank(),OneRule(),8,1)
        self.assertEqual(vector,[1])
        self.assertEqual([t['delay_ps'] for t in trace],[8,5])

    def test_caps_are_valid_and_explore_extremes(self):
        for limit in (0,1,2,3,8,16):
            tried = set()
            for _ in range(7):
                size = size_for_step(limit,tried,2)
                self.assertTrue(0 <= size <= limit)
                tried.add(size)
            self.assertIn(limit,tried)
            if limit:
                self.assertIn(1,tried)
