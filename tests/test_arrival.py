"""Analytical DAG fixtures; no synthetic values are experimental timing labels."""
import unittest

import torch

from global_timing.arrival import prepare_arrival_layers, reference_arrivals, smooth_arrivals


def fixture():
    # Parallel PI ports, repeated source regions, reconvergence and skipped levels.
    sources = torch.tensor([-1, -1, -1, 0, 1, 0, 0, -1, 2, 3, 1])
    layers = [(torch.tensor([0, 1]), torch.arange(3), torch.tensor([0, 0, 1])),
              (torch.tensor([2, 3]), torch.arange(3, 8), torch.tensor([0, 0, 0, 1, 1])),
              (torch.tensor([4]), torch.arange(8, 11), torch.tensor([0, 0, 0]))]
    return prepare_arrival_layers(sources, layers)


class ArrivalTests(unittest.TestCase):
    def test_forward_and_reverse_match_reference_with_repeated_sources(self):
        plan = fixture()
        for batch, scale in [(1, 0.0), (3, 1.0), (2, 1000.0)]:
            generator = torch.Generator().manual_seed(61)
            costs = (torch.randn((batch, 11), generator=generator, dtype=torch.float64) * scale).requires_grad_()
            weights = torch.randn((batch, 5), generator=generator, dtype=torch.float64)
            actual = smooth_arrivals(costs, plan, 5, 0.37)
            expected = reference_arrivals(costs, plan, 5, 0.37)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            a = torch.autograd.grad((actual * weights).sum(), costs)[0]
            b = torch.autograd.grad((expected * weights).sum(), costs)[0]
            torch.testing.assert_close(a, b, atol=2e-13, rtol=2e-13)
            self.assertTrue(torch.isfinite(a).all())

    def test_first_derivative_matches_finite_differences(self):
        plan = fixture()
        costs = torch.randn((2, 11), generator=torch.Generator().manual_seed(13), dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda x: smooth_arrivals(x, plan, 5, 0.8), (costs,),
                                               atol=1e-5, rtol=1e-4))

    def test_deep_chain_preserves_values_and_gradients(self):
        count = 384
        sources = torch.tensor([source for region in range(count) for source in [region - 1, -1]])
        layers = [(torch.tensor([region]), torch.tensor([2 * region, 2 * region + 1]), torch.tensor([0, 0]))
                  for region in range(count)]
        plan = prepare_arrival_layers(sources, layers)
        costs = torch.randn((2, count * 2), generator=torch.Generator().manual_seed(7),
                            dtype=torch.float64, requires_grad=True)
        actual = smooth_arrivals(costs, plan, count, 0.3)
        expected = reference_arrivals(costs, plan, count, 0.3)
        self.assertEqual(sum(t.numel() for t in actual.grad_fn.saved_tensors),
                         costs.shape[0] * (costs.shape[1] + count))
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        a = torch.autograd.grad(actual[:, [-1, count // 2]].sum(), costs)[0]
        b = torch.autograd.grad(expected[:, [-1, count // 2]].sum(), costs)[0]
        torch.testing.assert_close(a, b, atol=2e-12, rtol=2e-12)


if __name__ == '__main__':
    unittest.main()
