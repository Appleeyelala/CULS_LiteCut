"""Native device-dependency tests; fixture numbers are not timing labels."""
import os
import unittest

import torch

from global_timing.mffc_gpu import RegionGpuPlan, graph_layout, region_arrivals


def reference(costs, sources, targets, regions, temperature):
    """Independent scalar-region Torch graph; no native queue/CSR reduction."""
    layout = graph_layout(sources, targets, regions)
    arrivals = {}
    for region in layout['order']:
        values = [costs[:, edge] + (arrivals[source] if source >= 0 else 0)
                  for edge, (source, target) in enumerate(zip(sources, targets)) if target == region]
        arrivals[region] = temperature * torch.logsumexp(torch.stack(values, -1) / temperature, -1)
    return torch.stack([arrivals[r] for r in range(regions)], -1)


class LayoutTests(unittest.TestCase):
    def test_duplicate_ports_are_real_dependencies_and_csr_keeps_edge_identity(self):
        plan = graph_layout([-1, 0, -1, 0, 1], [0, 1, 1, 1, 2], 3)
        self.assertEqual(plan['forward_degree'], [0, 2, 1])
        self.assertEqual(plan['backward_degree'], [2, 1, 0])
        self.assertEqual(plan['roots'], [0])
        self.assertEqual(plan['sinks'], [2])
        self.assertEqual(plan['in_edge'], [0, 1, 2, 3, 4])
        self.assertEqual((plan['layers'], plan['max_layer_width']), (3, 1))

    def test_invalid_or_cyclic_graph_rejected_before_native_launch(self):
        for sources, targets, count in [([1, 0], [0, 1], 2), ([-1], [0], 2),
                                         ([-2], [0], 1), ([-1], [1], 1), ([], [], 0)]:
            with self.subTest(sources=sources, targets=targets, count=count), self.assertRaises(ValueError):
                graph_layout(sources, targets, count)


@unittest.skipUnless(os.environ.get('MFFC_GPU_LIBRARY') and torch.cuda.is_available(),
                     'Requires a recorded /data native build and a Slurm GPU allocation.')
class NativeQueueTests(unittest.TestCase):
    def plan(self, sources, targets, regions):
        return RegionGpuPlan(sources, targets, regions, os.environ['MFFC_GPU_LIBRARY'])

    def compare(self, sources, targets, regions, batch, scale=1.0, temperature=0.37):
        plan = self.plan(sources, targets, regions)
        generator = torch.Generator().manual_seed(73)
        costs = (torch.randn((batch, len(sources)), generator=generator, dtype=torch.float64) * scale).requires_grad_()
        weights = torch.randn((batch, regions), generator=generator, dtype=torch.float64)
        expected = reference(costs, sources, targets, regions, temperature)
        expected_grad = torch.autograd.grad((expected * weights).sum(), costs)[0]
        device_costs = costs.detach().cuda().requires_grad_()
        actual = region_arrivals(device_costs, plan, temperature)
        actual_grad = torch.autograd.grad((actual * weights.cuda()).sum(), device_costs)[0]
        profile = plan.collect_profile()
        torch.testing.assert_close(actual.cpu(), expected, rtol=2e-11, atol=2e-10)
        torch.testing.assert_close(actual_grad.cpu(), expected_grad, rtol=2e-8, atol=2e-9)
        self.assertEqual(profile['calls_cumulative'], {'forward': 1, 'backward': 1})
        self.assertGreater(profile['kernel_ms_since_collect']['forward'], 0)
        self.assertGreater(profile['kernel_ms_since_collect']['backward'], 0)

    def test_reconvergence_repeated_sources_all_pi_ports_and_arbitrary_edge_order(self):
        sources = [-1, -1, -1, 0, 1, 0, 0, -1, 2, 3, 1]
        targets = [0, 0, 1, 2, 2, 2, 3, 3, 4, 4, 4]
        order = [10, 7, 3, 9, 2, 8, 4, 1, 5, 0, 6]
        sources, targets = [sources[i] for i in order], [targets[i] for i in order]
        for batch, scale in [(1, 0.0), (3, 1.0), (2, 10000.0)]:
            with self.subTest(batch=batch, scale=scale):
                self.compare(sources, targets, 5, batch, scale)

    def test_gpu_first_derivative_matches_finite_differences(self):
        plan = self.plan([-1, -1, 0, -1, 0, 1], [0, 0, 1, 1, 2, 2], 3)
        costs = torch.randn((1, 6), generator=torch.Generator().manual_seed(7), dtype=torch.float64).cuda().requires_grad_()
        self.assertTrue(torch.autograd.gradcheck(lambda c: region_arrivals(c, plan, 0.8),
                                               (costs,), atol=2e-5, rtol=2e-4))
        plan.collect_profile()

    def test_deep_narrow_chain_4096_and_reverse_dependency_completion(self):
        count = 4096
        plan = self.plan(list(range(-1, count - 1)), list(range(count)), count)
        generator = torch.Generator().manual_seed(71)
        costs = torch.rand((2, count), generator=generator, dtype=torch.float64).cuda().requires_grad_()
        weights = torch.randn((2, count), generator=generator, dtype=torch.float64).cuda()
        actual = region_arrivals(costs, plan, 0.3)
        grad = torch.autograd.grad((actual * weights).sum(), costs)[0]
        expected_grad = weights.flip(-1).cumsum(-1).flip(-1)
        torch.testing.assert_close(actual, costs.cumsum(-1), rtol=2e-12, atol=2e-10)
        torch.testing.assert_close(grad, expected_grad, rtol=2e-8, atol=2e-8)
        profile = plan.collect_profile()
        self.assertEqual(profile['region_dag_layers'], count)
        self.assertEqual(profile['max_layer_width'], 1)

    def test_wide_reconvergent_queue_and_repeated_reinitialization(self):
        width = 512
        # PI roots -> pairwise sums -> one high-fanin output; duplicates matter.
        sources, targets = [-1] * width, list(range(width))
        for r in range(width, 2 * width):
            sources.extend([r - width, (r - width + 1) % width, r - width])
            targets.extend([r] * 3)
        sources.extend(range(width, 2 * width))
        targets.extend([2 * width] * width)
        for _ in range(3):
            self.compare(sources, targets, 2 * width + 1, 3)

    def test_signed_endpoint_adjoints_and_sparse_random_levels(self):
        generator = torch.Generator().manual_seed(211)
        sources, targets = [], []
        for r in range(192):
            parents = [-1] if r < 16 else [-1] + torch.randint(0, r, (4,), generator=generator).tolist()
            sources.extend(parents)
            targets.extend([r] * len(parents))
        self.compare(sources, targets, 192, 4, scale=3.0, temperature=0.8)


if __name__ == '__main__':
    unittest.main()
