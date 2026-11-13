"""GPU candidate/region aggregation + full DAG, versus independent CPU model."""
import os
import unittest

import torch

from global_timing.mffc_model import MffcGpuSurrogate
from global_timing.mffc_search import gradient_blocks, prepare_selection_mask
from test_mffc_search import permissive_rule
from test_surrogate import model_fixture


@unittest.skipUnless(os.environ.get('MFFC_GPU_LIBRARY') and torch.cuda.is_available(), 'Native CUDA build required.')
class MffcModelTests(unittest.TestCase):
    def models(self):
        cpu = model_fixture()
        scores = []
        for i in range(2):
            for c in range(3):
                raw = [0.] * 19
                raw[8], raw[11] = 3 - .4 * c, 5 - c
                scores.append({'decision_index': i, 'choice_index': c, 'continuous_features_19': raw,
                               'predicted_local_abc_delay': 20 - c * 3})
        gpu = MffcGpuSurrogate(cpu.partition, cpu.catalogue, scores,
                              {'delay_ps': 100., 'area': 10., 'po_arrivals_ps': [60., 100.]},
                              os.environ['MFFC_GPU_LIBRARY'])
        return cpu, gpu

    def test_outputs_candidate_gradients_and_all_parameter_gradients(self):
        cpu, gpu = self.models()
        with torch.no_grad():
            cpu.interaction_weights.copy_(torch.tensor([.2, -.1, .3, .4], dtype=torch.float64))
        gpu.load_state_dict(cpu.state_dict())
        x = torch.softmax(torch.randn((3, 2, 3), generator=torch.Generator().manual_seed(13), dtype=torch.float64), -1).requires_grad_()
        y = x.detach().cuda().requires_grad_()
        cp, ca = cpu(x)
        gp, ga = gpu(y)
        cg = torch.autograd.grad(cp.sum() + ca.sum() * 2, (x, *cpu.parameters()))
        gg = torch.autograd.grad(gp.sum() + ga.sum() * 2, (y, *gpu.parameters()))
        torch.testing.assert_close(gp.cpu(), cp, rtol=1e-11, atol=1e-10)
        torch.testing.assert_close(ga.cpu(), ca, rtol=1e-11, atol=1e-10)
        for actual, expected in zip(gg, cg):
            torch.testing.assert_close(actual.cpu(), expected, rtol=2e-9, atol=2e-9)
        self.assertGreater(gpu.native_plan.collect_profile()['calls_cumulative']['backward'], 0)

    def test_calibration_only_updates_twelve_proxy_parameters(self):
        cpu, gpu = self.models()
        rows = [{'index': 0, 'choices': [0, 0], 'status': 'PASS', 'metrics': {'po_arrivals_ps': [60., 100.], 'area': 10.}},
                {'index': 1, 'choices': [2, 1], 'status': 'PASS', 'metrics': {'po_arrivals_ps': [57., 92.], 'area': 9.5}}]
        a, b = cpu.fit_observations(rows, steps=3), gpu.fit_observations(rows, steps=3)
        torch.testing.assert_close(torch.tensor(a['losses']), torch.tensor(b['losses']), rtol=2e-5, atol=1e-8)
        self.assertEqual(sum(p.numel() for p in gpu.parameters()), 12)
        for x, y in zip(cpu.parameters(), gpu.parameters()):
            torch.testing.assert_close(x, y.cpu(), rtol=2e-6, atol=2e-7)
        gpu.native_plan.collect_profile()

    def test_gpu_global_direction_proposals_match_cpu_reference(self):
        cpu, gpu = self.models()
        outputs = []
        for model in [cpu, gpu]:
            rule = permissive_rule(model)
            prepare_selection_mask(model, rule)
            outputs.append(gradient_blocks(model, rule, [0, 0], {(0, 0)}, maximum_edits=2))
        self.assertEqual([p['choices'] for p in outputs[0][0]], [p['choices'] for p in outputs[1][0]])
        self.assertAlmostEqual(outputs[0][1]['gradient_norm'], outputs[1][1]['gradient_norm'], places=10)
        gpu.native_plan.collect_profile()


if __name__ == '__main__':
    unittest.main()
