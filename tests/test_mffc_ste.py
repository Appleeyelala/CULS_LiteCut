"""One small selection/complete-proxy check; fixture labels are synthetic."""
import unittest

import torch

from global_timing.mffc_ste_search import ste_adam_search
from test_mffc_model import MffcModelTests
from test_mffc_search import permissive_rule


class SteAdamCheck(unittest.TestCase):
    def test_logits_update_hard_legality_softmax_gradient_and_original_anchor(self):
        _, model = MffcModelTests().models()
        rule = permissive_rule(model)
        before = [p.detach().clone() for p in model.parameters()]
        def hard_forward(module, args):
            x = args[0].detach()
            torch.testing.assert_close(x, x.round(), atol=1e-15, rtol=0)
            torch.testing.assert_close(x.sum(-1), torch.ones(x.shape[0], device=x.device, dtype=x.dtype))
            rule.validate(x.argmax(-1).cpu().tolist())
        handle = model.register_forward_pre_hook(hard_forward)
        best, trace = ste_adam_search(model, rule)
        handle.remove()
        self.assertEqual(trace['steps'], 40)
        self.assertGreater(trace['logits_total_change_norm'], 0)
        self.assertTrue(all(r['logits_gradient_norm'] > 0 and r['adam_update_norm'] > 0 for r in trace['trajectory']))
        self.assertGreater(trace['distinct_hard_combinations_including_noop'], 1)
        self.assertTrue(any(best['choices']))
        self.assertLess(trace['anchor_po_max_error_ps'], 1e-8)
        for old, new in zip(before, model.parameters()):
            torch.testing.assert_close(old, new, atol=0, rtol=0)
            self.assertIsNone(new.grad)
        logits = torch.tensor([[2., .01, -.01]], dtype=torch.float64, device='cuda', requires_grad=True)
        p = torch.softmax(logits / .7, -1)
        hard = torch.tensor([[1., 0., 0.]], dtype=torch.float64, device='cuda')
        weights = torch.tensor([[1., 2., 4.]], dtype=torch.float64, device='cuda')
        g = torch.autograd.grad(((hard - p.detach() + p) * weights).sum(), logits, retain_graph=True)[0]
        ref = torch.autograd.grad((p * weights).sum(), logits)[0]
        torch.testing.assert_close(g, ref, atol=0, rtol=0)
        print('STE fixture: logits update norm=', trace['logits_total_change_norm'],
              'distinct hard combinations=', trace['distinct_hard_combinations_including_noop'],
              'native calls=', model.native_plan.collect_profile()['calls_cumulative'])


if __name__ == '__main__':
    unittest.main()
