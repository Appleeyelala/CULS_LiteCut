import os
import unittest
from types import SimpleNamespace

import torch

from global_timing.endpoint_context import EndpointContextSurrogate, reference_influence
from global_timing.learned_ports import LearnedPortSurrogate, PortTimingParameters
from test_learned_ports import fixture


def make_model():
    ref,scores,lib=fixture()
    # Two independent POs: only PO0 is downstream of the selectable root.
    ref.partition.aig.po_literals.append(2)
    ref.baseline_po=torch.tensor([100.,1.],dtype=torch.double)
    shared=PortTimingParameters().double()
    base=LearnedPortSurrogate(ref,scores,lib,shared,variant='global')
    return base,EndpointContextSurrogate.from_prepared(base)


class EndpointContextTests(unittest.TestCase):
    def test_reconvergent_equal_paths_share_sensitivity(self):
        layer1=([2,3,4,5],[0,1,2,3],[0,1,2,3],[0,0,0,0],[0,1,2,3],[0,1,2,3])
        layer2=([6,7],[4,5],[4,5,6,7],[2,4,3,5],[0,0,1,1],[0,1])
        layers=[tuple(torch.tensor(v) for v in layer) for layer in [layer1,layer2]]
        ref=SimpleNamespace(initial=torch.zeros(8,dtype=torch.double),n_decisions=2,width=2,
            flat_indices=torch.tensor([1,3]),root_indices=torch.tensor([0,1]),
            option_choice=torch.tensor([-1,-1,-2,-2,-3,-3]),
            edge_prior=torch.ones(8,dtype=torch.double),outputs=torch.tensor([6]),
            temperature=.1,_layers=lambda device:layers)
        sensitivity,reach=reference_influence(ref)
        torch.testing.assert_close(sensitivity,torch.tensor([[.5,.5]],dtype=torch.double))
        torch.testing.assert_close(reach,torch.ones((1,2),dtype=torch.double))

    def test_upstream_signal_reaches_terminal_po(self):
        base,model=make_model()
        x=model.one_hot([1])[None].requires_grad_(True)
        expected,old=base._context(x)
        self.assertEqual(float(old.abs().max()),0.)
        if os.environ.get('CHECK_OLD_CONTEXT')=='1':
            # Deliberate red-phase reproduction against the old implementation.
            context=old[:,base.po_regions[:1]]
        else:
            context=model.endpoint_context(expected)
        self.assertGreater(float(context[0,0].abs().sum()),0.)
        gradient=torch.autograd.grad(context[0,0].sum(),x)[0]
        self.assertGreater(float(gradient.abs().max()),0.)

    def test_unreachable_po_has_no_edit_context(self):
        base,model=make_model()
        self.assertEqual(float(model.po_root_sensitivity[0,0]),1.)
        self.assertEqual(float(model.po_root_reachability[1,0]),0.)
        expected,_=base._context(model.one_hot([1])[None])
        self.assertEqual(float(model.endpoint_context(expected)[0,1].abs().sum()),0.)

    def test_no_new_parameters_and_identity_at_zero_head(self):
        base,model=make_model()
        self.assertEqual(sum(p.numel() for p in base.parameters()),sum(p.numel() for p in model.parameters()))
        self.assertIs(base.shared,model.shared)
        self.assertNotIn('po_root_sensitivity',base._buffers)
        x=torch.stack([base.one_hot([0]),base.one_hot([1])])
        torch.testing.assert_close(base(x)[0],model(x)[0],rtol=0,atol=0)
        with torch.no_grad():
            model.shared.global_head[-1].weight.fill_(.1)
        torch.testing.assert_close(model(base.one_hot([0]))[0],base.baseline_po,rtol=0,atol=0)

    def test_selection_gradient_matches_finite_difference(self):
        _,model=make_model()
        torch.manual_seed(17)
        with torch.no_grad():
            model.shared.global_head[-1].weight.normal_(0,.05)
        x=torch.tensor([[.4,.6]],dtype=torch.double,requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda z:model(z)[0],(x,),eps=1e-6,atol=1e-5))

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_cuda_forward_and_gradient(self):
        _,cpu=make_model()
        torch.manual_seed(18)
        with torch.no_grad():
            cpu.shared.global_head[-1].weight.normal_(0,.05)
        x=torch.tensor([[.4,.6]],dtype=torch.double,requires_grad=True)
        p=cpu(x)[0];g=torch.autograd.grad(p.sum(),x)[0]
        gpu=cpu.cuda();y=x.detach().cuda().requires_grad_(True)
        q=gpu(y)[0];h=torch.autograd.grad(q.sum(),y)[0]
        torch.testing.assert_close(p.detach(),q.detach().cpu(),rtol=1e-9,atol=1e-9)
        torch.testing.assert_close(g,h.cpu(),rtol=1e-9,atol=1e-9)
        a,b=reference_influence(gpu)
        torch.testing.assert_close(a,gpu.po_root_sensitivity)
        torch.testing.assert_close(b,gpu.po_root_reachability)


if __name__=='__main__':unittest.main()
