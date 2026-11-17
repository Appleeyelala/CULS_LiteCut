import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from global_timing.mapper_input import aiger_bytes, canonical_graph
from global_timing.cut_library import function_truth
from global_timing.cut_timing import CutTimingProxy, local_transfers, local_original, local_peak
from global_timing.candidates import Catalogue


class SyntheticLibrary:
    inverter=0
    def __init__(self):
        self.matches={}
        for truth in (1,7,8,14):
            # Explicit AND/NAND/OR/NOR implementations with asymmetric pins.
            self.matches[2,truth]=((truth,(0,1),0),)
    def arcs(self,ci,fanout):
        if ci==0:return np.array([[[-np.inf,1.],[1.,-np.inf]]])
        negative=ci in (1,7)
        a=np.full((2,2,2),-np.inf)
        for pin in (0,1):
            for out in (0,1):a[pin,1-out if negative else out,out]=(2.,5.)[pin]+out*.1
        return a


def graph(fanins,outputs,pis=3):
    return SimpleNamespace(pi_count=pis,fanin_literals=fanins,po_literals=outputs)


class CutTimingTests(unittest.TestCase):
    def test_normalization_node_order_polarity_and_idempotence(self):
        a=graph({4:(2,5),5:(4,6),6:(8,11)},[12,9])
        b=graph({4:(6,4),5:(5,2),6:(9,10)},[12,11])
        x=canonical_graph(a);y=canonical_graph(b)
        self.assertEqual(aiger_bytes(x),aiger_bytes(y))
        self.assertEqual(aiger_bytes(x),aiger_bytes(canonical_graph(x)))
        self.assertEqual(len(x.fanin_literals),3)
        with_dead=graph({**a.fanin_literals,7:(2,6)},a.po_literals)
        self.assertEqual(aiger_bytes(x),aiger_bytes(canonical_graph(with_dead)))

    def test_boolean_parser(self):
        self.assertEqual(function_truth('(!A * B) + (A * !B)',('A','B')),6)
        self.assertEqual(function_truth("(A * B)'",('A','B')),7)

    def test_serial_parallel_phase_and_distinct_ports(self):
        lib=SyntheticLibrary()
        a=graph({4:(2,4),5:(8,6)},[10])
        matrix=local_transfers(a,lib)
        # Cut leaves are ordered by node ID: the third PI is the fast pin of
        # the second gate, while the first PI crosses two different pin arcs.
        self.assertEqual(matrix[0,8],2.)
        self.assertEqual(matrix[0,0],7.)
        positive=matrix[0].max()
        b=graph(a.fanin_literals,[11])
        complement=local_transfers(b,lib)
        np.testing.assert_array_equal(matrix[:2],complement[2:])
        self.assertGreater(positive,5.)

    def test_candidate_weight_and_arc_gradients(self):
        # A hand-built conditional port stage exercises actual forward without
        # needing the large rewrite library. Both physical transitions share a match.
        p=CutTimingProxy.__new__(CutTimingProxy);torch.nn.Module.__init__(p)
        p.layers=[tuple(torch.tensor(v,dtype=torch.double if k==1 else torch.long)
            for k,v in enumerate([[4,5,4,5],[2.,2.1,4.,4.1],[0,1,2,3],[0,1],[8,8],[-2,1],[2,2]]))]
        p.register_buffer('initial',torch.zeros(12,dtype=torch.double));p.register_buffer('outputs',torch.tensor([8,9]))
        p.root_decisions={2:[0]};p._device_layers={}
        p.root_count=2;p.register_buffer('flat_choices',torch.tensor([1]));p.register_buffer('choice_roots',torch.tensor([1]))
        x=torch.tensor([[[.6,.4]]],dtype=torch.double,requires_grad=True)
        scale=torch.tensor(1.,dtype=torch.double,requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda a,b:p(a,b),(x,scale)))
        if torch.cuda.is_available():
            expected=p(x,scale)
            cpu_grad=torch.autograd.grad(expected.sum(),(x,scale));p=p.cuda()
            gpu_x=x.detach().cuda().requires_grad_();gpu_scale=scale.detach().cuda().requires_grad_()
            actual=p(gpu_x,gpu_scale)
            torch.testing.assert_close(actual.cpu(),expected.detach())
            for actual_grad,expected_grad in zip(torch.autograd.grad(actual.sum(),(gpu_x,gpu_scale)),cpu_grad):
                torch.testing.assert_close(actual_grad.cpu(),expected_grad)

    def test_functional_support_reference_keeps_omitted_pi_paths(self):
        a=graph({4:(2,4),5:(8,6)},[10])
        local=local_original(a,{'root':5,'cut_leaves':[1,2]})
        self.assertEqual(local.extra_boundary_pis,[3])
        self.assertEqual(local.pi_count,3)
        self.assertEqual(local_peak(local,SyntheticLibrary()),float(local_transfers(local,SyntheticLibrary()).max()))

    def test_two_serial_choices_propagate_upstream_gradients(self):
        p=CutTimingProxy.__new__(CutTimingProxy);torch.nn.Module.__init__(p)
        arrays=[[[4,5,4,5],[2.,2.1,4.,4.1],[0,1,2,3],[0,1],[8,8],[-2,1],[2,2]],
                [[8,9,8,9],[3.,3.,1.,1.],[0,1,2,3],[0,1],[12,12],[-3,3],[3,3]]]
        p.layers=[tuple(torch.tensor(v,dtype=torch.double if k==1 else torch.long)
                  for k,v in enumerate(layer)) for layer in arrays]
        p.register_buffer('initial',torch.zeros(16,dtype=torch.double));p.register_buffer('outputs',torch.tensor([12,13]))
        p.root_count=3;p._device_layers={}
        p.register_buffer('flat_choices',torch.tensor([1,3]));p.register_buffer('choice_roots',torch.tensor([1,2]))
        x=torch.tensor([[[.7,.3],[.6,.4]]],dtype=torch.double,requires_grad=True)
        result=p(x)
        torch.testing.assert_close(result,torch.tensor([[[4.8,4.9]]],dtype=torch.double))
        gradient=torch.autograd.grad(result[0,0,0],x)[0]
        torch.testing.assert_close(gradient,torch.tensor([[[0.,2.],[0.,-2.]]],dtype=torch.double))
        self.assertTrue(torch.autograd.gradcheck(p,(x,)))


if __name__=='__main__':unittest.main()
