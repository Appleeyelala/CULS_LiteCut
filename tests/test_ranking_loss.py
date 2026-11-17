import os
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from train_joint_gradient_ports import timing_loss
from evaluate_v3_ranking import ranks
from global_timing.ranking_loss import gap_ranking_loss, true_tie_groups


class RankingLossTests(unittest.TestCase):
    def loss(self, p, y):
        fn=timing_loss if os.environ.get('CHECK_ORIGINAL_LOSS')=='1' else gap_ranking_loss
        return fn(p,y,torch.tensor([100.],dtype=p.dtype,device=p.device),100.)

    def test_common_label_offset_does_not_change_ranking_objective(self):
        p=torch.tensor([[99.],[101.]],dtype=torch.double)
        y=torch.tensor([[100.],[102.]],dtype=torch.double)
        torch.testing.assert_close(self.loss(p,y),self.loss(p,y+20),rtol=0,atol=0)

    def test_wrong_and_predicted_tied_pairs_receive_correct_gradient(self):
        y=torch.tensor([[100.],[102.]],dtype=torch.double)
        for values in [[[102.],[100.]],[[101.],[101.]]]:
            p=torch.tensor(values,dtype=torch.double,requires_grad=True)
            g=torch.autograd.grad(gap_ranking_loss(p,y,y[0],100.),p)[0]
            self.assertGreater(float(g[0]),0)
            self.assertLess(float(g[1]),0)

    def test_bounded_gap_weights_and_near_tie_discount(self):
        p=torch.tensor([[100.],[100.]],dtype=torch.double,requires_grad=True)
        losses=[]
        for gap in [.0005,.1,1.,1e9]:
            y=torch.tensor([[100.],[100.+gap]],dtype=torch.double)
            losses.append(float(gap_ranking_loss(p,y,y[0],100.)))
        self.assertEqual(losses[0],0.)
        self.assertAlmostEqual(losses[1],.1*losses[2],places=12)
        self.assertEqual(losses[2],losses[3])

    def test_groups_match_evaluation_and_permutation(self):
        y=torch.tensor([1.0018,1.,1.0009,2.],dtype=torch.double)
        g=true_tie_groups(y).tolist();r=ranks(y.tolist(),.001)
        for i in range(4):
            for j in range(4):self.assertEqual(g[i]==g[j],r[i]==r[j])
        p=torch.tensor([[3.],[1.],[2.],[4.]],dtype=torch.double)
        perm=torch.tensor([2,0,3,1]);t=y[:,None]
        torch.testing.assert_close(gap_ranking_loss(p,t,t[0],100.),gap_ranking_loss(p[perm],t[perm],t[0],100.))

    def test_singleton_and_all_true_ties_are_differentiable_zero(self):
        for n in [1,8]:
            p=torch.arange(n,dtype=torch.double)[:,None].requires_grad_(True)
            y=torch.ones_like(p)*100
            loss=gap_ranking_loss(p,y,y[0],100.)
            self.assertEqual(float(loss),0.)
            self.assertEqual(float(torch.autograd.grad(loss,p)[0].abs().sum()),0.)

    def test_stability_and_finite_difference(self):
        y=torch.tensor([[100.],[103.]],dtype=torch.double)
        p=torch.tensor([[102.],[99.]],dtype=torch.double,requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda x:gap_ranking_loss(x,y,y[0],100.),(p,)))
        extreme=torch.tensor([[1e9],[-1e9]],dtype=torch.double,requires_grad=True)
        loss=gap_ranking_loss(extreme,y,y[0],100.)
        gradient=torch.autograd.grad(loss,extreme)[0]
        self.assertTrue(torch.isfinite(loss).all() and torch.isfinite(gradient).all())
        self.assertLessEqual(float(gradient.abs().max()),1.)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_gpu_forward_and_gradient(self):
        p=torch.tensor([[102.],[99.],[100.]],dtype=torch.double,requires_grad=True)
        y=torch.tensor([[100.],[103.],[100.0005]],dtype=torch.double)
        a=gap_ranking_loss(p,y,y[0],100.);g=torch.autograd.grad(a,p)[0]
        q=p.detach().cuda().requires_grad_(True);t=y.cuda()
        b=gap_ranking_loss(q,t,t[0],100.);h=torch.autograd.grad(b,q)[0]
        torch.testing.assert_close(a,b.cpu());torch.testing.assert_close(g,h.cpu())


if __name__=='__main__':unittest.main()
