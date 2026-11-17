import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from train_joint_gradient_ports import timing_loss,move_graph
from global_timing.learned_ports import PortTimingParameters,LearnedPortSurrogate
from test_learned_ports import fixture


class PortLearningTests(unittest.TestCase):
    def test_delta_loss_gradient_rewards_correct_direction(self):
        predicted=torch.tensor([[101.,90.],[99.,90.]],dtype=torch.double,requires_grad=True)
        target=torch.tensor([[98.,90.],[102.,90.]],dtype=torch.double)
        loss=timing_loss(predicted,target,torch.tensor([100.,90.]),100.)
        gradient=torch.autograd.grad(loss,predicted)[0]
        self.assertGreater(float(gradient[0,0]),0.)
        self.assertLess(float(gradient[1,0]),0.)

    def test_shared_float32_heads_update_from_double_timing_tape(self):
        torch.manual_seed(23);reference,scores,library=fixture()
        shared=PortTimingParameters()
        model=LearnedPortSurrogate(reference,scores,library,shared)
        original=shared.port[-1].weight.detach().clone()
        optimizer=torch.optim.AdamW(shared.parameters(),lr=1e-3,weight_decay=1e-4)
        x=model.one_hot([1])[None];target=torch.tensor([[80.]],dtype=torch.double)
        po,_=model(x);loss=timing_loss(po,target,model.baseline_po,model.delay_reference)
        optimizer.zero_grad();loss.backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in shared.parameters()))
        optimizer.step();self.assertFalse(torch.equal(original,shared.port[-1].weight))
        identities=[id(p) for p in shared.parameters()]
        move_graph(model,'cpu');self.assertEqual(identities,[id(p) for p in shared.parameters()])


if __name__=='__main__':unittest.main()
