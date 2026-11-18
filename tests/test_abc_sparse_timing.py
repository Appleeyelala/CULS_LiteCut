import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import torch
from global_timing.abc_sparse_timing import SparseTiming


class SparseTimingTest(unittest.TestCase):
    def test_shared_source_and_conditional_gradients(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            anchor=root/"a";anchor.write_text("101 0 0 0 0\n")
            bank=root/"b"
            # The selected root fans into both arms of a reconvergent two-input gate.
            def cell(i,cost):
                return f"S {i} " + " ".join(map(str,([cost,0,0,cost]*2 + [0]*16)))+"\n"
            bank.write_text("I .1 .1\nA 0 101\nN 1\n"+cell(1,6)+cell(2,2)+
                "M 1 1 1 0 1 0\nM 1 2 2 0 1 0\nN 2\nM 1 3 2 0 1 1\n"
                "N 3\nM 1 3 1 0 1 1\nN 4\nM 1 3 2 0 1 2 3\nP 4 1\nE 5\n")
            devices=["cpu","cuda"] if torch.cuda.is_available() else ["cpu"]
            results=[]
            for device in devices:
                m=SparseTiming(bank,anchor,{1:[(-1,0),(0,1)]},device)
                z=torch.zeros(2,device=device,dtype=torch.float64,requires_grad=True)
                self.assertEqual(float(m.forward(z,[0])),14.)
                self.assertEqual(float(m.forward(z,[1])),10.)
                self.assertTrue(torch.autograd.gradcheck(lambda v:m.forward(v),z,atol=1e-5))
                st=m.straight_through(z,[0])
                self.assertEqual(float(st),14.)
                g=torch.autograd.grad(st,z)[0]
                self.assertGreater(float(g[0]),0)
                self.assertLess(float(g[1]),0)
                results.append(g.tolist())
            for g in results:
                torch.testing.assert_close(torch.tensor(g),torch.tensor(results[0]))

    def test_constant_no_decision(self):
        with TemporaryDirectory() as tmp:
            a=Path(tmp)/"a";a.write_text("7 1 2 3 4\n")
            b=Path(tmp)/"b";b.write_text("I .2 .3\nA 0 7\nP 0 1\nP -1 0\nE 0\n")
            m=SparseTiming(b,a,{})
            self.assertEqual(float(m.forward(torch.empty(0),[])),4.)


if __name__=="__main__":
    unittest.main()
