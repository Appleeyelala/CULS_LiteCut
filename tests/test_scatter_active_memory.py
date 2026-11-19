import unittest

import torch

from global_timing.liberty_cover import scatter_active_max


class ActiveLoadGradientTest(unittest.TestCase):
    def test_duplicate_fanouts_ties_and_untouched_nets(self):
        for device in (['cpu','cuda'] if torch.cuda.is_available() else ['cpu']):
            index=torch.tensor([2,2,4,4,5,5],device=device)
            base=torch.tensor([1.,2.,3.,4.,5.,6.,7.],device=device,requires_grad=True)
            updates=torch.tensor([3.,3.,6.,7.,4.,2.],device=device,requires_grad=True)
            upstream=torch.arange(1.,8.,device=device)
            ref=base.scatter_reduce(0,index,updates,reduce='amax')
            actual=scatter_active_max(base,index,updates)
            torch.testing.assert_close(actual,ref,rtol=0,atol=0)
            expected=torch.autograd.grad(ref,(base,updates),upstream)
            found=torch.autograd.grad(actual,(base,updates),upstream)
            for a,b in zip(found,expected):
                torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_reverse_sweep_saves_no_full_net_vector(self):
        count=65536
        active=torch.zeros(count,requires_grad=True)
        index=torch.tensor([1,2,1,3])
        updates=torch.tensor([0.,1.,2.,3.],requires_grad=True)
        sizes=[]
        def pack(tensor):
            sizes.append(tensor.numel())
            return tensor
        with torch.autograd.graph.saved_tensors_hooks(pack,lambda tensor:tensor):
            result=scatter_active_max(active,index,updates)
        result.sum().backward()
        self.assertLess(max(sizes),count)

if __name__=='__main__':
    unittest.main()
