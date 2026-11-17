import unittest
import torch

from global_timing.fixed_incidence import fixed_incidence_mm


class FixedIncidenceTests(unittest.TestCase):
    def check_device(self, device):
        # Non-square, shared columns, an empty row/column, and weighted entries.
        indices = torch.tensor([[0, 0, 2, 3, 3], [0, 2, 1, 0, 2]], device=device)
        values = torch.tensor([1., 2., 3., 4., 5.], dtype=torch.float64, device=device)
        incidence = torch.sparse_coo_tensor(indices, values, (5, 4)).coalesce()
        x = torch.arange(12, dtype=torch.float64, device=device).reshape(4, 3).requires_grad_()
        weights = torch.arange(15, dtype=torch.float64, device=device).reshape(5, 3) / 7
        actual = fixed_incidence_mm(incidence, x, chunk_entries=2)
        expected = incidence.to_dense() @ x
        torch.testing.assert_close(actual, expected)
        actual_grad, = torch.autograd.grad((actual * weights).sum(), x)
        expected_grad, = torch.autograd.grad((expected * weights).sum(), x)
        torch.testing.assert_close(actual_grad, expected_grad)
        self.assertTrue(torch.autograd.gradcheck(lambda z: fixed_incidence_mm(incidence, z, 2), (x,)))

    def test_cpu_gradient_and_chunk_boundaries(self):
        self.check_device('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA allocation required')
    def test_cuda_gradient_and_chunk_boundaries(self):
        self.check_device('cuda')

    def test_fixed_matrix_contract(self):
        a = torch.eye(2, dtype=torch.float64).to_sparse().requires_grad_()
        with self.assertRaisesRegex(ValueError, 'constant'):
            fixed_incidence_mm(a, torch.ones(2, 1, dtype=torch.float64))


if __name__ == '__main__':
    unittest.main()
