"""Constant incidence aggregation with a bounded-memory first-order adjoint."""
import torch
from torch.autograd.function import once_differentiable


class _FixedIncidenceMm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, incidence, values, chunk_entries):
        ctx.save_for_backward(incidence)
        ctx.chunk_entries = chunk_entries
        # Keep the existing CUDA sparse forward and its exact arithmetic.
        return torch.sparse.mm(incidence, values)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        incidence, = ctx.saved_tensors
        indices, coefficients = incidence.indices(), incidence.values()
        result = grad_output.new_zeros((incidence.shape[1], grad_output.shape[1]))
        # d(A X)/dX = A^T grad, but A is constant. Do not materialize/sort a
        # full transposed COO or ask sparse autograd to build large temporaries.
        # Both gathered values and their weighted copy fit in this byte budget.
        budget_entries = max(1, (64 * 1024**2) // (2 * grad_output.element_size() * grad_output.shape[1]))
        block = min(ctx.chunk_entries, budget_entries)
        for start in range(0, incidence._nnz(), block):
            stop = start + block
            contributions = grad_output.index_select(0, indices[0, start:stop])
            contributions.mul_(coefficients[start:stop, None])
            result.index_add_(0, indices[1, start:stop], contributions)
        return None, result, None


def fixed_incidence_mm(incidence, values, chunk_entries=1048576):
    """Same AX and first derivative w.r.t. X; incidence must remain constant."""
    if incidence.requires_grad:
        raise ValueError('Incidence must be constant; only dense values are differentiable.')
    if incidence.layout != torch.sparse_coo or not incidence.is_coalesced():
        raise ValueError('Expected the existing coalesced COO incidence buffer.')
    if chunk_entries < 1:
        raise ValueError('Chunk size must be positive.')
    return _FixedIncidenceMm.apply(incidence, values, chunk_entries)
