"""Device-ready Region DAG propagation; MFFC jobs, not search trajectories.

The native library is built once in a committed /data Slurm workspace. This
module never JIT-compiles or installs anything during a method measurement.
"""
from collections import Counter, deque
import ctypes
import os
from pathlib import Path

import torch
from torch.autograd.function import once_differentiable


def graph_layout(sources, targets, regions):
    sources, targets = list(sources), list(targets)
    if regions < 1 or regions >= 2**31 or len(sources) != len(targets):
        raise ValueError('Invalid Region DAG dimensions.')
    incoming, outgoing = [[] for _ in range(regions)], [[] for _ in range(regions)]
    degrees = [0] * regions
    for edge, (source, target) in enumerate(zip(sources, targets)):
        if not -1 <= source < regions or not 0 <= target < regions:
            raise ValueError('Region port identity is outside the graph.')
        incoming[target].append(edge)
        if source >= 0:
            outgoing[source].append(target)
            degrees[target] += 1
    if any(not row for row in incoming):
        raise ValueError('Every region requires an incoming PI/constant/region port.')
    roots = [r for r, degree in enumerate(degrees) if degree == 0]
    work, pending, order = deque(roots), degrees.copy(), []
    layers = [1] * regions
    while work:
        node = work.popleft()
        order.append(node)
        for target in outgoing[node]:
            layers[target] = max(layers[target], layers[node] + 1)
            pending[target] -= 1
            if pending[target] == 0:
                work.append(target)
    if len(order) != regions:
        raise ValueError('Region propagation requires a DAG; cycle detected.')

    def csr(rows):
        pointers, values = [0], []
        for row in rows:
            values.extend(row)
            pointers.append(len(values))
        return pointers, values

    in_ptr, in_edge = csr(incoming)
    out_ptr, out_target = csr(outgoing)
    return dict(in_ptr=in_ptr, in_edge=in_edge, source=sources, out_ptr=out_ptr,
                out_target=out_target, forward_degree=degrees,
                backward_degree=[len(row) for row in outgoing], roots=roots,
                sinks=[r for r, row in enumerate(outgoing) if not row], order=order,
                regions=regions, edges=len(sources), layers=max(layers),
                max_layer_width=max(Counter(layers).values()))


class RegionGpuPlan:
    def __init__(self, sources, targets, regions, library, device='cuda'):
        if not os.environ.get('SLURM_JOB_ID'):
            raise RuntimeError('GPU Region propagation requires a Slurm allocation.')
        path = Path(library).resolve(strict=True)
        if not path.is_relative_to('/data/wushangge'):
            raise ValueError('Use a recorded native library under /data/wushangge.')
        self.layout = graph_layout(sources, targets, regions)
        self.device = torch.device(device)
        if self.device.type != 'cuda':
            raise ValueError('RegionGpuPlan is the CUDA backend, not a CPU fallback.')
        self.library = ctypes.CDLL(str(path))
        self.launch = self.library.mffc_region_lse
        self.launch.argtypes = [ctypes.c_int] * 4 + [ctypes.c_void_p] * 12 + [ctypes.c_double, ctypes.c_int, ctypes.c_void_p]
        self.launch.restype = ctypes.c_int
        self.library.mffc_region_error.argtypes = [ctypes.c_int]
        self.library.mffc_region_error.restype = ctypes.c_char_p
        self.regions, self.edges = regions, len(sources)
        self.topology = [torch.tensor(self.layout[name], dtype=torch.int64, device=self.device)
                         for name in ('in_ptr', 'in_edge', 'source', 'out_ptr', 'out_target')]
        self.degrees = [torch.tensor(self.layout[name], dtype=torch.int32, device=self.device)
                        for name in ('forward_degree', 'backward_degree')]
        self.queues, self.controls = [], []
        for name in ('roots', 'sinks'):
            ready = self.layout[name]
            self.queues.append(torch.tensor(ready + [-1] * (regions - len(ready)), dtype=torch.int32, device=self.device))
            self.controls.append(torch.tensor([0, len(ready), 0, 0], dtype=torch.int32, device=self.device))
        sms = torch.cuda.get_device_properties(self.device).multi_processor_count
        self.blocks = min(sms, 32, max(1, (self.layout['max_layer_width'] + 3) // 4))
        self.statuses, self.events = [], []
        self.calls = {'forward': 0, 'backward': 0}

    def run(self, reverse, costs, arrivals, temperature, adjoint=None, gradient=None):
        if costs.device != self.topology[0].device or costs.dtype != torch.float64 or not costs.is_contiguous():
            raise ValueError('Native Region costs must be contiguous float64 on the plan device.')
        pending = self.degrees[reverse].clone()
        queue = self.queues[reverse].clone()
        state = self.controls[reverse].clone()
        stream = torch.cuda.current_stream(self.device)
        before, after = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        before.record(stream)
        pointers = [t.data_ptr() for t in self.topology] + [costs.data_ptr(), arrivals.data_ptr(),
            adjoint.data_ptr() if adjoint is not None else None,
            gradient.data_ptr() if gradient is not None else None,
            pending.data_ptr(), queue.data_ptr(), state.data_ptr()]
        error = self.launch(reverse, self.regions, self.edges, costs.shape[0], *pointers,
                            temperature, self.blocks, stream.cuda_stream)
        if error:
            raise RuntimeError(self.library.mffc_region_error(error).decode())
        after.record(stream)
        # Torch buffers are used by this library on the current stream. Keep
        # temporary queue storage alive until that stream's work is complete.
        pending.record_stream(stream)
        queue.record_stream(stream)
        self.statuses.append(state)
        name = 'backward' if reverse else 'forward'
        self.calls[name] += 1
        self.events.append((name, before, after))

    def collect_profile(self):
        torch.cuda.synchronize(self.device)
        if self.statuses:
            statuses = torch.stack(self.statuses).cpu().tolist()
            if any(row[2] != self.regions or row[3] != 0 for row in statuses):
                raise RuntimeError('Incomplete/failed Region device work queues: ' + str(statuses))
        times = {name: sum(a.elapsed_time(b) for kind, a, b in self.events if kind == name)
                 for name in ('forward', 'backward')}
        self.statuses.clear()
        self.events.clear()
        return {'backend': 'mffc_ready_queue_cuda', 'regions': self.regions, 'ports': self.edges,
                'region_dag_layers': self.layout['layers'], 'max_layer_width': self.layout['max_layer_width'],
                'blocks': self.blocks, 'threads_per_block': 128, 'region_thread_group': 'warp32',
                'calls_cumulative': self.calls.copy(), 'kernel_ms_since_collect': times,
                'scope': 'Native per-region forward and reverse dependencies on GPU; no CPU DAG adjoint. Kernel time excludes tensor aggregation and is not end-to-end runtime.'}


class _RegionLse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, costs, plan, temperature):
        if costs.ndim != 2 or costs.shape[1] != plan.edges or temperature <= 0:
            raise ValueError('Invalid Region propagation cost matrix or temperature.')
        values = costs.contiguous()
        arrivals = values.new_full((values.shape[0], plan.regions), float('nan'))
        plan.run(0, values, arrivals, temperature)
        ctx.plan, ctx.temperature = plan, temperature
        ctx.save_for_backward(values, arrivals)
        return arrivals

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_arrivals):
        costs, arrivals = ctx.saved_tensors
        adjoint = grad_arrivals.contiguous().clone()
        gradient = torch.empty_like(costs)
        ctx.plan.run(1, costs, arrivals, ctx.temperature, adjoint, gradient)
        return gradient, None, None


def region_arrivals(costs, plan, temperature):
    return _RegionLse.apply(costs, plan, temperature)
