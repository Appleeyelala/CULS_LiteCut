"""Small independent load and interpolation checks for the conditional Liberty graph."""
from pathlib import Path
import tempfile
import unittest
import torch

from global_timing.liberty_cover import Liberty, TimingGraph


def library(directory, device):
    lines = []
    for cell, sense, pins in [('BUF', 1, ['A']), ('INV', 2, ['A']), ('JOIN', 1, ['A', 'B'])]:
        lines.append(f'C {cell} 1 Y')
        for pin in pins:
            lines.append(f'P {pin} 2 3 {sense}')
            for base in (1, 2, 3, 4):
                # Each surface is base + 0.5 * input slew + output load.
                lines.append(f'T 2 2 0 10 0 10 {base} {base+10} {base+5} {base+15}')
    path = Path(directory) / 'tables'
    path.write_text('\n'.join(lines + ['END']))
    return Liberty(path, device)


class LoadCoverTest(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.lib = library(scratch.name, 'cpu')

    def test_reconvergent_sinks_constant_and_inversion(self):
        gates = {'d': ('BUF', ['p']), 'n': ('INV', ['d']), 'out': ('JOIN', ['d', 'n'])}
        graph = TimingGraph(self.lib, gates, ['p'], ['out', "1'b0"])
        arrival, area, load, _ = graph.forward(torch.empty(0), detail=True)
        torch.testing.assert_close(load[graph.positions['d']], torch.tensor([4., 6.]))
        torch.testing.assert_close(arrival[1], torch.zeros(2))
        # d.fall = 8, slew = 10. n.rise = 8+1+5+2 = 16; its slew is 10.
        self.assertEqual(float(arrival[0, 0]), 22.)
        self.assertEqual(float(area), 3.)

    def test_soft_load_gradient_and_static_cache(self):
        gates = {'d': ('BUF', ['p']), 'n': ('INV', ['d']), 'out': ('BUF', ['choice']),
                 'fixed': ('BUF', ['q'])}
        groups = {'choice': (0, ['d', 'n'], [1., 1.])}
        graph = TimingGraph(self.lib, gates, ['p','q'], ['out', 'fixed'], groups)
        def prediction(z):
            mass = torch.stack((1-z, z))
            return graph.forward(mass)[0]
        variable = torch.tensor(.34, requires_grad=True)
        prediction(variable).backward()
        epsilon = .001
        delta = (prediction(variable.detach()+epsilon)-prediction(variable.detach()-epsilon))/(2*epsilon)
        self.assertGreater(abs(float(delta)), .01)
        torch.testing.assert_close(variable.grad, delta, atol=.003, rtol=.003)
        self.assertTrue(graph.cached_stages)
        for mass in ([0.,1.], [1.,0.], [.3,.7]):
            weights = torch.tensor(mass)
            cached = graph.forward(weights, detail=True)
            fresh = TimingGraph(self.lib, gates, ['p','q'], ['out','fixed'], groups)
            for x, y in zip(cached, fresh.forward(weights, detail=True)):
                torch.testing.assert_close(x, y)

    def test_seventy_independent_muxes(self):
        gates = {'d': ('BUF', ['p'])}
        groups = {f'x{i}': (2*i, ['p', 'd'], [1., 1.]) for i in range(70)}
        graph = TimingGraph(self.lib, gates, ['p'], list(groups), groups)
        weights = torch.tensor([1.,0.] * 69 + [0.,1.])
        po, area, _, _ = graph.forward(weights, detail=True)
        self.assertEqual(float(po[:69].sum()), 0.)
        torch.testing.assert_close(po[69], torch.tensor([1.,2.]))
        self.assertEqual(float(area), 1.)
