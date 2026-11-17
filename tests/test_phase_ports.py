import unittest
from types import SimpleNamespace
from global_timing.phase_ports import candidate_transfers, hard_phase_arrivals
from global_timing.compositional_proxy import candidate_paths


def nand_fixture():
    # ~(ab) AND ab = 0 would test simplification instead. Use NAND4 expressed as
    # a complemented OR of the two complemented pairs, then complemented PO.
    # Original root is ~(abcd), encoded by duplicated inverted AND4 literal.
    aig = SimpleNamespace(pi_count=4, and_count=4, and_nodes={5,6,7,8},
        fanin_literals={5:(2,4),6:(6,8),7:(10,12),8:(15,15)}, po_literals=[17])
    d={'root':8,'cut_leaves':[1,2,3,4],'truth':0x7fff,'truth_class':0,'cut_slot':0,
       'candidate_subgraph_indices':[0]}
    lib={'p_phases':{0x7fff:0},'p_perms':{0x7fff:0},'perms4':[[0,1,2,3]],
         'classes':{0:{'fanins':[(0,1,0,0),(2,3,0,0),(4,5,0,0)],'roots':[6]}}}
    return aig,d,lib


class PhasePortTests(unittest.TestCase):
    def test_output_and_consumer_complements_cancel(self):
        aig,d,lib=nand_fixture()
        # Installed candidate is AND4: library output complement XOR PO = 0.
        self.assertEqual(hard_phase_arrivals(aig,[d],[1],lib),[2.])
        # Evidence of the old error, kept as a regression counterexample.
        legacy=max(candidate_paths(d,0,lib).values())+.2
        self.assertAlmostEqual(legacy,2.4)

    def test_noop_and_both_output_polarities(self):
        aig,d,lib=nand_fixture();aig.po_literals=[16,17]
        self.assertEqual(hard_phase_arrivals(aig,[d],[0],lib),[3.2,3.4000000000000004])
        self.assertEqual(hard_phase_arrivals(aig,[d],[1],lib),[2.2,2.])

    def test_complemented_wire_keeps_boundary_phase(self):
        _,d,lib=nand_fixture();d=dict(d,truth=0x5555)
        lib['p_phases'][0x5555]=0;lib['p_perms'][0x5555]=0
        lib['classes'][0]['roots']=[0]
        self.assertEqual(candidate_transfers(d,0,lib),({(1,1):0.},{(1,0):0.}))


if __name__=='__main__':unittest.main()
