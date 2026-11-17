import unittest
from types import SimpleNamespace
import torch
from global_timing.compositional_proxy import condense_original,candidate_paths,CompositionalTimingSurrogate


def fixture():
    aig=SimpleNamespace(and_nodes={5,6,7,8},fanin_literals={5:(2,4),6:(10,6),7:(12,8),8:(14,2)},
                        pi_count=4,and_count=4,po_literals=[16])
    decision={'root':7,'cut_leaves':[1,2,3,4],'truth':0x8000,'truth_class':0,'cut_slot':0,
              'candidate_subgraph_indices':[0]}
    library={'p_phases':{0x8000:0},'p_perms':{0x8000:0},'perms4':[[0,1,2,3]],
             'classes':{0:{'fanins':[(0,1,0,0),(2,3,0,0),(4,5,0,0)],'roots':[6]}}}
    catalogue=SimpleNamespace(decisions=[decision],validate=lambda x:None)
    reference=SimpleNamespace(catalogue=catalogue,partition=SimpleNamespace(aig=aig),n_decisions=1,width=2,counts=[2],
        delay_reference=100.,area_reference=10.,area_baseline=10.,features=torch.zeros((1,2,4),dtype=torch.float64),
        valid=torch.ones((1,2),dtype=torch.bool),sizes=torch.ones(1,dtype=torch.float64),
        baseline_po=torch.tensor([100.],dtype=torch.float64),area_weights=torch.tensor([0.,0.,1.,0.],dtype=torch.float64))
    scores=[{'decision_index':0,'choice_index':c,'predicted_local_abc_delay':v} for c,v in [(0,3.),(1,2.)]]
    return aig,decision,library,reference,scores


class CompositionTests(unittest.TestCase):
    def test_condensation_matches_explicit_gate_traversal(self):
        aig,*_=fixture()
        for retained in [{8},{5,8},{5,6,7,8}]:
            transfers=condense_original(aig,retained)
            arrivals={0:0.}
            for node,paths in transfers.items():arrivals[node]=max(arrivals[p]+d for p,d in paths.items())
            direct={i:0. for i in range(5)}
            for n,pair in aig.fanin_literals.items():direct[n]=max(direct[l>>1]+1+.2*(l&1) for l in pair)
            self.assertAlmostEqual(arrivals[8],direct[8])

    def test_candidate_ports_phase_and_shape(self):
        _,d,lib,*_=fixture()
        self.assertEqual(candidate_paths(d,0,lib),{1:2.,2:2.,3:2.,4:2.})
        # Complemented function has an output inversion, checked by truth.
        d=dict(d,truth=0x7fff);lib['p_phases'][0x7fff]=0;lib['p_perms'][0x7fff]=0
        self.assertEqual(candidate_paths(d,0,lib),{1:2.2,2:2.2,3:2.2,4:2.2})

    def test_anchor_hard_rewrite_and_batch_gradients(self):
        _,_,lib,reference,scores=fixture()
        model=CompositionalTimingSurrogate(reference,scores,lib)
        po,area=model(model.one_hot([0]));self.assertEqual(float(po),100.);self.assertEqual(float(area),10.)
        # Explicit balanced replacement shortens a 4-level path to 3 levels.
        po,_=model(model.one_hot([1]));self.assertAlmostEqual(float(po),75.,delta=.1)
        x=torch.tensor([[.4,.6]],dtype=torch.float64,requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda v:model(v)[0],(x,),eps=1e-6,atol=1e-5))
        batch=torch.stack([model.one_hot([0]),model.one_hot([1])])
        result,_=model(batch);self.assertEqual(result.shape,(2,1));self.assertLess(float(result[1]),float(result[0]))


if __name__=='__main__':unittest.main()
