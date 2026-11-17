import unittest
from types import SimpleNamespace

import torch

from global_timing.learned_ports import LearnedPortSurrogate, PortTimingParameters
from global_timing.phase_ports import hard_phase_arrivals
from test_phase_ports import nand_fixture


def fixture():
    aig,d,lib = nand_fixture()
    d.update(candidate_count=1,decision_owner_region=0)
    cat = SimpleNamespace(decisions=[d],validate=lambda choices:None)
    partition = SimpleNamespace(aig=aig,owner={n:0 for n in aig.and_nodes},regions=[None],edges=[])
    features = torch.zeros((1,2,4),dtype=torch.double); features[0,1] = torch.tensor([-.2,-.2,-.25,1.])
    reference = SimpleNamespace(catalogue=cat,partition=partition,n_decisions=1,width=2,counts=[2],
        delay_reference=100.,area_reference=10.,area_baseline=10.,features=features,
        valid=torch.ones((1,2),dtype=torch.bool),sizes=torch.tensor([4.],dtype=torch.double),
        baseline_po=torch.tensor([100.],dtype=torch.double),area_weights=torch.tensor([0.,0.,1.,0.],dtype=torch.double))
    scores = [dict(decision_index=0,choice_index=c,predicted_local_abc_delay=v,
                   continuous_features_19=[1.]*19) for c,v in [(0,3.4),(1,2.)]]
    return reference,scores,lib


class LearnedPortTests(unittest.TestCase):
    def test_literal_prior_matches_installed_structural_model(self):
        r,s,l = fixture()
        for variant in ['fixed','ports','global','combined']:
            model = LearnedPortSurrogate(r,s,l,variant=variant,smoothing_error=1e-8)
            x = torch.stack([model.one_hot([0]),model.one_hot([1])])
            po,_ = model(x)
            self.assertEqual(float(po[0]),100.)
            actual = hard_phase_arrivals(r.partition.aig,r.catalogue.decisions,[1],l)[0]
            self.assertAlmostEqual(float(po[1]),actual*model.unit_ps,places=5)

    def test_first_order_tape_matches_reference_parameters_and_selection(self):
        torch.manual_seed(17)
        r,s,l = fixture()
        parameters = PortTimingParameters().double()
        with torch.no_grad():
            parameters.port[-1].weight.normal_(0.,.05)
            parameters.global_head[-1].weight.normal_(0.,.05)
        models = [LearnedPortSurrogate(r,s,l,parameters,backend=b) for b in ['reference','bounded_tape']]
        gradients=[]
        for model in models:
            x=torch.tensor([[.4,.6]],dtype=torch.double,requires_grad=True)
            out,_=model(x)
            gradients.append(torch.autograd.grad(out.sum(),[x]+list(parameters.parameters())))
        for a,b in zip(*gradients): torch.testing.assert_close(a,b,rtol=1e-9,atol=1e-9)
        x=torch.tensor([[.4,.6]],dtype=torch.double,requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda y:models[1](y)[0],(x,),eps=1e-6,atol=1e-5))

    def test_anchor_is_preserved_after_learning_and_batch(self):
        r,s,l=fixture();model=LearnedPortSurrogate(r,s,l)
        with torch.no_grad():
            for p in model.shared.parameters(): p.normal_(0.,.05)
        a,_=model(model.one_hot([0])); self.assertEqual(float(a),100.)
        batch=torch.stack([model.one_hot([0]),model.one_hot([1])])
        p,_=model(batch)
        q,_=model(batch[1]);torch.testing.assert_close(p[1],q)

    def test_serial_joint_propagation_and_downstream_context(self):
        r,s,l=fixture();a=r.partition.aig
        a.fanin_literals.update({9:(16,4),10:(18,6),11:(20,8)})
        a.and_nodes.update({9,10,11});a.and_count=7;a.po_literals=[22]
        r.partition.owner.update({9:1,10:1,11:1});r.partition.regions=[None,None];r.partition.edges=[(0,1)]
        d=dict(r.catalogue.decisions[0],root=11,cut_leaves=[8,2,3,4],truth=0x8000,decision_owner_region=1)
        l['p_phases'][0x8000]=0;l['p_perms'][0x8000]=0
        r.catalogue.decisions.append(d);r.n_decisions=2;r.counts=[2,2]
        for name in ['features','valid','sizes']:setattr(r,name,getattr(r,name).repeat((2,1,1) if name=='features' else (2,1) if name=='valid' else (2,)))
        s=s+[dict(row,decision_index=1) for row in s]
        model=LearnedPortSurrogate(r,s,l,variant='ports',smoothing_error=1e-8)
        for choices in [[0,0],[0,1],[1,0],[1,1]]:
            pred=model.propagate(model.one_hot(choices)[None])[0]
            actual=hard_phase_arrivals(a,r.catalogue.decisions,choices,l)[0]
            self.assertAlmostEqual(float(pred),actual*model.unit_ps,places=5)
        x=torch.tensor([[.4,.6],[.3,.7]],dtype=torch.double,requires_grad=True)
        _,context=model._context(x[None])
        gradient=torch.autograd.grad(context[0,0].sum(),x)[0]
        self.assertGreater(float(gradient[1,1].abs()),0.)
        self.assertTrue(torch.autograd.gradcheck(lambda y:model(y)[0],(x,),eps=1e-6,atol=1e-5))

    def test_unused_domain_extension_does_not_dilute_global_edit_features(self):
        r,s,l=fixture();shared=PortTimingParameters().double()
        with torch.no_grad():
            for p in shared.parameters():p.normal_(0.,.1)
        first=LearnedPortSurrogate(r,s,l,shared)
        predicted,_=first(first.one_hot([1]))
        r.catalogue.decisions.append(dict(r.catalogue.decisions[0]));r.n_decisions=2;r.counts=[2,2]
        for name in ['features','valid','sizes']:setattr(r,name,getattr(r,name).repeat((2,1,1) if name=='features' else (2,1) if name=='valid' else (2,)))
        s=s+[dict(row,decision_index=1) for row in s]
        second=LearnedPortSurrogate(r,s,l,shared)
        extended,_=second(second.one_hot([1,0]))
        torch.testing.assert_close(predicted,extended,rtol=1e-12,atol=1e-12)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required for device parity')
    def test_gpu_forward_and_gradient_match_cpu(self):
        r,s,l=fixture();model=LearnedPortSurrogate(r,s,l)
        x=torch.tensor([[.4,.6]],dtype=torch.double,requires_grad=True)
        y,_=model(x);g=torch.autograd.grad(y.sum(),x)[0]
        model=model.cuda();z=x.detach().cuda().requires_grad_();v,_=model(z)
        h=torch.autograd.grad(v.sum(),z)[0]
        torch.testing.assert_close(v.cpu(),y);torch.testing.assert_close(h.cpu(),g)


if __name__=='__main__':unittest.main()
