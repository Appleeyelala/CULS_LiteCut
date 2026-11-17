import unittest
import torch
from global_timing.joint_gradient import JointGradientConfig,critical_objective,final_better,joint_prox_search,feasible_blocks


class Rule:
    def __init__(self,n,width=2):
        self.static=[set() for _ in range(n)]
        self.forbidden=[[set() for _ in range(width)] for _ in range(n)]
    def allowed(self):return [[True]*len(row) for row in self.forbidden]
    def validate(self,x):
        active={i for i,c in enumerate(x) if c}
        for i in active:
            if self.static[i] & (active-{i}) or self.forbidden[i][x[i]] & (active-{i}):raise ValueError('conflict')


class TwoPath(torch.nn.Module):
    def __init__(self):
        super().__init__();self.n_decisions=4;self.width=2;self.delay_reference=100.;self.area_reference=10.
        self.register_buffer('features',torch.zeros((4,2,4),dtype=torch.float64))
        self.register_buffer('valid',torch.ones((4,2),dtype=torch.bool))
    def one_hot(self,x):return torch.nn.functional.one_hot(torch.tensor(x),2).double()
    def forward(self,x):
        v=x[...,1]
        return torch.stack((100-5*v[...,0]-5*v[...,1],100-5*v[...,2]-5*v[...,3]),-1),10+0*v.sum(-1)


class JointGradientTests(unittest.TestCase):
    def test_smoothing_bound_and_gradient(self):
        po=torch.tensor([100.,100.,99.],dtype=torch.float64,requires_grad=True)
        cfg=JointGradientConfig()
        value=critical_objective(po,torch.tensor(10.),100.,10.,cfg)
        self.assertGreaterEqual(float(value),1.)
        self.assertLessEqual(float(value),1.+cfg.smoothing_error)
        g=torch.autograd.grad(value,po)[0]
        self.assertTrue(bool((g>=0).all()));self.assertAlmostEqual(float(g.sum()),.01)

    def test_smaller_smooth_loss_does_not_displace_equal_max_noop(self):
        model=TwoPath();best={'delay_ps':100.,'area':10.,'edits':[],'choices':[0]*4}
        candidate={'delay_ps':100.,'area':9.9,'edits':[(0,1)],'choices':[1,0,0,0]}
        self.assertFalse(final_better(candidate,best,model,JointGradientConfig()))

    def test_joint_updates_switch_bottlenecks_and_obey_budgets(self):
        cfg=JointGradientConfig(max_backward=8,max_forward=80)
        best,trace=joint_prox_search(TwoPath(),Rule(4),cfg)
        self.assertEqual(best['choices'],[1]*4)
        self.assertEqual(best['delay_ps'],90.)
        self.assertLessEqual(trace['forward_calls'],80);self.assertLessEqual(trace['backward_calls'],8)
        self.assertTrue(any(sum(v!=0 for v in r['accepted']['choices'])>=2 for r in trace['trajectory'] if r['accepted']))

    def test_alternative_and_atomic_conflict_swap_are_considered(self):
        rule=Rule(2,3)
        rule.forbidden[0][1]={1};rule.forbidden[1][1]={0}
        gradients=[[0,-5,-4],[0,-5,-1]]
        # Candidate 0:2 is compatible with 1:2 despite both leading choices conflicting.
        proposals,_=feasible_blocks([0,0],gradients,rule.allowed(),rule,2,JointGradientConfig())
        self.assertIn((2,2),proposals)
        self.assertNotIn((1,1),proposals)
        swapped,_=feasible_blocks([1,0],gradients,rule.allowed(),rule,2,JointGradientConfig())
        self.assertIn((0,1),swapped)

    def test_gradient_finite_difference_is_for_proxy_only(self):
        model=TwoPath();x=torch.full((4,2),.5,dtype=torch.float64,requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda v:critical_objective(*model(v),100.,10.),(x,),eps=1e-6,atol=1e-5))


if __name__=='__main__':unittest.main()
