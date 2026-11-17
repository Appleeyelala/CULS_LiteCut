"""Critical-delay aligned, proxy-only experimental selection.

No ABC/evaluator/measurement callback is accepted by this module. The bounded
discrete proximal solver is approximate; accepted steps are checked by a full
proxy forward. Its monotonicity is about the proxy, never real mapped timing.
"""
from dataclasses import asdict, dataclass
import math

import torch

from .mffc_search import prepare_selection_mask
from .mffc_ste_search import project_logits, sparse_hard


@dataclass(frozen=True)
class JointGradientConfig:
    max_backward: int = 40
    max_forward: int = 160
    beam_width: int = 32
    candidate_pool: int = 64
    initial_radius: int = 2
    smoothing_error: float = .001
    area_limit: float = 1.05
    area_penalty: float = 20.
    proximal_weight: float = 1e-6
    relinearize: bool = True

    def validate(self):
        if min(self.max_backward,self.max_forward,self.beam_width,self.candidate_pool,self.initial_radius)<1:
            raise ValueError('Positive search budgets required.')
        if not all(math.isfinite(x) and x>0 for x in
                   (self.smoothing_error,self.area_limit,self.area_penalty,self.proximal_weight)):
            raise ValueError('Positive finite objective parameters required.')


def critical_objective(po, area, delay_reference, area_reference, config=JointGradientConfig()):
    """Output smoothing has normalized error <= config.smoothing_error.

    This does not bound the error of internal propagation or physical timing.
    """
    n=po.shape[-1]
    if n<1:raise ValueError('At least one PO required.')
    normalized=po/max(delay_reference,1e-6)
    tau=config.smoothing_error/max(math.log(n),1.)
    smooth=tau*torch.logsumexp(normalized/tau,dim=-1)
    return smooth+config.area_penalty*torch.relu(area/max(area_reference,1e-6)-config.area_limit).square()


def prediction(model, po, area, choices, config):
    return {'choices':list(choices),'edits':[(i,c) for i,c in enumerate(choices) if c],
            'delay_ps':float(po.detach().max()),'area':float(area.detach()),
            'predicted_objective':float(critical_objective(po,area,model.delay_reference,model.area_reference,config).detach())}


def final_better(candidate,best,model,config):
    if candidate['area']>model.area_reference*config.area_limit:return False
    tol=max(2e-5,2e-6*max(abs(candidate['delay_ps']),abs(best['delay_ps'])))
    if candidate['delay_ps']<best['delay_ps']-tol:return True
    # Equal predicted critical delay never displaces the zero-edit anchor.
    if abs(candidate['delay_ps']-best['delay_ps'])<=tol and best['edits']:
        return (len(candidate['edits']),candidate['area'],candidate['choices']) < (len(best['edits']),best['area'],best['choices'])
    return False


def aligned_ste_search(model,feasibility,seed=20261117,steps=40,learning_rate=.12,maximum_edits=16,
                       config=JointGradientConfig()):
    """Objective/final-ranking control; retains STE initialization/projection."""
    config.validate();model.requires_grad_(False);prepare_selection_mask(model,feasibility)
    generator=torch.Generator(device=model.features.device).manual_seed(seed)
    logits=torch.nn.Parameter(torch.randn(model.features.shape[:2],dtype=torch.float64,
                              device=model.features.device,generator=generator)*.01)
    with torch.no_grad():logits[:,0].add_(2.)
    optimizer=torch.optim.Adam([logits],lr=learning_rate)
    with torch.no_grad():
        po,area=model(sparse_hard(model,[]))
        best=prediction(model,po,area,[0]*model.n_decisions,config)
    trace=[]
    for step in range(steps):
        temperature=1.-.7*step/max(steps-1,1)
        p=torch.softmax(logits.masked_fill(~model.selection_allowed,-torch.inf)/temperature,-1)
        choices,edits,rows=project_logits(model,feasibility,logits,maximum_edits)
        hard=sparse_hard(model,edits)
        po,area=model(hard-p.detach()+p)
        loss=critical_objective(po,area,model.delay_reference,model.area_reference,config)
        optimizer.zero_grad(set_to_none=True);loss.backward()
        if logits.grad is None or not torch.isfinite(logits.grad).all():raise ValueError('Invalid logits gradient.')
        norm=float(torch.nn.utils.clip_grad_norm_([logits],5.))
        row=prediction(model,po,area,choices,config)
        if final_better(row,best,model,config):best=dict(row,step=step)
        trace.append(dict(step=step,**row,gradient_norm=norm,host_candidate_rows=rows))
        optimizer.step()
    return dict(best,source='aligned_ste'),{'config':asdict(config),'trajectory':trace,
           'forward_calls':steps+1,'backward_calls':steps,'scope':'Objective/selection-only ablation; proxy values, no measured feedback.'}


def feasible_blocks(anchor, gradient, allowed, feasibility, radius, config):
    """Beam search in a finite candidate pool with atomic conflict removals.

    Includes second/third alternatives and nonpositive linear gains. The pool
    is an explicit approximation, not a claim that all candidates were searched.
    """
    n=len(anchor)
    ranked=[]
    for i,row in enumerate(gradient):
        for c,value in enumerate(row):
            if allowed[i][c] and c!=anchor[i]:ranked.append((row[anchor[i]]-value,i,c))
    ranked.sort(key=lambda r:(-r[0],r[1],r[2]))
    ranked=ranked[:config.candidate_pool]
    base=tuple(anchor)
    def key(vector):
        delta=sum(gradient[i][c]-gradient[i][anchor[i]] for i,c in enumerate(vector) if c!=anchor[i])
        changes=sum(a!=b for a,b in zip(vector,anchor))
        return delta+config.proximal_weight*changes,changes,vector
    beam=[base];visited={base};proposals=set()
    for depth in range(radius):
        expanded=set()
        for state in beam:
            for gain,i,c in ranked:
                if state[i]==c:continue
                vector=list(state)
                if c:
                    # Conflict resolution changes the explicit proposed vector;
                    # never execute a partial request and reinterpret success.
                    active={j for j,v in enumerate(vector) if v and j!=i}
                    blocked=feasibility.static[i] | feasibility.forbidden[i][c]
                    blocked=blocked | {j for j in active if i in feasibility.forbidden[j][vector[j]]}
                    for j in active & blocked:vector[j]=0
                vector[i]=c
                vector=tuple(vector)
                if vector in visited or sum(a!=b for a,b in zip(vector,base))>radius:continue
                try:feasibility.validate(list(vector))
                except ValueError:continue
                expanded.add(vector);visited.add(vector)
        if not expanded:break
        beam=sorted(expanded,key=key)[:config.beam_width]
        proposals.update(beam)
    return sorted(proposals,key=key),{'pool_entries':len(ranked),'ranked_edits':ranked,
                                     'generated_states':len(visited)-1}


def joint_prox_search(model,feasibility,config=JointGradientConfig()):
    config.validate();model.requires_grad_(False);prepare_selection_mask(model,feasibility)
    anchor=[0]*model.n_decisions;feasibility.validate(anchor)
    allowed=model.selection_allowed.detach().cpu().tolist()
    with torch.no_grad():po,area=model(model.one_hot(anchor))
    best=prediction(model,po,area,anchor,config);incumbent=best
    forward,backward=1,0
    radius=min(config.initial_radius,max(model.n_decisions,1))
    trace=[];frozen_gradient=None;seen={tuple(anchor)};reason='budget'
    while backward<config.max_backward and forward<config.max_forward:
        x=model.one_hot(anchor).requires_grad_()
        po,area=model(x);forward+=1
        loss=critical_objective(po,area,model.delay_reference,model.area_reference,config)
        if not torch.isfinite(loss):raise ValueError('Non-finite objective.')
        if config.relinearize or frozen_gradient is None:
            gradient=torch.autograd.grad(loss,x)[0];backward+=1
            if not torch.isfinite(gradient).all():raise ValueError('Non-finite global gradient.')
            frozen_gradient=gradient.detach()
        gradient=frozen_gradient
        proposals,details=feasible_blocks(anchor,gradient.cpu().tolist(),allowed,feasibility,radius,config)
        proposals=[p for p in proposals if p not in seen][:max(0,config.max_forward-forward)]
        if not proposals:
            reason='no_unvisited_legal_proposal';break
        rows=[]
        # Batch axis is independent proxy proposals; each proposal still runs
        # the complete global DAG, not independent-MFFC scores.
        with torch.no_grad():
            outputs,areas=model(torch.stack([model.one_hot(list(v)) for v in proposals]))
        forward+=len(proposals)
        for vector,p,a in zip(proposals,outputs,areas):
            row=prediction(model,p,a,list(vector),config)
            if not all(math.isfinite(row[k]) for k in ['delay_ps','area','predicted_objective']):raise ValueError('Non-finite proposal.')
            rows.append(row);seen.add(vector)
            if final_better(row,best,model,config):best=dict(row)
        candidates=[r for r in rows if r['area']<=model.area_reference*config.area_limit
                    and r['delay_ps']<=incumbent['delay_ps']+1e-10*model.delay_reference
                    and r['predicted_objective']<incumbent['predicted_objective']-1e-12]
        accepted=min(candidates,key=lambda r:(r['delay_ps'],r['predicted_objective'],len(r['edits']),r['choices'])) if candidates else None
        entry={'iteration':len(trace),'radius':radius,'anchor':anchor,'proposals':rows,
               'accepted':accepted,'forward_calls':forward,'backward_calls':backward,**details}
        if accepted:
            target=model.one_hot(accepted['choices'])
            linear=-float((gradient*(target-x.detach())).sum())
            reduction=incumbent['predicted_objective']-accepted['predicted_objective']
            agreement=reduction/linear if linear>1e-15 else None
            entry.update(linear_predicted_reduction=linear,full_proxy_reduction=reduction,agreement=agreement)
            anchor=list(accepted['choices']);incumbent=accepted
            if agreement is not None and agreement>.75:radius=min(max(radius*2,2),model.n_decisions)
            elif agreement is not None and agreement<.25:radius=max(1,radius//2)
        elif radius>1:radius=max(1,radius//2)
        else:
            reason='no_acceptable_proxy_descent';trace.append(entry);break
        trace.append(entry)
    feasibility.validate(best['choices'])
    return dict(best,source='joint_prox_gradient'),{'schema':'joint_prox_gradient_v1','config':asdict(config),
           'trajectory':trace,'forward_calls':forward,'backward_calls':backward,'termination':reason,
           'distinct_hard_combinations':len(seen),'scope':'Proxy-only approximate feasible block descent. No real timing feedback or physical improvement guarantee.'}
