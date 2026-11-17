#!/usr/bin/env python3
"""Trace saved v3 ranking errors without fitting parameters or calling ABC."""
import argparse
import csv
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))
from train_joint_gradient_ports import load_prepared_graphs, move_graph, timing_loss
from evaluate_v3_ranking import evaluate, write_csv, sha


def main():
    import torch
    import torch.nn.functional as F
    p = argparse.ArgumentParser()
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(1)
    prepared = load_prepared_graphs(a.prepared)
    for name, digest in prepared['code_hashes'].items():
        assert sha(ROOT/'global_timing'/name) == digest
    examples = prepared['examples']
    shared = examples[0]['domains'][0]['model'].shared
    shared.cuda()
    parameters = list(shared.global_head.parameters())
    a.output.mkdir(parents=True, exist_ok=False)
    rows, gradients, pairs, structures = [], [], [], []
    reproduction = []
    for seed in [20261117, 20261118, 20261119]:
        checkpoint = a.weights/f'global_{seed}'/'weights.pt'
        shared.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['state_dict'])
        saved = json.loads((ROOT/f'analysis/joint_gradient_v3/pilot/global_{seed}_validation.json').read_text())
        expected = {(r['circuit_id'], r['index']): r['predicted_delay_ps'] for r in saved['predictions'] if r['domain']=='sampled'}
        maximum_error = 0.
        for case in examples:
            for domain in case['domains']:
                model = move_graph(domain['model'], 'cuda');model.variant = 'global'
                x = domain['x'].cuda(); target = domain['target'].cuda()
                po, _ = model(x)
                pd = po.max(-1).values
                yd = target.max(-1).values
                scale = model.delay_reference*.01
                ii,jj = torch.triu_indices(len(x),len(x),offset=1,device=x.device)
                gap = yd[ii]-yd[jj];valid=gap.abs()>.001
                rank = F.softplus(-gap[valid].sign()*(pd[ii]-pd[jj])[valid]/scale).mean() if valid.any() else pd.sum()*0
                old = timing_loss(po,target,model.baseline_po,model.delay_reference)
                def grad(loss):
                    gs = torch.autograd.grad(loss, parameters, retain_graph=True)
                    return torch.cat([g.flatten() for g in gs])
                go, gr = grad(old), grad(rank)
                norm = float(go.norm()*gr.norm())
                gradients.append(dict(seed=seed,circuit_id=case['circuit_id'],role=case['role'],domain=domain['name'],
                    old_loss=float(old),candidate_rank_loss=float(rank),old_grad_norm=float(go.norm()),rank_grad_norm=float(gr.norm()),
                    gradient_cosine=float(go.dot(gr))/norm if norm else None))
                if domain['name']=='sampled':
                    with torch.no_grad():
                        model.variant='fixed'; fixed,_=model(x); model.variant='global'
                        local,context=model._context(x)
                        pad=torch.cat([context,context.new_zeros((len(x),1,8))],1)
                        ctx=pad[:,model.po_regions]
                        metrics,_,_,_=evaluate(pd.detach().tolist(),yd.tolist(),list(range(1,9)),.001,1.)
                        for k, record in enumerate(domain['records']):
                            actual_po=int(target[k].argmax()); pred_po=int(po[k].argmax())
                            changes=[]
                            for d,c in enumerate(record['choices']):
                                if c:
                                    decision=model.catalogue.decisions[d]
                                    changes.append(dict(decision=d,choice=c,root=decision['root'],region=decision['decision_owner_region'],
                                        normalized_features=model.features[d,c].tolist(),cone_size=float(model.sizes[d])))
                            row=dict(seed=seed,circuit_id=case['circuit_id'],role=case['role'],index=record['index'],edits=len(changes),
                                predicted_delay_ps=float(pd[k]),actual_delay_ps=float(yd[k]),fixed_critical_ps=float(fixed[k].max()),
                                pred_critical_po=pred_po,actual_critical_po=actual_po,
                                fixed_at_pred_critical_ps=float(fixed[k,pred_po]),residual_at_pred_critical_ps=float(po[k,pred_po]-fixed[k,pred_po]),
                                po_context_max_abs=float(ctx[k].abs().max()),
                                critical_po_context_max_abs=float(ctx[k,pred_po].abs().max()),
                                sum_local_delta=float(local[k,:,0].sum()),mean_edit_features=json.dumps(local[k].sum(0).tolist()),
                                choices=json.dumps(changes),serialized_noop_delay_ps=case['serialized_noop_delay_ps'],
                                original_delay_ps=model.delay_reference)
                            rows.append(row)
                            if case['role']=='validation':
                                maximum_error=max(maximum_error,abs(float(pd[k])-expected[case['circuit_id'],record['index']]))
                        for i,j in itertools.combinations(range(8),2):
                            true_gap=float(yd[j]-yd[i]);pred_gap=float(pd[j]-pd[i])
                            pairs.append(dict(seed=seed,circuit_id=case['circuit_id'],role=case['role'],index_a=i+1,index_b=j+1,
                                actual_gap_ps=true_gap,predicted_gap_ps=pred_gap,
                                actual_gap_percent=100*abs(true_gap)/float(yd.min()),
                                reversed=true_gap*pred_gap<0 and abs(true_gap)>.001 and abs(pred_gap)>.001))
                        if seed==20261117:
                            structures.append(dict(circuit_id=case['circuit_id'],role=case['role'],regions=len(model.partition.regions),
                                decisions=model.n_decisions,po_count=len(model.outputs),
                                zero_context_po_count=int((ctx.abs().amax((0,2))==0).sum()),
                                original_delay_ps=model.delay_reference,serialized_noop_delay_ps=case['serialized_noop_delay_ps'],
                                context_change_max=float((ctx-ctx[:1]).abs().max()),
                                fixed_critical_distinct=len(set(fixed.max(-1).values.tolist())),
                                actual_critical_distinct=len(set(yd.tolist()))))
                move_graph(model,'cpu')
        reproduction.append(dict(seed=seed,maximum_saved_prediction_error_ps=maximum_error,checkpoint_sha256=sha(checkpoint)))
        assert maximum_error<.001, reproduction[-1]
    for name,rs in [('combinations',rows),('loss_gradients',gradients),('pairs',pairs),('structures',structures)]:
        write_csv(a.output/(name+'.csv'),rs)
    summary=dict(reproduction=reproduction,prepared_sha256=sha(a.prepared),new_mapping_calls=0,new_training_runs=0,
                 scope='Existing train and previously inspected validation (now development). No independent test.')
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
