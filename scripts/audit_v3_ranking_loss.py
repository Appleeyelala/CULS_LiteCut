#!/usr/bin/env python3
"""Separate fitting and ranking gradients at the three original v3 checkpoints."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from train_joint_gradient_ports import load_prepared_graphs, move_graph, timing_loss
from evaluate_v3_ranking import read, sha, write_csv


def main():
    import torch
    import torch.nn.functional as F
    from global_timing.ranking_loss import gap_ranking_loss, true_tie_groups
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',type=Path,required=True);p.add_argument('--old-run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1)
    protocol=read(ROOT/'analysis/joint_gradient_v3/ranking_loss/PROTOCOL.json')
    assert sha(a.prepared)==protocol['prepared_sha256']
    prepared=load_prepared_graphs(a.prepared)
    for name,digest in prepared['code_hashes'].items():assert sha(ROOT/'global_timing'/name)==digest
    examples=prepared['examples'];shared=examples[0]['domains'][0]['model'].shared;shared.cuda()
    parameters=list(shared.global_head.parameters());rows=[];summaries=[]
    def cosine(x,y):
        n=float(x.norm()*y.norm())
        return float(x.dot(y))/n if n else None
    for seed in protocol['seeds']:
        path=a.old_run/'output'/f'global_{seed}'/'weights.pt'
        shared.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['state_dict'])
        saved=read(ROOT/f'analysis/joint_gradient_v3/pilot/global_{seed}_validation.json')
        expected={(r['circuit_id'],r['index']):r['predicted_delay_ps'] for r in saved['predictions'] if r['domain']=='sampled'}
        max_error=0.;totals={}
        for case in examples:
            total_records=sum(len(d['x']) for d in case['domains'])
            for domain in case['domains']:
                model=move_graph(domain['model'],'cuda');model.variant='global'
                x=domain['x'].cuda();target=domain['target'].cuda()
                po,_=model(x);scale=.01*model.delay_reference
                pred=po.max(-1).values;actual=target.max(-1).values
                fitting=F.smooth_l1_loss((po-model.baseline_po[None])/scale,
                    (target-model.baseline_po[None])/scale,beta=1.)
                fitting=fitting+F.smooth_l1_loss((pred-model.delay_reference)/scale,
                    (actual-model.delay_reference)/scale,beta=1.)
                original=timing_loss(po,target,model.baseline_po,model.delay_reference)
                ranking=gap_ranking_loss(po,target,model.baseline_po,model.delay_reference)
                def grad(loss):
                    return torch.cat([g.flatten() for g in torch.autograd.grad(loss,parameters,retain_graph=True)])
                gf,go,gr=grad(fitting),grad(original),grad(ranking)
                old_pair=go-gf
                groups=true_tie_groups(actual);i,j=torch.triu_indices(len(x),len(x),offset=1,device=x.device)
                weights=((actual[i]-actual[j]).abs()/scale)[groups[i]!=groups[j]].clamp_max(1.)
                rows.append(dict(seed=seed,circuit_id=case['circuit_id'],role='development' if case['role']=='validation' else 'train',
                    domain=domain['name'],records=len(x),true_distinct_pairs=len(weights),capped_pairs=int((weights==1).sum()),
                    pair_weight_sum=float(weights.sum()),old_loss=float(original),fitting_loss=float(fitting),new_ranking_loss=float(ranking),
                    fitting_grad_norm=float(gf.norm()),old_pair_grad_norm=float(old_pair.norm()),
                    old_total_grad_norm=float(go.norm()),new_ranking_grad_norm=float(gr.norm()),
                    fitting_vs_new_ranking_cosine=cosine(gf,gr),old_total_vs_new_ranking_cosine=cosine(go,gr)))
                if case['role']=='train':
                    for key,g in [('fitting',gf),('original',go),('ranking',gr)]:
                        contribution=g.detach()*len(x)/total_records/12
                        totals[key]=totals.get(key,torch.zeros_like(g))+contribution
                if case['role']=='validation' and domain['name']=='sampled':
                    for k,r in enumerate(domain['records']):
                        max_error=max(max_error,abs(float(pred[k])-expected[case['circuit_id'],r['index']]))
                move_graph(model,'cpu')
        assert max_error==0.,max_error
        summaries.append(dict(seed=seed,maximum_saved_prediction_error_ps=max_error,checkpoint_sha256=sha(path),
            equal_circuit_aggregate_fitting_vs_ranking_cosine=cosine(totals['fitting'],totals['ranking']),
            equal_circuit_aggregate_old_total_vs_ranking_cosine=cosine(totals['original'],totals['ranking'])))
    a.output.mkdir(parents=True,exist_ok=False);write_csv(a.output/'gradient_components.csv',rows)
    summary=dict(checkpoints=summaries,prepared_sha256=protocol['prepared_sha256'],
        environment=dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name()),new_mapping_calls=0,new_training_runs=0)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
