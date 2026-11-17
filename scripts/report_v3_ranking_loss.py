#!/usr/bin/env python3
"""Paired ranking report for the frozen loss-only original-v3 experiment."""
import argparse
import csv
import itertools
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from evaluate_v3_ranking import evaluate, aggregate, read, sha, write_csv


def summarize(rows):
    result=aggregate(rows)
    for key in ['severe_pairs','severe_reversals','severe_predicted_ties']:
        result[key+'_count']=sum(r[key] for r in rows)
    result['negative_spearman_count']=sum(r['spearman'] is not None and r['spearman']<0 for r in rows)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-run',type=Path,required=True)
    p.add_argument('--old-run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    protocol_path=ROOT/'analysis/joint_gradient_v3/ranking_loss/PROTOCOL.json'
    protocol=read(protocol_path)
    metric_protocol=read(ROOT/protocol['evaluation_protocol'])
    manifest_path=ROOT/'analysis/joint_gradient_v3/learning_manifest.json'
    manifest=read(manifest_path)
    names={c['circuit_id']:c['input'] for c in manifest['selected']}
    files=[protocol_path,ROOT/protocol['evaluation_protocol'],manifest_path]
    rows=[];combo_rows=[];seeds=[];training=[];pair_dict={}
    for seed in protocol['seeds']:
        old_path=a.old_run/'output'/f'global_{seed}'/'summary.json'
        new_path=a.training_run/'output'/f'gap_ranking_{seed}'/'summary.json'
        old,new=read(old_path),read(new_path)
        assert new['global_context']=='original' and new['loss_mode']=='gap-ranking'
        assert old['variant']==new['variant']=='global'
        assert old['manifest_sha256']==new['manifest_sha256']==sha(manifest_path)
        files.extend([old_path,new_path])
        training.append(dict(seed=seed,old_best_epoch=old['best_epoch'],new_best_epoch=new['best_epoch'],
            old_epochs=len(old['history']),new_epochs=len(new['history']),
            old_train_loss=old['train']['mean_circuit_loss'],new_train_loss=new['train']['mean_circuit_loss'],
            old_development_loss=old['validation']['mean_circuit_loss'],new_development_loss=new['validation']['mean_circuit_loss']))
        for role,key in [('train','train'),('development','validation')]:
            ids=[c['circuit_id'] for c in manifest['selected'] if c['role']==key]
            expected={(cid,i) for cid in ids for i in range(1,9)}
            old_predictions={(r['circuit_id'],r['index']):r for r in old[key]['predictions'] if r['domain']=='sampled'}
            new_predictions={(r['circuit_id'],r['index']):r for r in new[key]['predictions'] if r['domain']=='sampled'}
            assert set(old_predictions)==set(new_predictions)==expected
            for k in expected:
                assert old_predictions[k]['actual_delay_ps']==new_predictions[k]['actual_delay_ps']
            for label,predictions in [('v3_global',old_predictions),('gap_ranking',new_predictions)]:
                group=[]
                for cid in ids:
                    data=[predictions[cid,i] for i in range(1,9)]
                    pred=[r['predicted_delay_ps'] for r in data];actual=[r['actual_delay_ps'] for r in data]
                    m,pr,ar,pairs=evaluate(pred,actual,list(range(1,9)),metric_protocol['absolute_tie_tolerance_ps'],metric_protocol['near_optimal_percent'])
                    m['severe_pairs']=m['severe_reversals']=m['severe_predicted_ties']=0
                    for pair in pairs:
                        i,j=pair['index_a']-1,pair['index_b']-1
                        gap=100*abs(actual[i]-actual[j])/min(actual)
                        severe=gap>5 and ar[i]!=ar[j]
                        m['severe_pairs']+=int(severe)
                        m['severe_reversals']+=int(severe and pair['classification']=='wrong')
                        m['severe_predicted_ties']+=int(severe and pair['classification']=='predicted_tie_only')
                        pair_dict[role,seed,cid,i+1,j+1,label]=dict(classification=pair['classification'],actual_gap_percent=gap,severe=severe)
                    row=dict(role=role,model=label,seed=seed,circuit_id=cid,input=names[cid],
                             selected_predicted_delay_ps=pred[m['top1_index']-1],**m)
                    rows.append(row);group.append(row)
                    for i in range(8):
                        combo_rows.append(dict(role=role,model=label,seed=seed,circuit_id=cid,index=i+1,
                            predicted_delay_ps=pred[i],actual_delay_ps=actual[i],predicted_rank=pr[i],true_rank=ar[i]))
                seeds.append(dict(role=role,model=label,seed=seed,**summarize(group)))
    overall=[]
    for role in ['train','development']:
        for label in ['v3_global','gap_ranking']:
            overall.append(dict(role=role,model=label,**summarize([r for r in rows if r['role']==role and r['model']==label])))
    old,new=[r for r in overall if r['role']=='development']
    assert abs(old['pairwise_accuracy_macro']-.5491029032695699)<1e-12
    assert abs(old['top1_regret_percent_mean']-1.629253233754699)<1e-12
    old_seeds={r['seed']:r for r in seeds if r['role']=='development' and r['model']=='v3_global'}
    new_seeds={r['seed']:r for r in seeds if r['role']=='development' and r['model']=='gap_ranking'}
    old_cases={(r['seed'],r['circuit_id']):r for r in rows if r['role']=='development' and r['model']=='v3_global'}
    new_cases={(r['seed'],r['circuit_id']):r for r in rows if r['role']=='development' and r['model']=='gap_ranking'}
    new_severe=[dict(seed=k[0],circuit_id=k[1],old_regret_percent=old_cases[k]['top1_regret_percent'],
                    new_regret_percent=new_cases[k]['top1_regret_percent']) for k in old_cases
                if old_cases[k]['top1_regret_percent']<=5 and new_cases[k]['top1_regret_percent']>5]
    gate=dict(every_seed_pairwise_better=all(new_seeds[s]['pairwise_accuracy_macro']>old_seeds[s]['pairwise_accuracy_macro'] for s in old_seeds),
        top1_not_lower=new['top1_hit_mean']>=old['top1_hit_mean'],
        every_seed_mean_regret_not_higher=all(new_seeds[s]['top1_regret_percent_mean']<=old_seeds[s]['top1_regret_percent_mean'] for s in old_seeds),
        mean_regret_lower=new['top1_regret_percent_mean']<old['top1_regret_percent_mean'],
        max_regret_not_higher=new['top1_regret_percent_max']<=old['top1_regret_percent_max'],
        severe_reversals_not_more=new['severe_reversals_count']<=old['severe_reversals_count'],
        no_new_regret_above_five_percent=not new_severe)
    paired=[]
    for key,values in pair_dict.items():
        if key[-1]!='v3_global':continue
        newvalues=pair_dict[key[:-1]+('gap_ranking',)]
        paired.append(dict(role=key[0],seed=key[1],circuit_id=key[2],index_a=key[3],index_b=key[4],
            actual_gap_percent=values['actual_gap_percent'],severe=values['severe'],
            old_classification=values['classification'],new_classification=newvalues['classification']))
    a.output.mkdir(parents=True,exist_ok=False)
    for name,rs in [('overall_metrics',overall),('seed_metrics',seeds),('per_circuit',rows),
                    ('all_combinations',combo_rows),('pair_changes',paired),('training_accounting',training)]:
        write_csv(a.output/(name+'.csv'),rs)
    evidence=dict(protocol=protocol,training_run=str(a.training_run),old_run=str(a.old_run),
        source_commit=read(a.training_run/'run.json')['source_commit'],
        counts=dict(train_circuits=12,development_circuits=12,independent_circuits=0,
            combinations_per_circuit=8,seeds=3,new_mapping_calls=0,missing_records=0),
        files=[dict(path=str(f),sha256=sha(f)) for f in files],
        output_sha256={f.name:sha(f) for f in a.output.glob('*.csv')},
        new_severe_regret_cases=new_severe,
        advancement_gate=gate,consider_independent_confirmation=all(gate.values()))
    (a.output/'evidence_index.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps(dict(development=[old,new],gate=gate),indent=2))


if __name__=='__main__':main()
