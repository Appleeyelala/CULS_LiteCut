#!/usr/bin/env python3
"""Read saved STE100 records and derive diagnostic tables; never run tools or optimization.

Usage: python -B recompute_tables.py --output /data/.../new-empty-directory
The only imported project routine parses an AIG; no torch/model/optimizer import.
"""
import argparse,collections,csv,hashlib,json,math,statistics
from pathlib import Path
from saved_proxy_sensitivity import inspect,RUN

COMMIT='9a199bf0ea0dcf20bf55e40da8caa8b8bbff5fac'
SOURCE={}
def read(p):
 b=p.read_bytes();SOURCE[str(p)]={'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}
 return json.loads(b)
def evidence(p):
 b=p.read_bytes();SOURCE[str(p)]={'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}
def dump(p,data):p.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csvout(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def outcome(a,b):
 t=max(2e-5,2e-6*max(abs(a),abs(b)));return 'tie' if abs(a-b)<=t else 'win' if a<b else 'loss'
def comparison(pairs):
 c=collections.Counter(outcome(a,b) for a,b in pairs)
 return {'pairs':len(pairs),'counts':{x:c[x] for x in ('win','tie','loss')},'delay_ratio_geomean':math.exp(statistics.mean(math.log(a/b) for a,b in pairs if min(a,b)>0))}

def main(out):
 out.mkdir(parents=True,exist_ok=False)
 status=read(RUN/'output/STE100_STATUS.json');sample=read(RUN/'output/STE100_SAMPLE_MANIFEST.json');points={(x['index'],x['method']):x for x in status['points']};records={}
 assert status['source_commit']==COMMIT and len(points)==300 and len(sample['rows'])==100
 for item in sample['rows']:
  for m in ('A','B','Ours'):
   x=read(RUN/'output/records'/f"{item['index']:04d}_{m}.json");assert x['status']=='PASS' and x['input_sha256']==item['sha256'];records[item['index'],m]=x
 cases=[];qor=[];runtime=[];features=[];all100=[];trajectory_summary=[]
 for item in sample['rows']:
  i=item['index'];w=RUN/'output'/f'case_{i:04d}_Ours/work';s=read(w/'summary.json');f=s['final_candidate'];best=s['proxy_best'];full=records[i,'Ours']['end_to_end_runtime_s'];second=0.;clock_rel='';tr=None
  assert s['selected']['metrics']['delay_ps']==records[i,'Ours']['metrics']['delay_ps']
  if s['adam_steps']:
   t=read(w/'optimization/trajectory.json');best=read(w/'optimization/proxy_best.json');tr=t['trajectory'];assert len(tr)==40 and t['proxy_parameters_frozen']
   trajectory_summary.append({'circuit_id':i,'source':s['selected']['source'],'nonempty_steps':sum(bool(x['hard_edits']) for x in tr),'first_nonempty_step':next((x['step'] for x in tr if x['hard_edits']),None),'host_rows_max':max(x['host_candidate_rows'] for x in tr),'hard_edits_max':max(len(x['hard_edits']) for x in tr),'logits_total_change_norm':t['logits_total_change_norm'],'initial_loss':tr[0]['loss'],'best_loss':best['predicted_objective']})
  if f:
   cp=w/'final/eval_0000/evaluation/mapping.command.json';clock=read(cp);assert clock['returncode']==0 and not clock.get('timed_out');second=clock['elapsed_s'];clock_rel=str(cp.relative_to(RUN))
   assert f['status']=='PASS' and f['metrics']['cec']=='PASS' and f['choices']==best['choices'];rp=read(w/'final/eval_0000/rewrite/replay.json');assert not rp['rejected'] and rp['applied_rewrites']==len(rp['requested'])
   assert f['metrics']['area']<=s['original']['area']*1.05
   row={'circuit_id':i,'input':item['relative_path'],'candidate_delay_ps':f['metrics']['delay_ps'],'own_original_delay_ps':s['original']['delay_ps'],'A_delay_ps':records[i,'A']['metrics']['delay_ps'],'B_delay_ps':records[i,'B']['metrics']['delay_ps'],'historical_selected_delay_ps':records[i,'Ours']['metrics']['delay_ps'],'historical_final_source':s['selected']['source'],'candidate_cec':f['metrics']['cec'],'installed_edits':rp['applied_rewrites'],'requested_edits':len(rp['requested']),'candidate_area_ratio':f['metrics']['area']/s['original']['area'],'candidate_vs_original':outcome(f['metrics']['delay_ps'],s['original']['delay_ps']),'proxy_initial_objective':tr[0]['loss'],'proxy_best_objective':best['predicted_objective'],'proxy_objective_delta':best['predicted_objective']-tr[0]['loss'],'proxy_initial_delay_ps':tr[0]['predicted_delay_ps'],'proxy_best_delay_ps':best['delay_ps'],'proxy_delay_vs_initial':outcome(best['delay_ps'],tr[0]['predicted_delay_ps']),'summary_evidence':str((w/'summary.json').relative_to(RUN))}
   assert row['proxy_objective_delta']<0;qor.append(row)
  assert s['mapping_calls']==1+bool(f)
  rt={'circuit_id':i,'input':item['relative_path'],'measured_full_runtime_s':full,'second_mapping_abc_time_s':second,'optimization_runtime_s':full-second,'A_measured_full_runtime_s':records[i,'A']['end_to_end_runtime_s'],'B_measured_full_runtime_s':records[i,'B']['end_to_end_runtime_s'],'Ours_mapping_calls':s['mapping_calls'],'second_mapping_clock':clock_rel,'timing_scope':'Recorded cold full wall minus final ABC command wall; not measured one-map deployment'}
  assert full>=second>=0;runtime.append(rt)
  all100.append({'id':i,'raw_policy_delay':f['metrics']['delay_ps'] if f else s['original']['delay_ps'],'A':records[i,'A']['metrics']['delay_ps'],'B':records[i,'B']['metrics']['delay_ps']})
  if s['selected']['source']!='original_noop':continue
  manifest=read(w/'candidates/raw/gate3c_raw_manifest.json');cat=read(w/'candidates/catalogue.json');audit=read(w/'cut_validation/audit.json')
  row={'circuit_id':i,'input':item['relative_path'],'group':'NO_CANDIDATE' if not s['decisions'] else 'CANDIDATE_BUT_NOOP','and_nodes':manifest['n_nodes'],'regions':s['regions'],'eligible_exported_cuts':manifest['eligible_cut_count'],'raw_executable_decisions':manifest['executable_decision_count'],'raw_candidates':manifest['candidate_count'],'boolean_excluded_decisions':audit['excluded_decisions'],'verified_decisions':audit['verified_decisions'],'feasibility_blocked_choices':None,'available_nonzero_candidates':None,'adam_steps':s['adam_steps'],'gradient_norm_min':None,'gradient_norm_max':None,'logits_total_change_norm':None,'sum_step_update_norms':None,'host_candidate_rows_max':None,'hard_edits_max':None,'visited_hard_combinations':None,'proxy_loss_changed':None,'negative_q_candidates':None,'zero_q_candidates':None,'positive_q_candidates':None,'most_negative_analytic_noop_derivative':None,'largest_positive_analytic_noop_derivative':None,'cause_code':'ZERO_GPU_EXPORTED_CANDIDATE_MASK','evidence_level':'export_boundary_confirmed; per-subgraph rejection_reason_not_recorded','evidence_prefix':str(w.relative_to(RUN))}
  assert manifest['candidate_count']==cat['candidate_count'] and manifest['executable_decision_count']==cat['executable_decisions'] and audit['verified_decisions']==s['decisions']
  assert audit['excluded_decisions']==0
  if not s['decisions']:
   assert cat['candidate_count']==0 and not (w/'candidate_feasibility.json').exists()
  else:
   fe=read(w/'candidate_feasibility.json');v=read(w/'cut_validation/verified_catalogue.json');read(w/'partition/partition.json');read(w/'litecut.json');lp=next((w/'litecut_cache').glob('*.json'));read(lp);evidence(w.parent/'input.aig')
   fs=inspect(i);features.extend(fs)
   negatives=[x['noop_choice_derivative'] for x in fs if x['q']<0];positives=[x['noop_choice_derivative'] for x in fs if x['q']>0]
   row.update(feasibility_blocked_choices=len(fe['blocked_choices']),available_nonzero_candidates=s['candidates']-len(fe['blocked_choices']),gradient_norm_min=min(x['logits_gradient_norm'] for x in tr),gradient_norm_max=max(x['logits_gradient_norm'] for x in tr),logits_total_change_norm=t['logits_total_change_norm'],sum_step_update_norms=sum(x['adam_update_norm'] for x in tr),host_candidate_rows_max=max(x['host_candidate_rows'] for x in tr),hard_edits_max=max(len(x['hard_edits']) for x in tr),visited_hard_combinations=t['distinct_hard_combinations_including_noop'],proxy_loss_changed=any(x['loss']!=tr[0]['loss'] for x in tr),negative_q_candidates=len(negatives),zero_q_candidates=sum(x['q']==0 for x in fs),positive_q_candidates=len(positives),most_negative_analytic_noop_derivative=min(negatives,default=0),largest_positive_analytic_noop_derivative=max(positives,default=0))
   assert len(fe['blocked_choices'])==0 and row['host_candidate_rows_max']==0 and row['hard_edits_max']==0 and row['visited_hard_combinations']==1 and not row['proxy_loss_changed']
   if i in (242,637):row.update(cause_code='IDENTICAL_FEATURES_PROXY_FLAT',evidence_level='confirmed_equal_features_scores_and_zero_gradient')
   elif i==643:row.update(cause_code='NO_DESCENT_IN_FROZEN_PROXY',evidence_level='all_q_nonnegative; monotone_timing_and_nonnegative_area_penalty')
   elif i==836:row.update(cause_code='ATTENUATED_GRADIENT_NO_FLOAT64_UPDATE',evidence_level='trace_confirmed; analytic_proxy_sensitivity_supports_cause')
   elif i in (546,571,658,744):row.update(cause_code='WEAK_GRADIENT_NO_ACTIVATION',evidence_level='trace_confirmed_small_updates; activation_margins_not_saved')
   else:row.update(cause_code='MIXED_PROXY_SIGNAL_NO_ACTIVATION',evidence_level='strong_bad_and_weak_good_proxy_derivatives; exact_probability_paths_not_saved')
  cases.append(row)
 assert len(cases)==33 and sum(x['group']=='NO_CANDIDATE' for x in cases)==17 and len(qor)==67 and len(runtime)==100
 csvout(out/'NO_REWRITE_CASES.csv',cases);csvout(out/'RAW_CANDIDATE_QOR.csv',qor);csvout(out/'RUNTIME_ACCOUNTING.csv',runtime);csvout(out/'SAVED_PROXY_SENSITIVITY.csv',features);csvout(out/'TRAJECTORY_OVERVIEW.csv',trajectory_summary)
 pair67={m:comparison([(x['candidate_delay_ps'],x[m+'_delay_ps']) for x in qor]) for m in ('A','B')};pair67['own_original']=comparison([(x['candidate_delay_ps'],x['own_original_delay_ps']) for x in qor])
 rtstats={k:{'median':statistics.median(x[k] for x in runtime),'sum':sum(x[k] for x in runtime)} for k in ('measured_full_runtime_s','second_mapping_abc_time_s','optimization_runtime_s')}
 ratios={m:{'optimization_over_baseline_median':statistics.median(x['optimization_runtime_s']/x[m+'_measured_full_runtime_s'] for x in runtime),'full_over_baseline_median':statistics.median(x['measured_full_runtime_s']/x[m+'_measured_full_runtime_s'] for x in runtime)} for m in ('A','B')}
 stats={'formal_source_commit':COMMIT,'case_counts':dict(collections.Counter(x['cause_code'] for x in cases)),'raw67_pairs':pair67,'raw67_delay_median_ps':statistics.median(x['candidate_delay_ps'] for x in qor),'raw67_own_original_delay_median_ps':statistics.median(x['own_original_delay_ps'] for x in qor),'raw67_proxy_prediction_counts':dict(collections.Counter(x['proxy_delay_vs_initial'] for x in qor)),'proxy_actual_crosstab':dict(collections.Counter(x['proxy_delay_vs_initial']+'->'+x['candidate_vs_original'] for x in qor)),'posthoc_no_fallback_all100_pairs':{m:comparison([(x['raw_policy_delay'],x[m]) for x in all100]) for m in ('A','B')},'runtime_stats':rtstats,'runtime_ratios':ratios,'second_mapping_median_among_67_s':statistics.median(x['second_mapping_abc_time_s'] for x in runtime if x['second_mapping_abc_time_s']>0),'missing_mapping_clocks':0,'scope':'All experimental measurements are existing records. SAVED_PROXY_SENSITIVITY is independent scalar arithmetic at no-op from saved features/topology, not a recorded per-candidate gradient, not a CUDA/model/optimizer rerun. Does not reconstruct missing logits/probabilities.'}
 dump(out/'DIAGNOSTIC_STATS.json',stats)
 for path in ['global_timing/mffc_ste_search.py','global_timing/mffc_ste_experiment.py','global_timing/candidates.py','global_timing/cut_validation.py','global_timing/feasibility.py','global_timing/mffc_model.py','global_timing/mffc_gpu.py','global_timing/surrogate.py','global_timing/mffc_search.py','global_timing/common.py','global_timing/structure.py','projects/culs/src/algorithms/rewrite.cu','projects/culs/tools/gate3b_analyze.py','framework/mffc_gpu/region_lse.cu']:evidence(RUN/'src'/path)
 dump(out/'EVIDENCE_INDEX.json',{'formal_run':str(RUN),'formal_source_commit':COMMIT,'source_files':SOURCE,'published_data_files':{p.name:{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in out.iterdir() if p.is_file()},'verification':{'sample_circuits':100,'method_records':300,'no_rewrite_rows':33,'raw_candidate_rows':67,'runtime_rows':100,'no_rewrite_boolean_exclusions':0,'no_rewrite_feasibility_blocks':0,'no_rewrite_positive_margin_steps':0,'missing_second_mapping_clocks':0},'scope':'Original input AIGs read only to recover topology; only file hashes and derived scalars published. No original files changed; no new ABC/CUDA/inference/optimizer execution. Markdown and analysis scripts excluded from this generated data manifest.'})
 print(json.dumps(stats,ensure_ascii=False,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();main(args.output)
