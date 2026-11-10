#!/usr/bin/env python3
"""Freeze local predictions before opening existing candidate full-map labels.

This validation/export job invokes no ABC mapping or timing executable.
Run under Slurm in the staged /data experiment directory.
"""
import csv
import hashlib
import json
from pathlib import Path
import sys
from gate3b_analyze import parse_binary_aig

run=Path(sys.argv[1])
prior=Path('/data/wushangge/runs/CULS-mapper-internal-20261003-34U6OZ')
lines=(run/'local_prediction.log').read_text().splitlines()
assert not any('R9STOP' in s for s in lines)
assert 'R9CHECK baseline_local=PASS' in lines
ctx=[json.loads(s[6:]) for s in lines if s.startswith('R9CTX ')]
pred=[json.loads(s[8:]) for s in lines if s.startswith('R9LOCAL ')]
assert len(pred)==2 and [r['variant'] for r in pred]==['baseline','candidate']
assert len([r for r in ctx if r['type']=='meta'])==1
for row in ctx+pred:
    # On this Linux build Abc_Clock uses CLOCK_MONOTONIC, i.e. wall time.
    # Initial debug field names said cpu_s; normalize without rerunning map.
    for k in list(row):
        if k.endswith('_cpu_s'): row[k.replace('_cpu_s','_wall_s')]=row.pop(k)
for p in pred:
    assert not p['candidate_full_mapping_called'] and not p['downstream_matched']
    assert p['phase']==1
prevalidation=run/'local_predictions_prevalidation.json'
prevalidation.write_text(json.dumps(dict(predictions=pred,full_map_labels_read=False,
    prediction_input_files=['baseline AIG','single accepted candidate AIG','baseline initialized library','baseline mode0 frozen context']),indent=2)+'\n')
prediction_sha=hashlib.sha256(prevalidation.read_bytes()).hexdigest()
# Prediction output now exists and is immutable in this validation job.
# Only AFTER this write is the historical candidate full-map label opened.
reference_file=Path('/data/wushangge/runs/CULS-R9-pin-audit-20261014-iLFKEK/pin_audit_evidence.json')
ref=json.loads(reference_file.read_text())
refs={}
for s in ref['summary']:
    event=s['first_selections'][0]
    refs[s['variant']]=event['leaf_arrivals'][2]['worst']
assert refs==dict(baseline=109.23999,candidate=106.219986)
assert (run/'baseline_observed.blif').read_text().splitlines()[1:]==(prior/'output/baseline_original.blif').read_text().splitlines()[1:]
structure=json.loads((run/'frontier_structure.json').read_text())
baseline_pi=[r for r in ctx if r['type']=='pi']
assert len(baseline_pi)==41 and all(r['positive']['worst']==0 for r in baseline_pi)
def pi_frontier(path,node):
    g=parse_binary_aig(path); seen=set(); pis=set()
    def visit(n):
        if n in seen:return
        seen.add(n)
        if 1<=n<=g.pi_count:pis.add(n)
        for lit in g.fanin_literals.get(n,[]):visit(lit>>1)
    visit(node);return sorted(pis)
pi_ids=pi_frontier(prior/'input/c1355_orig.aig',213)
assert pi_ids==pi_frontier(prior/'input/S002.aig',322) and len(pi_ids)==14
context=dict(stage='initial_delay_oriented_mode0_before_cover_and_area_recovery',
    extraction_source='single normal baseline map only',metadata=next(r for r in ctx if r['type']=='meta'),
    pi_context=baseline_pi,compact_structural_frontier_context=[r for r in ctx if r['type']=='logical_frontier'],
    adopted_frontier='PI-closed R9 transitive fanin cone; NOT scalar-only compact frontier',
    adopted_pi_function_points=pi_ids,clock='Abc_Clock / CLOCK_MONOTONIC wall seconds',
    isolated_context_export_time=None,context_export_time_note='not measured independently; no extra baseline map run for timing',
    library=dict(name='asap7',redistributable_library_included=False,
        source_library_sha256='819ecfc2609a1f53b9d298bf0cba75af0f1a8c737f33d4bbc5c9fb1156e8fa55',
        implementation='native immutable Map_SuperLib_t initialized during baseline map, reused by local managers',
        full_table_serialized=False,baseline_root_best_supergate_arc_metadata=pred[0]['best']),
    rules=dict(native_functions=['Map_MappingCuts','Map_MappingTruths','Map_MappingMatches','Map_MatchNodeCut','Map_TimeCutComputeArrival','Map_MatchCompare','Map_NodeTryDroppingOnePhase','Map_NodeTransferArrivalTimes'],
        leaf_order='preserved original full subject-graph Map_Node_t.Num order',
        scalar_frontier_expansion_warning='native cuts can span earlier logical boundaries; PI closure avoids removing them'),
    state_policy=dict(frozen_external=['PI positive seeds','PI structural reference counts','library delay/match tables','mode0 mapping flags'],
        locally_recomputed=['cuts','truths','phase matches','rise/fall arrivals','area flow'],
        structurally_reconstructed=['full subject graph IDs/topological order','candidate internal structural fanout counts'],
        forbidden=['candidate full-map arrival/matches','downstream timing','area-recovery state','PO delay']),
    candidate_full_map_state_in_context=False)
(run/'R9_FROZEN_BASELINE_CONTEXT.json').write_text(json.dumps(context,indent=2)+'\n')
rows=[]
for p in pred:
    actual=refs[p['variant']]
    delta=p['arrival']-pred[0]['arrival'] if p['variant']=='candidate' else 0.0
    rows.append(dict(variant=p['variant'],function_point=p['function_point'],timing_phase='positive_mode0_all_PI',
        full_map_reference_arrival=actual,local_predicted_arrival=p['arrival'],predicted_delta=delta,
        absolute_error=abs(p['arrival']-actual),relative_error=abs(p['arrival']-actual)/abs(actual),
        local_nodes=p['local_ands'],evaluated_cuts=p['evaluated_cuts'],evaluated_matches=p['evaluated_matches'],
        evaluated_phases=p['evaluated_phases'],elapsed_time=p['elapsed_wall_s'],match_wall_time=p['match_wall_s'],
        frontier_pis=p['frontier_pis'],subject_graph_nodes_constructed=p['subject_graph_nodes_constructed'],
        full_candidate_mapping=False,below_108_59_ps=p['arrival']<108.59))
with (run/'R9_LOCAL_ARRIVAL_PREDICTION.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(run/'local_arrival_validation.json').write_text(json.dumps(dict(rows=rows,prediction_sha256=prediction_sha,
    reference_read_after_prediction_persisted=True,reference_file=str(reference_file),
    normal_baseline_maps=1,normal_candidate_maps=0,baseline_instrumentation_blif_identical=True,
    baseline_initial_mode0_preprocessing_wall_s=sum(context['metadata'][k] for k in ['baseline_cut_wall_s','baseline_truth_wall_s','baseline_match_wall_s']),
    preprocessing_time_excludes_library_loading_and_build=True,frontier_mapping=structure['mapping']),indent=2)+'\n')
print(json.dumps(rows,indent=2))
