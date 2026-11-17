#!/usr/bin/env python3
"""Read-only integrity audit of the fixed paired table; never invokes a tool."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(panel,report):
    manifest=read(panel/'panel.json');barrier=read(panel/'PREDICTIONS_FROZEN.json')
    assert sha(panel/'panel.json')==barrier['panel_sha256']
    assert len(manifest['circuits'])==12 and len(barrier['rows'])==12
    assert read(panel/'preflight.json')['status']=='PASS'
    predictions={r['circuit_id']:r for r in barrier['rows']}
    with (report/'PREDICTED_VS_ABC_DELAY.csv').open() as stream:
        reader=csv.DictReader(stream)
        assert reader.fieldnames==['circuit','predicted_mapping_delay_ps','actual_abc_mapping_delay_ps']
        table=list(reader)
    assert len(table)==96 and len({r['circuit'] for r in table})==96
    table={r['circuit']:r for r in table};evidence=[];exceptions=[];passed=0
    for circuit in manifest['circuits']:
        folder=Path(circuit['folder']);anchor=read(folder/'anchor/result.json')
        prediction=read(folder/'predictions.json');frozen=read(folder/'frozen.json')
        assert sha(folder/'predictions.json')==predictions[circuit['circuit_id']]['sha256']
        assert sha(folder/'anchor/result.json')==prediction['anchor_sha256']
        assert sha(folder/'frozen.json')==prediction['frozen_sha256']
        assert sha(folder/'input.aig')==frozen['input_sha256']
        assert sha(folder/'normalized_original.aig')==anchor['input_sha256']==prediction['normalized_original_sha256']
        combos={c['index']:c for c in frozen['combinations'] if c['index']}
        assert len(prediction['rows'])==len(combos)==8
        for p in prediction['rows']:
            assert p['choices']==combos[p['index']]['choices'] and any(p['choices'])
            assert p['saved_unix_ns']<barrier['created_unix_ns']
            physical=[[max(0.,a[k]+c[j]-z[j]) for j,k in enumerate(['rise_ps','fall_ps'])]
                      for a,c,z in zip(anchor['po_rise_fall'],p['model_combo_po_ps'],p['model_noop_po_ps'])]
            assert math.isclose(max(map(max,physical)),p['predicted_mapping_delay_ps'],abs_tol=1e-9,rel_tol=1e-12)
            row=table[p['circuit']]
            assert float(row['predicted_mapping_delay_ps'])==p['predicted_mapping_delay_ps']
            path=folder/'measurements'/f'combo_{p["index"]:02d}'/'observation.json';o=read(path)
            assert o['prediction_file_sha256']==sha(folder/'predictions.json')
            assert o['replay_start_unix_ns']>barrier['created_unix_ns']
            assert o['choices']==p['choices']
            if o['status']=='PASS':
                m=o['metrics'];passed+=1
                assert m['cec']=='PASS' and m['timing_model']=='ABC_SCL_LIBERTY'
                assert o['execution']['applied_rewrites']==sum(bool(c) for c in p['choices'])
                assert o['symbolic_structure_matches']
                assert sha(path.parent/'normalized_candidate.aig')==p['symbolic_normalized_sha256']==m['input_sha256']
                assert float(row['actual_abc_mapping_delay_ps'])==m['delay_ps']
                assert m['provenance']['abc']['sha256']==anchor['provenance']['abc']['sha256']
                assert m['provenance']['library']['sha256']==anchor['provenance']['library']['sha256']
                assert o['area_feasible']==(m['area']<=anchor['area']*1.05)
            else:assert row['actual_abc_mapping_delay_ps']=='N/A'
            if o['status']!='PASS' or not o.get('area_feasible',False):
                exceptions.append({'circuit':p['circuit'],'status':o['status'],'area_feasible':o.get('area_feasible'),'error':o.get('error')})
            evidence.append({'circuit':p['circuit'],'observation':str(path),'sha256':sha(path),
                             'prediction_sha256':sha(folder/'predictions.json'),'normalized_input_sha256':o.get('input_sha256')})
    return {'status':'PASS','rows':len(table),'successful_pairs':passed,'exceptions':exceptions,
            'prediction_source_commit':barrier['source_commit'],'panel':str(panel),
            'prediction_freeze_sha256':sha(panel/'PREDICTIONS_FROZEN.json'),
            'csv_sha256':sha(report/'PREDICTED_VS_ABC_DELAY.csv'),'evidence':evidence,
            'recorded_mapping_commands':len(list(panel.glob('case_*/**/mapping.command.json'))),
            'preflight':read(panel/'preflight.json')}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--panel',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=verify(a.panel,a.report);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['evidence','preflight']},ensure_ascii=False))
