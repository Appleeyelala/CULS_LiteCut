#!/usr/bin/env python3
"""Verify delivered identities, counts, gain sums and runtime accounting."""
import argparse
import csv
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from run_nonpositive_coverage import ZERO17


def rows(path):
    return list(csv.DictReader(path.open()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('delivery',type=Path)
    args=p.parse_args();d=args.delivery
    sample=json.loads((ROOT/'analysis/ste_adam_100/config/sampled_100_manifest.json').read_text())['rows']
    expected={r['index']:r['sha256'] for r in sample}
    r100=rows(d/'CANDIDATE_COUNTS_100.csv')
    assert len(r100)==100 and {int(r['circuit_id']):r['input_sha256'] for r in r100}==expected
    for r in r100:
        if r['status']=='PASS':
            assert int(r['gain_rule_rejected'])==0
            assert int(r['subgraphs_considered'])==int(r['self_root_reuse_rejected'])+int(r['gain_rule_rejected'])+int(r['new_raw_candidates'])
        for stage in ('raw','verified','feasible'):
            n=r[f'new_{stage}_candidates']
            if n:
                assert int(n)==sum(int(r[f'new_{stage}_{sign}']) for sign in ('positive','zero','negative'))
        if int(r['circuit_id']) in ZERO17:
            assert int(r['old_raw_candidates'])==0
    stats=json.loads((d/'COVERAGE_STATS.json').read_text())
    solved=sum(all(r[f'new_{s}_candidates'] and int(r[f'new_{s}_candidates'])>0 for s in ('raw','verified','feasible')) for r in r100 if int(r['circuit_id']) in ZERO17)
    assert stats['original17_resolved']==solved
    full=d/'full905'
    if full.exists():
        assert solved==17 and stats['GO_905']
        c=rows(full/'FULL905_CANDIDATE_COUNTS.csv');r=rows(full/'FULL905_OURS_RESULTS.csv')
        assert len(c)==len(r)==905
        assert {int(x['circuit_id']) for x in c}=={int(x['circuit_id']) for x in r}==set(range(905))
        inventory=json.loads((ROOT/'benchmark905_manifest.json').read_text())['rows']
        identities={x['index']:(x['relative_path'],x['sha256']) for x in inventory}
        for a,b in zip(c,r):
            assert a['circuit_id']==b['circuit_id'] and a['input_sha256']==b['input_sha256']
            assert identities[int(a['circuit_id'])]==(a['input'],a['input_sha256'])
            assert a['input']==b['input']
            for stage in ('raw','verified','feasible'):
                signs=[a[f'{stage}_{sign}'] for sign in ('positive','zero','negative')]
                if all(signs):
                    assert int(a[f'{stage}_candidates'])==sum(map(int,signs))
            if a['gain_rule_rejected']:
                assert int(a['gain_rule_rejected'])==0
                assert int(a['subgraphs_considered'])==int(a['self_root_reuse_rejected'])+int(a['raw_candidates'])
            if a['feasible_candidates']:
                assert (int(a['feasible_candidates'])>0)==(b['has_legal_candidate']=='True')
            if b['optimization_runtime_s']:
                assert b['second_mapping_abc_time_s'] and b['measured_full_runtime_s']
                assert all(math.isfinite(float(b[k])) and float(b[k])>=0 for k in ('optimization_runtime_s','second_mapping_abc_time_s','measured_full_runtime_s'))
                assert math.isclose(float(b['optimization_runtime_s']),float(b['measured_full_runtime_s'])-float(b['second_mapping_abc_time_s']),abs_tol=1e-7)
            if b['candidate_delay_ps']:
                assert b['cec']=='PASS' and b['replay_status']=='PASS'
            if b['mapping_calls']:
                assert int(b['mapping_calls'])+int(b['mapping_calls_with_missing_clock'])<=2
        status=json.loads((full/'FULL905_RUN_STATUS.json').read_text())
        assert status['completed']==sum(x['status']!='NOT_COMPLETED' for x in r)
        assert status['zero_candidates']==sum(x['has_legal_candidate']=='False' for x in c)
        assert len(status['old94_resolved'])+len(status['old94_still_zero'])+len(status['old94_unknown'])==94
    for path in d.rglob('*'):
        if path.is_file():
            assert path.suffix in ('.md','.json','.csv'),path
            assert path.stat().st_size<10*1024*1024,path
    print(json.dumps({'status':'PASS','fixed100':100,'original17_resolved':solved,'full905_present':full.exists()}))

if __name__=='__main__':main()
