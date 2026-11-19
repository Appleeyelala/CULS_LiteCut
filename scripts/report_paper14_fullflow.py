"""Summarize accepted, rejected and failed raw proposals against matched baseline maps."""
import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

def read(path):
    return json.loads(path.read_text())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def report():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--references',required=True,type=Path)
    parser.add_argument('--run',required=True,action='append',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--progress',action='store_true',help='Include cases still running in an interim table.')
    args=parser.parse_args()
    output=args.output
    output.mkdir(exist_ok=True,parents=True)
    with (Path(__file__).resolve().parents[1]/'analysis/paper14_fullflow/circuits.csv').open() as f:
        manifest=list(csv.DictReader(f))
    rows=[]
    for entry in manifest:
        n=entry['circuit']
        ref=read(args.references/'output/refs'/n/'metrics.json')
        original,baseline,paper=(ref[k] for k in ('original','baseline_b','paper_litecut'))
        assert all(d['status']=='PASS' and d['cec']=='PASS' for d in (original,baseline,paper))
        assert original['input_sha256']==entry['sha256']
        attempts=[r for r in args.run if (r/'output/cases'/(n+'.attempt.json')).exists()]
        if not attempts:
            if not args.progress:
                raise ValueError('No completed attempt for '+n)
            rows.append(dict(circuit=n,status='PENDING',original_ps=original['delay_ps'],
                             baseline_b_ps=baseline['delay_ps'],litecut_ps=paper['delay_ps']))
            continue
        attempt_run=attempts[-1]
        case=attempt_run/'output/cases'/n
        attempt=read(case.with_suffix('.attempt.json'))
        row=dict(circuit=n,original_ps=original['delay_ps'],baseline_b_ps=baseline['delay_ps'],
                 litecut_ps=paper['delay_ps'],original_area=original['area'],baseline_b_area=baseline['area'],
                 litecut_area=paper['area'],baseline_b_seconds=ref['baseline_b_seconds'],wall_s=attempt['wall_s'],work=str(case))
        if attempt['exit_code']:
            row.update(status='FAILED',exit_code=attempt['exit_code'])
        else:
            d=read(case/'result.json')
            provenance=read(case/'inputs.json')
            assert provenance['original_sha256']==entry['sha256'] and provenance['alpha']==.5 and provenance['steps']==8
            assert sha(case/'selected.aig')==d['selected_aig_sha256'] and d['cec']=='PASS'
            assert abs(d['original_delay_ps']-original['delay_ps'])<1e-5
            accepted=d['status']=='ACCEPTED'
            if accepted:
                assert d['candidate_cec']=='PASS' and d['delay_ps']<d['original_delay_ps']
                selected=read(case/'final/result.json')
                assert selected['input_sha256']==d['selected_aig_sha256'] and selected['delay_ps']==d['delay_ps']
            else:
                assert d['selected_aig_sha256']==entry['sha256'] and d['delay_ps']==d['original_delay_ps']
            row.update(status=d['status'],selected_ps=d['delay_ps'],proposal_ps=d['candidate_delay_ps'],
                       selected_area=d['area_ratio']*original['area'],area_change_pct=100*(d['area_ratio']-1),
                       vs_original_pct=100*(d['delay_ps']/original['delay_ps']-1),
                       vs_b_pct=100*(d['delay_ps']/baseline['delay_ps']-1),
                       vs_litecut_pct=100*(d['delay_ps']/paper['delay_ps']-1),
                       cec=d['cec'],proposal_cec=d['candidate_cec'],edits=sum(bool(x) for x in d['vector']),
                       decisions=d['decisions'],search_s=d['search_seconds'],algorithm_s=d['elapsed_s'],
                       cuda_peak_gib=d['cuda_peak_bytes']/2**30,process_peak_gib=d['process_maxrss_kib']/2**20,
                       stages=d['stages'],selected_aig_sha256=d['selected_aig_sha256'],tool_failure=d['tool_failure'])
        rows.append(row)
    keys=list(dict.fromkeys(k for row in rows for k in row if k!='stages'))
    with (output/'results.csv').open('w') as f:
        w=csv.DictWriter(f,keys,extrasaction='ignore',lineterminator='\n');w.writeheader();w.writerows(rows)
    valid=[r for r in rows if r['status'] not in ('FAILED','PENDING')]
    summary={'total':len(rows),'completed':len(valid),'accepted':sum(r['status']=='ACCEPTED' for r in valid),
             'reference_run':str(args.references),'flow_runs':list(map(str,args.run)),
             'fallbacks':{s:sum(r['status']==s for r in valid) for s in sorted({r['status'] for r in valid})}}
    for ref in ('original_ps','baseline_b_ps','litecut_ps'):
        ratios=[r['selected_ps']/r[ref] for r in valid]
        summary[ref]={'faster':sum(x<1 for x in ratios),'same':sum(x==1 for x in ratios),
                      'slower':sum(x>1 for x in ratios),'geomean_ratio':statistics.geometric_mean(ratios)}
    summary['geomean_area_ratio']=statistics.geometric_mean(r['selected_area']/r['original_area'] for r in valid)
    summary['wall_s_median']=statistics.median(r['wall_s'] for r in valid)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (output/'stages.json').write_text(json.dumps({r['circuit']:r.get('stages') for r in rows},indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    report()
