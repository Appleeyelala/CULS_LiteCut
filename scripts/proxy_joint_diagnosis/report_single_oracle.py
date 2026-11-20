"""Verify and aggregate the three measured-edit cases from their Slurm work directories."""
import argparse
import json
from math import fsum, isclose
from pathlib import Path

from diagnostic_io import read_csv, read_json, sha256, write_csv


def export(out, runs):
    out.mkdir(parents=True, exist_ok=True)
    singles, arms, sources = [], [], []
    for work in runs:
        name = work.name
        manifest_path = next(d/'run.json' for d in work.parents if (d/'run.json').exists())
        manifest = read_json(manifest_path)
        assert manifest['state'] == 'COMPLETED' and manifest['exit_code'] == 0
        rows = read_csv(work/'oracle_single_edits.csv')
        selected = read_json(work/'selections.json')
        by_method = read_csv(work/'oracle_arms.csv')
        assert len(rows) == (10 if name == 'cabac_ucontext' else 16)
        assert len(by_method) == len({a['group'] for a in by_method}) == 8
        assert all(a['circuit'] == name for a in rows + by_method)
        for arm in by_method:
            tag = arm['group']
            if tag in selected:
                ps = sorted(tuple(p) for p in selected[tag])
                # Each known positive-gain upper bound was feasible on this cohort.
                positives = sorted((int(s['decision']), int(s['choice'])) for s in rows if float(s[tag+'_gain_ps'])>0)
                assert ps == positives
                assert ps == sorted(map(tuple, json.loads(arm['pairs'])))
                assert ps == sorted((int(s['decision']),int(s['choice'])) for s in rows if s[tag+'_chosen']=='True')
                gains = [float(s[tag+'_gain_ps']) for s in rows if s[tag+'_chosen']=='True']
                assert isclose(float(arm['additive_gain_ps']), fsum(gains))
            p = Path(arm['work'])
            measurement = read_json(p/('final/result.json' if tag=='original_adam' else 'evaluation/result.json'))
            assert arm['cec'] == measurement['cec'] == 'PASS'
            assert float(arm['raw_ps']) == measurement['delay_ps']
            assert arm['input_sha256'] == measurement['input_sha256']
        arms.extend(by_method)
        singles.extend(rows)
        sources.append(dict(circuit=name, work=str(work), manifest=str(manifest_path), job=manifest['slurm_job_id'],
                            commit=manifest['source_commit'], state=manifest['state'], command=manifest['command'],
                            resources=manifest['resources'], source_file_sha256=sha256(work/'source.json'),
                            selections_sha256=sha256(work/'selections.json'),
                            arms_sha256=sha256(work/'oracle_arms.csv'), singles_sha256=sha256(work/'oracle_single_edits.csv')))
    assert {r['circuit'] for r in arms} == {'c5315','intra_pred_PE','cabac_ucontext'} and len(arms)==24 and len(singles)==42
    write_csv(out/'oracle_single_edits.csv', singles)
    write_csv(out/'oracle_arms.csv', arms)
    with (out/'oracle_sources.json').open('x') as stream:
        json.dump(sources,stream,indent=2)
        stream.write('\n')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('work',type=Path,nargs=3)
    args=parser.parse_args()
    export(args.output,args.work)
