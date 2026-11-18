"""Replay frozen endpoint scores and remove one edit at a time in the fixed proxy."""
import argparse
import csv
import json
from pathlib import Path

import torch

from train_joint_gradient_ports import ROOT, load_examples, read, sha
from global_timing.learned_ports import PortTimingParameters


def write_csv(path, rows):
    with path.open('x') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def ancestor_nodes(aig, po):
    pending = [aig.po_literals[po] >> 1]
    seen = set()
    while pending:
        n = pending.pop()
        if n in seen:
            continue
        seen.add(n)
        pending.extend(lit >> 1 for lit in aig.fanin_literals.get(n, ()))
    return seen


@torch.no_grad()
def trace(cid, indices, endpoints, data, weights, manifest, audit):
    params = PortTimingParameters()
    entry = next(r for r in manifest['selected'] if r['circuit_id'] == cid)
    folder = data / f'case_{cid:04d}'
    frozen = read(folder/'frozen.json')
    for path in [Path(frozen['source'])/'input.aig', *(folder/name for name in
                 ['frozen.json','baseline.json','scores.json','measurements.json','catalogue.json','historical.json'])]:
        audit[str(path)] = sha(path)
    examples, _ = load_examples(data, {'selected':[entry]}, params, 'fixed')
    dom = next(d for d in examples[0]['domains'] if d['name']=='sampled')
    model = dom['model']
    measured = {r['index']:r for r in read(folder/'measurements.json')['results']}
    ancestors = {p:ancestor_nodes(model.partition.aig,p) for p in endpoints}
    checks = list(csv.DictReader((ROOT/'analysis/joint_gradient_v3/ranking_improvement/diagnosis/combinations.csv').open()))
    records, edits = [], []
    for idx in indices:
        record = measured[idx]
        choices = record['choices']
        x = model.one_hot(choices)
        model.variant = 'fixed'
        fixed, _ = model(x)
        actual = record['metrics']['po_arrivals_ps']
        active = [i for i,c in enumerate(choices) if c]
        for i in active:
            reduced = choices.copy()
            reduced[i] = 0
            minus, _ = model(model.one_hot(reduced))
            root = model.catalogue.decisions[i]['root']
            for p in endpoints:
                edits.append(dict(circuit=cid,combo=idx,po=p,decision=i,choice=choices[i],root=root,
                                  in_original_fanin=root in ancestors[p],fixed_minus_drop_ps=float(fixed[p]-minus[p])))
        for seed in (20261117,20261118,20261119):
            wp=weights/f'global_{seed}'/'weights.pt'
            audit[str(wp)]=sha(wp)
            params.load_state_dict(torch.load(wp,map_location='cpu',weights_only=False)['state_dict'])
            model.variant='global'
            pred, _ = model(x)
            if idx:
                row = next(r for r in checks if int(r['seed'])==seed and int(r['circuit_id'])==cid and int(r['index'])==idx)
                for field, value in [('predicted_delay_ps',float(pred.max())),('fixed_critical_ps',float(fixed.max())),('actual_delay_ps',max(actual))]:
                    assert abs(float(row[field])-value)<.001,(cid,idx,seed,field,value)
            for p in endpoints:
                records.append(dict(circuit=cid,combo=idx,po=p,seed=seed,baseline_ps=float(model.baseline_po[p]),
                                    fixed_ps=float(fixed[p]),corrected_ps=float(pred[p]),measured_ps=actual[p],
                                    fixed_peak_po=int(fixed.argmax()),measured_peak_po=actual.index(max(actual)),
                                    corrected_peak_po=int(pred.argmax()),ps_per_structural_unit=model.unit_ps))
    return records, edits


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','weights','output'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(1)
    manifest_path=ROOT/'analysis/joint_gradient_v3/learning_manifest.json'
    manifest=read(manifest_path)
    audit={str(path):sha(path) for path in (manifest_path,ROOT/'projects/culs/src/algorithms/rewrite_library.inc')}
    records,edits=[],[]
    for cid,indices,outputs in [(75,[0,6,8],[90,95]),(179,[0,1,7,8],[3,1035,1784])]:
        rr,ee=trace(cid,indices,outputs,args.data,args.weights,manifest,audit)
        records.extend(rr)
        edits.extend(ee)
    write_csv(args.output/'endpoints.csv',records)
    write_csv(args.output/'drop_one.csv',edits)
    (args.output/'input_hashes.json').write_text(json.dumps(audit,indent=2)+'\n')


if __name__=='__main__':
    main()
