"""Independently verify and dissect a selected comparison method on a development circuit."""
from collections import defaultdict
from pathlib import Path
import sys

from paper_joint_evidence import FrozenCase, read, save_json, digest


def main(out, prior, method):
    out.mkdir()
    case=FrozenCase(prior.name,out/prior.name)
    selection=read(prior/'frozen.json')['methods'][method]
    measured=next(r for r in read(prior/'selected.json') if r['label']==method)
    assert measured['status']=='PASS' and measured['edits']==selection['selection']
    pairs=selection['selection']
    def vector(edits):
        v=case.zero.copy()
        for i,c in edits:
            v[i]=c
        return v
    owners=defaultdict(list)
    for i,c in pairs:
        owners[case.cat.decisions[i]['decision_owner_region']].append((i,c))
    rows=[case.measure('original',None),case.measure('noop',case.zero),case.measure('joint',vector(pairs))]
    current=read(Path(rows[-1]['work'])/'evaluation/result.json')
    archived=read(Path(measured['work'])/'evaluation/result.json')
    parity={k:current[k]==archived[k] for k in ('input_sha256','input_structure_sha256','delay_ps','area','po_rise_fall','cec')}
    assert all(parity.values())
    for i,c in pairs:
        rows.append(case.measure(f'single_{i}',vector([(i,c)])))
    for owner, subset in sorted(owners.items()):
        rows.append(case.measure(f'only_{owner}',vector(subset)))
        taken=set(subset)
        rows.append(case.measure(f'without_{owner}',vector([(i,c) for i,c in pairs if (i,c) not in taken])))
    for i,t in enumerate(selection['trace']):
        v=vector(t['edits'])
        rows.append(case.measure(f'step_{i}',v if any(v) else None))
    save_json(case.out/'ablation.json',dict(method=method,freeze_sha256=digest(prior/'frozen.json'),
              frozen_selection=pairs,owners=owners,parity=parity,measurements=rows))
    print(prior.name,method,len(pairs),len(owners),rows[2]['delay_ps'],flush=True)


if __name__=='__main__':
    main(Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3])
