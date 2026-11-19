"""Test each edited MFFC as a block and its complement, using the archived joint vector unchanged."""
from collections import defaultdict
from pathlib import Path
import sys
from paper_joint_evidence import FrozenCase, save_json


def run(out,name):
    case=FrozenCase(name,out/name)
    groups=defaultdict(list)
    for i,c in enumerate(case.selected):
        if c:
            groups[case.cat.decisions[i]['decision_owner_region']].append((i,c))
    rows=[case.measure('original',None),case.measure('noop',case.zero),case.measure('joint',case.selected)]
    for owner,edits in sorted(groups.items()):
        only=case.zero.copy()
        rest=case.selected.copy()
        for i,c in edits:
            only[i]=c
            rest[i]=0
        rows.extend((case.measure('only_'+str(owner),only),case.measure('without_'+str(owner),rest)))
    save_json(case.out/'owners.json',dict(edits_by_owner=groups,measurements=rows))


if __name__=='__main__':
    out=Path(sys.argv[1])
    out.mkdir()
    for name in sys.argv[2:]:
        run(out,name)
