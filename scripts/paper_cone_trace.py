"""Compare the same PO's signed AIG and ABC-mapped cell cone across frozen nested edits."""
from collections import Counter
from graphlib import TopologicalSorter
import hashlib
from pathlib import Path
import sys

from audit_replay_noop_v3 import fingerprints
from paper_joint_evidence import ROOT, digest, read, save_json
from global_timing.liberty_cover import Liberty, mapped_graph
from global_timing.structure import parse_aig


def fanin(start,parents):
    seen=set()
    queue=[start]
    while queue:
        node=queue.pop()
        if node not in seen:
            seen.add(node)
            queue.extend(parents(node))
    return seen


def mapped(path,lib,po):
    inputs,outputs,gates=mapped_graph(path,lib)
    cone=fanin(outputs[po],lambda n:gates[n][1] if n in gates else ())
    ids={p:hashlib.sha256(f'pin/{i}'.encode()).digest() for i,p in enumerate(inputs)}
    depths={p:0 for p in inputs}
    for symbol in ("1'b0","1'b1"):
        ids[symbol]=hashlib.sha256(symbol.encode()).digest()
        depths[symbol]=0
    cells=Counter()
    for n in TopologicalSorter({n:gates[n][1] for n in gates if n in cone}).static_order():
        if n not in gates:
            continue
        cell,parents=gates[n]
        if cell is None:
            ids[n]=ids[parents[0]]
            depths[n]=depths[parents[0]]
        else:
            ids[n]=hashlib.sha256(cell.encode()+b'\0'+b''.join(ids[p] for p in parents)).digest()
            depths[n]=1+max((depths[p] for p in parents),default=0)
            cells[cell]+=1
    return dict(mapped_cone_sha256=ids[outputs[po]].hex(),mapped_cell_count=sum(cells.values()),
                max_mapped_cell_depth=depths[outputs[po]],cell_counts=cells,po_name=outputs[po])


def main(out,file,reference,po):
    out.mkdir()
    records=read(file)
    inputs=read(reference/'inputs.json')
    src=Path(inputs['original'])
    assert digest(src)==inputs['original_sha256']
    original=parse_aig(src)
    original_cone=fanin(original.po_literals[po]>>1,lambda n:(e>>1 for e in original.fanin_literals.get(n,())))
    cat=read(reference/'prepared/catalogue.json')['decisions']
    lib=Liberty(reference/'original/tables','cpu')
    rows=[]
    for row in records:
        label=row['label']
        if label not in ('original','noop') and not label.startswith('nested_'):
            continue
        work=Path(row['work'])
        original_file=src if row['edits'] is None else Path(read(work/'replay/replay.json')['aig'])
        assert digest(original_file)==row['input_sha256']
        aig=parse_aig(original_file)
        descriptors=fingerprints(aig)
        abc=read(work/'evaluation/result.json')
        roots=[cat[i]['root'] for i,c in row['edits'] or []]
        result=mapped(work/'evaluation/mapped.v',lib,po)
        rows.append(dict(label=label,po=po,roots_on_original_cone=[r for r in roots if r in original_cone],
                         edits=len(roots),signed_aig_cone_sha256=descriptors['output_hashes'][po],
                         abc_rise_ps=abc['po_rise_fall'][po]['rise_ps'],abc_fall_ps=abc['po_rise_fall'][po]['fall_ps'],
                         **result))
    save_json(out/'cones.json',rows)


if __name__=='__main__':
    main(Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]),int(sys.argv[4]))
