"""Compare both parsers on the same complete template Verilog and check ordered pin/gate identities."""
import csv
import hashlib
import importlib.util
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
import time

from paper_joint_evidence import ROOT, read, save_json
from global_timing.liberty_cover import Liberty,mapped_graph
import torch

def fingerprint(graph):
    h=hashlib.sha256()
    for part in (graph[0],graph[1],graph[2].items()):
        for row in part: h.update((repr(row)+'\n').encode())
        h.update(b'\x00')
    return h.hexdigest()

def run(task):
    name,old,out,reference=task
    spec=importlib.util.spec_from_file_location('old_cover',reference)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.set_num_threads(1)
    lib=Liberty(old/'original/tables','cpu')
    records=[]
    for filename in ('initial.v','templates.v'):
        hashes=[]
        for label,fn in [('original',module.mapped_graph),('indexed',mapped_graph)]:
            start=time.perf_counter()
            graph=fn(old/'templates'/filename,lib)
            seconds=time.perf_counter()-start
            hashes.append(fingerprint(graph))
            records.append(dict(filename=filename,version=label,seconds=seconds,sha256=hashes[-1],gates=len(graph[2])))
            del graph
        assert len(set(hashes))==1
    save_json(out/(name+'.json'),records)

if __name__=='__main__':
    out=Path(sys.argv[1]);out.mkdir()
    reference=Path(sys.argv[2])
    with (ROOT/'analysis/paper14_fullflow/results.csv').open() as f:
        tasks=[(r['circuit'],Path(r['work']),out,reference) for r in csv.DictReader(f) if r['circuit'] in ('b18_1_comb','jpeg')]
    with ProcessPoolExecutor(2) as pool:list(pool.map(run,tasks))
