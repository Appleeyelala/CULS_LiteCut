"""Shared, frozen-catalogue ABC measurement for the publication development cases."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from global_timing.candidates import Catalogue, replay
from global_timing.common import ToolError, digest, save_json
from global_timing.evaluation import Evaluator

def read(path):
    return json.loads(path.read_text())

class FrozenCase:
    def __init__(self,name,out):
        with (ROOT/'analysis/paper14_fullflow/results.csv').open() as stream:
            self.reference=next(row for row in csv.DictReader(stream) if row['circuit']==name)
        self.old=Path(self.reference['work'])
        self.out=out
        out.mkdir()
        self.inputs=read(self.old/'inputs.json')
        self.source=Path(self.inputs['original'])
        self.prep=self.old/'prepared'
        m=read(self.prep/'prepared.json')
        assert digest(self.source)==self.inputs['original_sha256']==m['original_sha256']
        self.tools={k:Path(v['path']) for k,v in self.inputs['tools'].items()}
        for k,p in self.tools.items(): assert digest(p)==self.inputs['tools'][k]['sha256']
        for k,h in m['files'].items(): assert digest(self.prep/k)==h
        self.cat=Catalogue(**read(self.prep/'catalogue.json'))
        self.zero=[0]*len(self.cat.decisions)
        self.selected=read(self.old/'proposal.json')['vector']
        self.ev=Evaluator(self.tools['frozen_abc'],self.tools['liberty'],self.source,900)
        self.cache={}

    def measure(self,label,v):
        edits=None if v is None else [(i,c) for i,c in enumerate(v) if c]
        key=json.dumps(edits)
        if key not in self.cache:
            w=self.out/label
            w.mkdir()
            target=self.source
            started=time.monotonic()
            try:
                if v is not None:
                    target,_=replay(self.tools['gpuls'],target,self.prep/'owners.csv',self.cat,v,w/'replay',
                                    900,allow_nonpositive_gain=True,allow_nonpositive_candidates=True)
                m=self.ev.evaluate(target,w/'evaluation')
                record={k:m[k] for k in ('delay_ps','area','cec','critical_po_indices','input_sha256')}
                record.update(status='PASS')
            except ToolError as exc:
                record=dict(status='FAIL',error=str(exc))
            record.update(work=str(w),seconds=time.monotonic()-started)
            self.cache[key]=record
        return dict(label=label,edits=edits,**self.cache[key])

def singles(case):
    v=case.selected
    records=[case.measure('original',None),case.measure('noop',case.zero),case.measure('joint',v)]
    assert records[2]['delay_ps']==read(case.old/'final/result.json')['delay_ps']
    for i,c in enumerate(v):
        if c:
            one=case.zero.copy()
            one[i]=c
            records.append(case.measure('single_'+str(i),one))
    assert len(records)==19
    return records

if __name__=='__main__':
    if not os.environ.get('SLURM_JOB_ID'):raise SystemExit('Slurm required')
    out=Path(sys.argv[1]).resolve()
    out.mkdir()
    for name in sys.argv[2:]:
        case=FrozenCase(name,out/name)
        rows=singles(case)
        save_json(case.out/'results.json',rows)
        print(name,[(r['label'],r.get('delay_ps')) for r in rows],flush=True)
