"""Replay the exact three stored searches, including zero-edit serialization controls."""
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from global_timing.candidates import Catalogue, replay
from global_timing.common import ToolError, digest, save_json
from global_timing.evaluation import Evaluator

COUNTS = {'c5315': 7, 'b17_1_comb': 4, 'systemcaes_comb': 7}

def read(path):
    return json.loads(path.read_text())

def vector_hash(vector):
    return hashlib.sha256(json.dumps(vector,separators=(',',':')).encode()).hexdigest()

def require(condition, message):
    if not condition:
        raise ValueError(message)

def freeze(output):
    with (ROOT/'analysis/paper14_fullflow/results.csv').open() as stream:
        prior = {r['circuit']: Path(r['work']) for r in csv.DictReader(stream)}
    result = {}
    for name, count in COUNTS.items():
        old = prior[name]
        inputs, search, selected = (read(old/n) for n in ('inputs.json','proposal.json','result.json'))
        prepared = read(old/'prepared/prepared.json')
        source = Path(inputs['original'])
        require(digest(source)==inputs['original_sha256']==prepared['original_sha256'],name+' AIG checksum')
        require(inputs['tools']['checkpoint']['sha256']==prepared['checkpoint_sha256'],name+' model')
        for key,item in inputs['tools'].items():
            require(digest(Path(item['path']))==item['sha256'],name+' '+key)
        for key,sha in prepared['files'].items():
            require(digest(old/'prepared'/key)==sha,name+' '+key)
        require(search['vector']==selected['vector'] and inputs['alpha']==.5,name+' selected identity')
        choices={}
        for step in search['trace']:
            vector=step['vector']
            h=vector_hash(vector)
            if h not in choices:
                choices[h]=dict(id='step'+str(step['step']) if any(vector) else 'noop',steps=[],
                                edits=[[i,c] for i,c in enumerate(vector) if c],length=len(vector),
                                sha256=h,proxy_ps=step['delay_ps'],chosen=vector==selected['vector'])
            choice=choices[h]
            require(abs(choice['proxy_ps']-step['delay_ps'])<1.e-4,name+' repeated score')
            choice['steps'].append(step['step'])
        require(len(choices)==count+1 and sum(c['chosen'] for c in choices.values())==1
                and len(search['trace'])==8 and any(c['id']=='noop' for c in choices.values()),name+' trace')
        result[name]=dict(old=str(old),inputs=inputs,prepared=prepared,proposal_sha256=digest(old/'proposal.json'),
                          choices=list(choices.values()))
    save_json(output/'freeze.json',result)
    return result

def run(output):
    output.mkdir(parents=True,exist_ok=False)
    cases=freeze(output)
    for name,case in cases.items():
        old=Path(case['old']); prepared=old/'prepared'; inputs=case['inputs']
        tools={k:Path(item['path']) for k,item in inputs['tools'].items()}
        source=Path(inputs['original'])
        cat=Catalogue(**read(prepared/'catalogue.json'))
        ev=Evaluator(tools['frozen_abc'],tools['liberty'],source,900)
        for choice in [dict(id='original',edits=[])]+case['choices']:
            key=choice['id']; work=output/name/key
            work.mkdir(parents=True)
            try:
                target=source
                if key!='original':
                    vector=[0]*choice['length']
                    for i,c in choice['edits']:vector[i]=c
                    require(vector_hash(vector)==choice['sha256'],name+' vector checksum')
                    target,_=replay(tools['gpuls'],source,prepared/'owners.csv',cat,vector,work/'replay',
                                    900,allow_nonpositive_gain=True,allow_nonpositive_candidates=True)
                measured=ev.evaluate(target,work/'evaluation')
                if choice.get('chosen'):
                    original_run=read(old/'final/result.json')
                    for field in ('input_sha256','delay_ps','area','po_rise_fall'):
                        require(measured[field]==original_run[field],name+' historical '+field)
                if key=='original':
                    require(measured['delay_ps']==read(old/'result.json')['original_delay_ps'],name+' baseline')
                attempt=dict(status='PASS',aig=str(target))
            except ToolError as exc:
                attempt=dict(status='FAILED',error=str(exc))
            save_json(work/'attempt.json',attempt)
            print(name,key,attempt['status'],flush=True)

if __name__=='__main__':
    if not os.environ.get('SLURM_JOB_ID'): raise SystemExit('Requires a Slurm allocation')
    run(Path(sys.argv[1]).resolve())
