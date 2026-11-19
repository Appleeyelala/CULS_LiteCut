"""Evaluate a frozen arm and controls on a specified shard of the blinded panel."""
import argparse
from pathlib import Path
import subprocess
import sys
import time
import torch
from paper_joint_evidence import ROOT, digest, read, save_json
from paper_joint_selectors import run
from global_timing.common import run_tool, tool_path
from global_timing.evaluation import Evaluator


def evaluate(record,case,arm):
    case.mkdir()
    source=Path(record['aig'])
    assert digest(source)==record['aig_sha256']
    tools=ROOT/'config/load_rewrite_tools.json'
    config=read(tools)
    abc=Path(config['frozen_abc'])
    ev=Evaluator(abc,Path(config['liberty']),source,900)
    b=case/'baseline.aig'
    before=time.perf_counter()
    run_tool([abc,'-c',f'read_aiger {tool_path(source)}; strash; rewrite; write_aiger {tool_path(b)}'],
             case,'rewrite',900)
    baseline=ev.evaluate(b,case/'baseline')
    save_json(case/'baseline.json',dict(seconds=time.perf_counter()-before,delay_ps=baseline['delay_ps'],
                                         area=baseline['area'],cec=baseline['cec']))
    raw=case/'raw'
    cmd=[sys.executable,'-B',str(ROOT/'scripts/load_rewrite.py'),'--original',str(source),'--tools',str(tools),
         '--output',str(raw),'--mode','delay','--method','gradient','--device','cuda','--steps','8',
         '--alpha','0.5','--seed','20261117','--max-edits','16']
    if arm.startswith('anchored'):
        cmd.append('--anchor-outputs')
    if arm.endswith('adaptive'):
        cmd.append('--adaptive-sizes')
    before=time.perf_counter()
    with (case/'raw.log').open('x') as log:
        process=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
    save_json(case/'attempt.json',dict(seconds=time.perf_counter()-before,command=cmd,exit_code=process.returncode))
    process.check_returncode()
    controls=case/'controls'
    controls.mkdir()
    run(record['family'],controls,'cuda',offline=False,reference_work=raw)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out',type=Path)
    parser.add_argument('manifest',type=Path)
    parser.add_argument('arm',choices=('raw_fixed','raw_adaptive','anchored_fixed','anchored_adaptive'))
    parser.add_argument('shard',type=int)
    parser.add_argument('shards',type=int)
    a=parser.parse_args()
    if not 0<=a.shard<a.shards:
        parser.error('Invalid shard')
    records=read(a.manifest)['rows']
    assert len({r['aig_sha256'] for r in records})==len(records)==24
    a.out.mkdir()
    torch.set_num_threads(1)
    save_json(a.out/'freeze.json',dict(arm=a.arm,manifest_sha256=digest(a.manifest),shards=a.shards,
                                       shard=a.shard,indices=list(range(a.shard,len(records),a.shards))))
    failures=[]
    for i in range(a.shard,len(records),a.shards):
        row=records[i]
        work=a.out/f'case_{i:02d}'
        try:
            evaluate(row,work,a.arm)
        except Exception as exc:
            failures.append(dict(index=i,family=row['family'],error=repr(exc)))
    save_json(a.out/'failures.json',failures)
    if failures:
        raise SystemExit(1)
