"""Evaluate original, project ABC baseline B and the frozen paper LiteCut on identical mapping settings."""
import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from global_timing.common import digest, run_tool, save_json, tool_path
from global_timing.evaluation import Evaluator

def compare(task):
    row,inputs,paper,out = task
    name=row['circuit']
    work=out/name
    work.mkdir()
    src=inputs/row['filename']
    prior=paper/name
    ref=json.loads((prior/'result.json').read_text())
    assert digest(src)==row['sha256']==ref['source_sha256']
    assert digest(prior/'rewritten.aig')==ref['rewritten_sha256']
    tools=json.loads((ROOT/'config/load_rewrite_tools.json').read_text())
    abc=Path(tools['frozen_abc'])
    ev=Evaluator(abc,Path(tools['liberty']),src,900)
    original=ev.evaluate(src,work/'original')
    b=work/'abc_rewrite.aig'
    start=time.monotonic()
    run_tool([abc,'-c',f'read_aiger {tool_path(src)}; strash; rewrite; write_aiger {tool_path(b)}'],work,'rewrite',900)
    baseline=ev.evaluate(b,work/'baseline_b')
    baseline_s=time.monotonic()-start
    native=ev.evaluate(prior/'rewritten.aig',work/'paper_litecut')
    save_json(work/'metrics.json',dict(original=original,baseline_b=baseline,baseline_b_seconds=baseline_s,paper_litecut=native))
    print(name,flush=True)

if __name__=='__main__':
    inputs,paper,out=map(lambda v:Path(v).resolve(),sys.argv[1:])
    out.mkdir()
    with (ROOT/'analysis/paper14_fullflow/circuits.csv').open() as f:
        tasks=[(row,inputs,paper,out) for row in csv.DictReader(f)]
    with ProcessPoolExecutor(max_workers=4) as pool:
        list(pool.map(compare,tasks))
