"""Fresh-process raw AIG device comparisons. Enumeration and replay use gpuls on both arms."""
import csv
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from global_timing.common import digest, save_json


def run(out, task, references):
    name, device = task.split('/')
    row = references[name]
    old = Path(row['work'])
    inputs = json.loads((old / 'inputs.json').read_text())
    source = Path(inputs['original'])
    assert digest(source) == inputs['original_sha256']
    tool_config = ROOT / 'config/load_rewrite_tools.json'
    for key, path in json.loads(tool_config.read_text()).items():
        if key in inputs['tools']:
            assert digest(Path(path)) == inputs['tools'][key]['sha256']
    dest = out / (name + '_' + device)
    cmd = [sys.executable, '-B', str(ROOT / 'scripts/load_rewrite.py'), '--original', str(source),
           '--output', str(dest), '--tools', str(tool_config), '--method', 'gradient', '--mode', 'delay',
           '--alpha', '0.5', '--steps', '8', '--seed', str(inputs['seed']), '--max-edits', '16',
           '--device', device]
    begin = time.perf_counter()
    with (out / (dest.name + '.log')).open('x') as stream:
        process = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT)
    seconds = time.perf_counter() - begin
    result = dict(command=cmd, exit_code=process.returncode, seconds=seconds, host=socket.gethostname(),
                  job=os.environ['SLURM_JOB_ID'], reference=str(old))
    save_json(out / (dest.name + '.attempt.json'), result)
    print(name, device, process.returncode, round(seconds, 2), flush=True)
    process.check_returncode()


if __name__ == '__main__':
    out = Path(sys.argv[1])
    out.mkdir()
    with (ROOT / 'analysis/paper14_fullflow/results.csv').open() as stream:
        references = {r['circuit']: r for r in csv.DictReader(stream)}
    for i, task in enumerate(sys.argv[2:]):
        work = out / f'trial_{i:02d}'
        work.mkdir()
        run(work, task, references)
