"""Execute independent published experiment commands in a single bounded GPU allocation."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]

if __name__=='__main__':
    if 'SLURM_JOB_ID' not in os.environ:
        raise SystemExit('Slurm required')
    out = Path(sys.argv[1])
    out.mkdir()
    tasks = json.loads(Path(sys.argv[2]).read_text())
    failures = []
    for task in tasks:
        target = out/task['name']
        env = dict(os.environ,**task.get('environment',{}))
        cmd=[sys.executable,'-B',str(ROOT/'scripts'/task['script']),str(target),*task['args']]
        begin = time.perf_counter()
        with (out/(task['name']+'.log')).open('x') as log:
            result = subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        (out/(task['name']+'.json')).write_text(json.dumps(dict(argv=cmd,exit_code=result.returncode,
                  seconds=time.perf_counter()-begin),indent=2))
        print(task['name'],result.returncode,flush=True)
        if result.returncode:
            failures.append(task['name'])
    if failures:
        raise SystemExit('Failed '+','.join(failures))
