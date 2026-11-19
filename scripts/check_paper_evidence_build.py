"""Run the workspace unit tests in a CPU Slurm allocation."""
import os
from pathlib import Path
import subprocess
import sys


if __name__=='__main__':
    assert 'SLURM_JOB_ID' in os.environ
    output = Path(sys.argv[1])
    output.mkdir()
    root = Path(__file__).resolve().parents[1]
    pattern=sys.argv[2] if len(sys.argv)>2 else 'test*.py'
    with (output/'tests.log').open('x') as log:
        subprocess.run([sys.executable,'-B','-m','unittest','discover','-s','tests','-p',pattern],
                       cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
