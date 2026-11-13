#!/usr/bin/env python3
"""Submit an already prepared /data snapshot after live aggregate quota checks."""
import argparse
import json
import os
from pathlib import Path
import pwd
import re
import subprocess


def output(*command):
    return subprocess.check_output(command, text=True)


def requested_resources(job):
    fields = dict(token.split('=', 1) for token in job.split() if '=' in token)
    raw = fields.get('ReqTRES', fields.get('TRES'))
    if not raw:
        raise ValueError('Cannot determine requested resources: ' + job)
    tres = dict(part.split('=', 1) for part in raw.split(','))
    memory = re.fullmatch(r'([0-9.]+)([KMGTP]?)', tres.get('mem', ''))
    if not memory or 'cpu' not in tres:
        raise ValueError('Unrecognized job CPU/memory request: ' + raw)
    gib = float(memory[1]) * {'': 1 / 1024**3, 'K': 1 / 1024**2, 'M': 1 / 1024,
                             'G': 1, 'T': 1024, 'P': 1024**2}[memory[2]]
    gpu = int(tres.get('gres/gpu', sum(int(v) for k, v in tres.items() if k.startswith('gres/gpu:'))))
    return [int(tres['cpu']), gib, gpu]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--command-json', type=Path, required=True, help='JSON argv array; no shell command expansion.')
    parser.add_argument('--env-json', type=Path, help='Explicit task environment overrides, without secrets.')
    parser.add_argument('--partition', required=True)
    parser.add_argument('--cpus', type=int, required=True)
    parser.add_argument('--memory-gib', type=int, required=True)
    parser.add_argument('--gpus', type=int, default=0)
    parser.add_argument('--walltime', required=True)
    parser.add_argument('--job-name', required=True)
    parser.add_argument('--nodelist', help='Optional single inspected host for consistent hardware across a suite/resume.')
    parser.add_argument('--afterany-job', help='Optional existing own job; run only after it reaches any terminal state.')
    args = parser.parse_args()
    run = args.run.resolve()
    if not run.is_relative_to('/data/wushangge') or not re.fullmatch(r'[/A-Za-z0-9_.-]+', str(run)):
        parser.error('Requires a prepared, shell-safe /data run path.')
    if not (1 <= args.cpus <= 64 and 1 <= args.memory_gib <= 256 and 0 <= args.gpus <= 2):
        parser.error('Per-job resources exceed project limits.')
    if not re.fullmatch(r'\d{1,3}:[0-5]\d:[0-5]\d', args.walltime) or args.walltime == '00:00:00':
        parser.error('A finite positive HH:MM:SS walltime is required.')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.job_name):
        parser.error('Use a simple job name.')
    if args.nodelist and not re.fullmatch(r'[A-Za-z0-9_-]+', args.nodelist):
        parser.error('Select one simple node name, not an unchecked host expression.')
    if args.afterany_job is not None and not re.fullmatch(r'[1-9][0-9]*', args.afterany_job):
        parser.error('Dependency must be an existing numeric Slurm job ID.')
    metadata_path = run / 'run.json'
    metadata = json.loads(metadata_path.read_text())
    if metadata['state'] != 'PREPARED_NOT_SUBMITTED' or (run / 'job.slurm').exists():
        parser.error('Run was already submitted or contains a job script; prepare a new run.')
    command = json.loads(args.command_json.read_text())
    if not isinstance(command, list) or not command or any(not isinstance(x, str) for x in command):
        parser.error('Command must be a nonempty JSON string array.')
    extra_env = json.loads(args.env_json.read_text()) if args.env_json else {}
    if any(not re.fullmatch(r'[A-Z][A-Z0-9_]*', k) or not isinstance(v, str) for k, v in extra_env.items()):
        parser.error('Invalid explicit environment override.')
    username = pwd.getpwuid(os.getuid()).pw_name
    cluster = {'sinfo': output('sinfo', '-o', '%P %a %l %D %G'),
               'partition': output('scontrol', 'show', 'partition', args.partition),
               'queue': output('squeue', '-r', '-h', '-u', username, '-o', '%i')}
    if args.nodelist:
        nodes = output('sinfo', '-h', '-N', '-p', args.partition, '-o', '%N')
        if args.nodelist not in nodes.split():
            parser.error('Requested host is not in the inspected partition.')
        cluster['selected_node'] = output('scontrol', 'show', 'node', args.nodelist)
    if args.afterany_job:
        cluster['dependency_job'] = output('scontrol', 'show', 'job', '-o', args.afterany_job)
        owner = re.search(r'\bUserId=([^\s(]+)\(', cluster['dependency_job'])
        if owner is None or owner[1] != username:
            parser.error('Dependency must be a confirmed job owned by the current OS user.')
    jobs, used = [], [0, 0., 0]
    for identifier in cluster['queue'].split():
        job = output('scontrol', 'show', 'job', '-o', identifier)
        jobs.append(job)
        used = [a + b for a, b in zip(used, requested_resources(job))]
    requested = [args.cpus, args.memory_gib, args.gpus]
    total = [a + b for a, b in zip(used, requested)]
    if any(a > b for a, b in zip(total, [64, 256, 2])):
        parser.error('Aggregate resources exceed project limits: ' + str(total))
    for filename, value in [('command.json', command), ('task_environment.json', extra_env)]:
        with (run / 'input' / filename).open('x') as stream:
            json.dump(value, stream, indent=2)
    metadata.update(command=command, resources={'partition': args.partition, 'cpus_per_task': args.cpus,
        'memory_gib': args.memory_gib, 'gpus': args.gpus, 'walltime': args.walltime, 'nodelist': args.nodelist})
    metadata['slurm_dependency'] = {'type': 'afterany', 'job_id': args.afterany_job} if args.afterany_job else None
    metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
    lines = ['#!/bin/bash', '#SBATCH --job-name=' + args.job_name, '#SBATCH --partition=' + args.partition,
             f'#SBATCH --cpus-per-task={args.cpus}', f'#SBATCH --mem={args.memory_gib}G',
             '#SBATCH --time=' + args.walltime, f'#SBATCH --chdir={run}',
             f'#SBATCH --output={run}/logs/slurm-%j.out', f'#SBATCH --error={run}/logs/slurm-%j.err',
             '#SBATCH --export=NONE']
    if args.gpus:
        lines.append(f'#SBATCH --gres=gpu:{args.gpus}')
    if args.nodelist:
        lines.append('#SBATCH --nodelist=' + args.nodelist)
    if args.afterany_job:
        lines.append('#SBATCH --dependency=afterany:' + args.afterany_job)
    lines += ['set -Eeuo pipefail',
              'export PATH=/home/share/process/anaconda3/bin:/usr/local/cuda-11.8/bin:/usr/local/bin:/usr/bin:/bin',
              'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MKL_THREADING_LAYER=GNU',
              'export PYTHONDONTWRITEBYTECODE=1', f'export TMPDIR={run}/tmp',
              f'export XDG_CACHE_HOME={run}/cache', f'export CUDA_CACHE_PATH={run}/cache/cuda',
              f'export TORCH_EXTENSIONS_DIR={run}/cache/torch', f'export MPLCONFIGDIR={run}/cache/matplotlib',
              f'export PYTHONPATH={run}/src', "/home/share/process/anaconda3/bin/python -B - <<'WORKER'",
              'import json,os,subprocess,sys', 'from pathlib import Path', f'run=Path({str(run)!r})',
              "p=run/'run.json'; m=json.loads(p.read_text())",
              "os.environ.update(json.loads((run/'input/task_environment.json').read_text()))",
              "m.update(state='RUNNING',slurm_job_id=os.environ['SLURM_JOB_ID'],experiment_executed=True,environment={'python':sys.version,'slurm':{k:v for k,v in os.environ.items() if k.startswith('SLURM_')}})",
              "p.write_text(json.dumps(m,indent=2)+'\\n')",
              "result=subprocess.run(json.loads((run/'input/command.json').read_text()),cwd=run)",
              'm=json.loads(p.read_text())',
              "m['exit_code']=result.returncode",
              "if result.returncode: m['state']='FAILED'",
              "elif m['state']=='RUNNING': m['state']='COMPLETED'",
              "p.write_text(json.dumps(m,indent=2)+'\\n')", 'raise SystemExit(result.returncode)', 'WORKER']
    job_path = run / 'job.slurm'
    with job_path.open('x') as stream:
        stream.write('\n'.join(lines) + '\n')
    subprocess.run(['bash', '-n', str(job_path)], check=True)
    identifier = output('sbatch', '--parsable', str(job_path)).strip()
    with (run / 'input/submission.json').open('x') as stream:
        json.dump({'job_id': identifier, 'cluster': cluster, 'other_jobs': jobs,
                   'resource_use_before': used, 'request': requested, 'resource_use_after': total,
                   'dependency': metadata['slurm_dependency']}, stream, indent=2)
    print(json.dumps({'job_id': identifier, 'run': str(run), 'total_resources': total}))


if __name__ == '__main__':
    main()
