"""Serialize GPU-resident cold cases sharing one Slurm GPU visibility set."""
import fcntl
import hashlib
import os
import time


def acquire_gpu_slot(run):
    job = os.environ['SLURM_JOB_ID']
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not visible:
        raise RuntimeError('A recorded CUDA_VISIBLE_DEVICES allocation is required.')
    key = hashlib.sha256(visible.encode()).hexdigest()[:16]
    stream = (run / f'gpu_slot_{job}_{key}.lock').open('a')
    started = time.perf_counter()
    # Acquire in the shard parent BEFORE starting/timing a cold case. Waiting
    # children must not consume the per-case experiment timeout or GPU memory.
    fcntl.flock(stream, fcntl.LOCK_EX)
    return stream, {'slurm_job_id': job, 'cuda_visible_devices': visible,
                    'slurm_job_gpus': os.environ.get('SLURM_JOB_GPUS'),
                    'slot_wait_s': time.perf_counter() - started,
                    'policy': 'one cold case worker per allocation visibility set',
                    'scope': 'Shard scheduling wait is separate from measured cold case runtime.'}
