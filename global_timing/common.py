"""Provenance, safe process execution, and explicit legacy module adapters."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def save_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    # NaN/Inf must never become reported timing measurements.
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
    with path.open('x') as stream:
        stream.write(encoded)


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def legacy(name: str):
    if name not in {'gate3a_analyze', 'gate3b_analyze', 'gate3c_analyze', 'gate3e_run'}:
        raise ValueError('Unreviewed legacy module: ' + name)
    module_name = '_culs_' + name
    if module_name not in sys.modules:
        path = ROOT / 'projects/culs/tools' / (name + '.py')
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return sys.modules[module_name]


def tool_path(path: Path) -> str:
    """CULS's command parser has no escaping protocol; require staged safe paths."""
    value = str(path.absolute())
    if not re.fullmatch(r'[/A-Za-z0-9_.:+-]+', value):
        raise ValueError('Stage tool inputs in a path without spaces or metacharacters: ' + value)
    return value


class ToolError(RuntimeError):
    pass


def run_tool(argv, work: Path, name: str, timeout: float = 300, env=None) -> str:
    work.mkdir(parents=True, exist_ok=True)
    log = work / (name + '.log')
    record = {'argv': list(map(str, argv)), 'cwd': str(work), 'timeout_s': timeout}
    start = time.monotonic()
    with log.open('x') as stream:
        proc = subprocess.Popen(record['argv'], cwd=work, stdout=stream, stderr=subprocess.STDOUT,
                                start_new_session=True, env=env)
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            record['timed_out'] = True
        record.update(returncode=proc.returncode, elapsed_s=time.monotonic() - start, log=str(log))
    save_json(work / (name + '.command.json'), record)
    text = log.read_text(errors='replace')
    if record.get('timed_out') or proc.returncode:
        raise ToolError(f'{name} failed: {record}; {text[-1500:]}')
    return text
