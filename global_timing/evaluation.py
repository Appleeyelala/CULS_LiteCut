"""Strict CEC and a single Liberty timing/area measurement contract."""
from __future__ import annotations

import math
import os
from pathlib import Path
import re
import time

from .common import ToolError, digest, run_tool, save_json, tool_path


def cec_passed(text: str) -> bool:
    if re.search(r'not\s+equivalent|verification\s+failed|\bUNDECIDED\b|\bUNSATISFIABLE\s+.*failed', text, re.I):
        return False
    return bool(re.search(r'^\s*Networks are equivalent\b', text, re.I | re.M))


def parse_timing(text: str) -> dict:
    metadata = [line.split() for line in text.splitlines() if line.startswith('GT_META ')]
    if len(metadata) != 1 or len(metadata[0]) != 5:
        raise ToolError('Expected exactly one machine-readable Liberty timing report.')
    _, area, delay, count, gates = metadata[0]
    count, gates = int(count), int(gates)
    area, delay = float(area), float(delay)
    rows = [line.split() for line in text.splitlines() if line.startswith('GT_PO ')]
    values = {}
    for row in rows:
        if len(row) != 4:
            raise ToolError('Malformed PO timing row.')
        index, rise, fall = int(row[1]), float(row[2]), float(row[3])
        if index in values:
            raise ToolError('Duplicate PO timing row.')
        values[index] = {'rise_ps': rise, 'fall_ps': fall, 'arrival_ps': max(rise, fall)}
    if count <= 0 or set(values) != set(range(count)):
        raise ToolError('Missing or invalid PO timing rows.')
    numbers = [area, delay] + [v[k] for v in values.values() for k in ('rise_ps', 'fall_ps')]
    if any(not math.isfinite(x) or x < 0 for x in numbers) or gates < 0:
        raise ToolError('Non-finite or negative actual timing/area measurement.')
    arrivals = [values[i]['arrival_ps'] for i in range(count)]
    if not math.isclose(max(arrivals), delay, abs_tol=2e-5, rel_tol=2e-6):
        raise ToolError('PO maximum does not match stime global delay.')
    path = re.search(r'Start-point = pi(\d+)\.\s+End-point = po(\d+)\.', text)
    return {'timing_model': 'ABC_SCL_LIBERTY', 'area': area, 'delay_ps': delay,
            'adp': area * delay, 'mapped_gates': gates, 'po_arrivals_ps': arrivals,
            'po_rise_fall': [values[i] for i in range(count)],
            'critical_po_indices': [i for i, x in enumerate(arrivals) if math.isclose(x, delay, abs_tol=2e-5, rel_tol=0)],
            'abc_native_trace_endpoints': list(map(int, path.groups())) if path else None,
            'abc_native_trace_lines': [line for line in text.splitlines() if re.match(r'Path\s*\d+', line)],
            'trace_semantics': 'Native ABC min-slack traceback; mapped IDs are local to this mapping.'}


class Evaluator:
    def __init__(self, abc: Path, library: Path, original: Path, timeout=300):
        self.abc, self.library, self.original = map(Path, (abc, library, original))
        self.timeout = timeout
        self.provenance = {name: {'path': str(path), 'sha256': digest(path)}
                           for name, path in [('abc', self.abc), ('library', self.library), ('original', self.original)]}

    def evaluate(self, candidate: Path, work: Path) -> dict:
        started = time.monotonic()
        work.mkdir(parents=True, exist_ok=False)
        result = {'status': 'RUNNING', 'input_sha256': digest(candidate), 'provenance': self.provenance}
        try:
            def interface(path):
                with path.open('rb') as stream:
                    fields = stream.readline().split()
                if len(fields) != 6 or fields[0] != b'aig' or int(fields[3]) != 0:
                    raise ToolError('Expected a combinational binary AIG interface.')
                return int(fields[2]), int(fields[4])
            if interface(candidate) != interface(self.original):
                raise ToolError('Replay changed PI/PO interface counts.')
            result['cec_interface'] = 'AIGER_PI_PO_POSITIONAL_ORDER; symbol names may be absent after replay'
            text = run_tool([self.abc, '-c', f'cec -n {tool_path(self.original)} {tool_path(candidate)}'], work, 'cec', self.timeout)
            if not cec_passed(text):
                raise ToolError('CEC did not explicitly prove equivalence.')
            result['cec'] = 'PASS'
            env = dict(os.environ, CULS_GLOBAL_TIMING_REPORT='1')
            command = (f'read_lib {tool_path(self.library)}; read_aiger {tool_path(candidate)}; '
                       f'map; topo; stime -p; write_verilog {tool_path(work / "mapped.v")}')
            text = run_tool([self.abc, '-c', command], work, 'mapping', self.timeout, env)
            result.update(parse_timing(text))
            if not (work / 'mapped.v').is_file():
                raise ToolError('Mapping did not produce a netlist.')
            result.update(status='PASS', mapped_sha256=digest(work / 'mapped.v'))
        except Exception as exc:
            result.update(status='FAIL', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            result['runtime_s'] = time.monotonic() - started
            save_json(work / 'result.json', result)
        return result
