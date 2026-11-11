"""Read terminal case failures without inventing measurements or root causes."""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

from .evaluation import failed_comparisons


def describe_failure(record, suite):
    """Attach exact saved evidence to FAIL/PARTIAL/SKIP; never change status.

    Completed tool times are a partial accounting, not a wall-time profile.
    Missing command records do not identify a running process or its duration.
    """
    if record['status'] not in {'FAIL', 'PARTIAL', 'SKIP'}:
        raise ValueError('Failure evidence requires a terminal unsuccessful case.')
    suite = Path(suite).resolve()
    case = (Path(record['summary']).resolve().parent if record.get('summary') else
            suite / f'case_{record["index"]:04d}')
    if not case.resolve().is_relative_to(suite):
        raise ValueError('Failure case is outside its declared suite.')
    result = {'case_status': record['status'], 'reason': record.get('reason'),
              'case_timed_out': record.get('timed_out', False), 'returncode': record.get('returncode'),
              'categories': [], 'artifacts': [], 'failed_methods': [], 'failed_tool_commands': [],
              'failed_evaluations': [], 'report_error': record.get('report_error')}

    def read(path, parent):
        path = Path(path).resolve()
        if not path.is_relative_to(parent):
            raise ValueError('Failure artifact is outside its declared directory.')
        raw = path.read_bytes()
        result['artifacts'].append({'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(),
                                    'size_bytes': len(raw)})
        return raw

    if record.get('log'):
        raw = read(record['log'], suite.parent.parent / 'logs')
        text = raw.decode(errors='replace')
        tail = '\n'.join(text.splitlines()[-40:])[-6000:]
        result.update(case_log_tail=tail, case_log_tail_truncated=len(tail) < len(text.rstrip('\n')))
    config_path = case / 'config.json'
    if config_path.is_file():
        config = json.loads(read(config_path, case))
        if config['input_sha256'] != record['sha256']:
            raise ValueError('Failure input identity differs from its inventory record.')
    if record['status'] == 'SKIP':
        result['categories'].append('DECLARED_SKIP')
    elif record['status'] == 'FAIL':
        result['categories'].append('CASE_TIMEOUT' if result['case_timed_out'] else 'CASE_PROCESS_FAILURE')
    else:
        summary = json.loads(read(record['summary'], case))
        if summary['original']['input_sha256'] != record['sha256']:
            raise ValueError('Partial summary input differs from its inventory record.')
        failed = failed_comparisons(summary)
        if 'failed_methods' in record and record['failed_methods'] != failed:
            raise ValueError('Declared failed methods disagree with the saved summary.')
        result['failed_methods'] = failed
        if failed:
            result['categories'].append('METHOD_COMPARISON_INCOMPLETE')
        if 'native_abc' in failed:
            result['native_abc_attempt'] = summary['native_abc_attempt']
        if record.get('report_status') == 'FAIL':
            result['categories'].append('REPORT_FAILURE')
        if not result['categories']:
            result['categories'].append('PARTIAL_REASON_NOT_SPECIFIED')

    counts, elapsed = Counter(), defaultdict(float)
    for path in sorted(case.rglob('*.command.json')):
        item = json.loads(read(path, case))
        seconds = item['elapsed_s']
        if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError('Invalid saved tool elapsed time.')
        name = path.name.removesuffix('.command.json')
        counts[name] += 1
        elapsed[name] += seconds
        if item.get('timed_out') or item['returncode']:
            result['failed_tool_commands'].append({'path': str(path), **item})
    for path in sorted(case.rglob('result.json')):
        item = json.loads(read(path, case))
        if item.get('status') == 'FAIL':
            result['failed_evaluations'].append({'path': str(path), **{
                key: item[key] for key in ['status', 'error', 'cec', 'runtime_s'] if key in item}})
    result['completed_tool_counts'] = dict(counts)
    result['completed_tool_elapsed_s'] = dict(elapsed)
    result['scope'] = ('Saved terminal evidence only. Categories describe recorded failure boundaries, not a proved '
                       'root cause. Tool failures can belong to recoverable search requests; they do not by themselves '
                       'explain the whole-case failure. Elapsed sums exclude interrupted commands and CPU work. '
                       'No new CEC, mapping, predictions or substituted metrics.')
    return result
