"""Freeze benchmark attempts and select records without inspecting their quality.

Suites are ordered explicitly. The last listed suite with a terminal record
wins, even when that record is a failure. Pending declared retries remain
visible and prevent final-cohort completion. No measurements are generated.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path


POLICY = 'last_declared_terminal_attempt'
TERMINAL = {'PASS', 'PARTIAL', 'FAIL', 'SKIP'}
IDENTITY = ('index', 'relative_path', 'sha256', 'classification')


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def complete_records(raw):
    end = raw.rfind(b'\n') + 1
    complete = raw[:end]
    records = [json.loads(line) for line in complete.splitlines()]
    return complete, records, len(raw) - end


def inventory_rows(manifest):
    rows = manifest['rows']
    indexed = {row['index']: row for row in rows}
    if len(indexed) != len(rows) or set(indexed) != set(range(len(rows))):
        raise ValueError('Inventory must have unique contiguous input indices.')
    if len({row['sha256'] for row in rows}) != len(rows):
        raise ValueError('Inventory contains duplicate input hashes.')
    return indexed


def declared_indices(metadata, inventory):
    args = metadata['benchmark_arguments']
    count, shard = args['shard_count'], args['shard_index']
    if count < 1 or not 0 <= shard < count:
        raise ValueError('Invalid declared shard.')
    indices = set(map(int, args['indices'].split(','))) if args.get('indices') else set(inventory)
    if not indices <= set(inventory):
        raise ValueError('Declared input index is outside the inventory.')
    return indices & {index for index in inventory if index % count == shard}


def select_attempts(inventory, sources):
    """Pure membership/priority policy; timing and area never affect selection."""
    attempts, chosen, pending = [], {}, []
    declared = {index: [] for index in inventory}
    for number, source in enumerate(sources):
        assigned = declared_indices(source['metadata'], inventory)
        seen = set()
        for index in sorted(assigned):
            declared[index].append(number)
        for record in source['records']:
            index = record['index']
            if index not in assigned or index in seen:
                raise ValueError('Unexpected or duplicate input in one suite.')
            seen.add(index)
            if any(record.get(key) != inventory[index][key] for key in IDENTITY):
                raise ValueError('Record identity differs from the inventory.')
            if record['status'] not in TERMINAL:
                raise ValueError('JSONL record is not a terminal benchmark attempt.')
            attempt = {'attempt': len(attempts), 'source': number, 'record': record}
            attempts.append(attempt)
            chosen[index] = attempt
        pending.extend({'source': number, 'index': index} for index in sorted(assigned - seen))
    cases = []
    for index, row in sorted(inventory.items()):
        selected = chosen.get(index)
        cases.append({'index': index, 'relative_path': row['relative_path'], 'sha256': row['sha256'],
                      'status': selected['record']['status'] if selected else 'NOT_RECORDED',
                      'selected_attempt': selected['attempt'] if selected else None,
                      'declared_sources': declared[index]})
    selected = [chosen[index] for index in sorted(chosen)]
    return {'attempts': attempts, 'selected_attempts': selected, 'cases': cases,
            'pending_declared_attempts': pending, 'expected_inputs': len(inventory),
            'recorded_inputs': len(selected), 'all_inputs_have_terminal_record': len(chosen) == len(inventory),
            'declared_attempts_complete': not pending,
            'coverage_complete': len(chosen) == len(inventory) and not pending,
            'case_status_counts': dict(Counter(row['status'] for row in cases)),
            'attempt_status_counts': dict(Counter(row['record']['status'] for row in attempts))}


def _snapshot(output, relative, data):
    path = output / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(data)
    return {'snapshot': relative, 'sha256': digest_bytes(data)}


def _read_snapshot(output, record):
    path = (output / record['snapshot']).resolve()
    if not path.is_relative_to(output.resolve()):
        raise ValueError('Snapshot escapes the cohort directory.')
    data = path.read_bytes()
    if digest_bytes(data) != record['sha256']:
        raise ValueError('Frozen cohort snapshot checksum changed: ' + str(path))
    return data


def _artifact(path):
    return {'path': str(path), 'sha256': digest_bytes(path.read_bytes())}


def _check_case(attempt, source):
    record = attempt['record']
    if record['status'] not in {'PASS', 'PARTIAL'}:
        if not record.get('log'):
            return {}
        log = Path(record['log']).resolve()
        if not log.is_relative_to(Path(source['suite']).parent.parent / 'logs'):
            raise ValueError('Failure log is outside its declared run.')
        return {'log': _artifact(log)}
    suite = Path(source['suite'])
    summary_path = Path(record['summary']).resolve()
    if not summary_path.is_relative_to(suite):
        raise ValueError('Successful summary is outside its original suite.')
    summary = json.loads(summary_path.read_text())
    config_path = summary_path.parent / 'config.json'
    config = json.loads(config_path.read_text())
    metadata = source['metadata']
    if config['input_sha256'] != record['sha256'] or summary['original']['input_sha256'] != record['sha256']:
        raise ValueError('Completed case uses a different input.')
    if any(config.get(key) != value for key, value in summary['config'].items()):
        raise ValueError('Summary search parameters differ from the case configuration.')
    binaries = metadata['binaries']
    assets = {item['name']: item['sha256'] for item in metadata['assets']}
    if (config['gpuls_sha256'] != binaries['gpuls']['sha256'] or
            config['toolchain']['abc']['sha256'] != binaries['abc']['sha256'] or
            config['toolchain']['library']['sha256'] != assets['asap7'] or
            config['checkpoint_sha256'] != assets['student_checkpoint']):
        raise ValueError('Case tool/model/library hashes differ from its source run.')
    names = {'evaluations': 'evaluation_budget', 'warmup': 'warmup_evaluations',
             'gradient_steps': 'gradient_steps', 'calibration_steps': 'calibration_steps'}
    if any(metadata['benchmark_arguments'][key] != config[value] for key, value in names.items()):
        raise ValueError('Case search budget differs from the declared suite arguments.')
    return {'summary': _artifact(summary_path), 'config': _artifact(config_path)}


def build_cohort(manifest_path, suites, output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    raw_manifest = Path(manifest_path).read_bytes()
    inventory = inventory_rows(json.loads(raw_manifest))
    manifest_snapshot = _snapshot(output, 'snapshots/input_manifest.json', raw_manifest)
    sources = []
    if len({Path(suite).resolve() for suite in suites}) != len(suites):
        raise ValueError('A source suite was listed more than once.')
    for number, suite in enumerate(suites):
        suite = Path(suite).resolve()
        metadata_path = suite.parent.parent / 'run.json'
        raw_metadata = metadata_path.read_bytes()
        metadata = json.loads(raw_metadata)
        source_manifest = Path(metadata['benchmark_arguments']['manifest'])
        raw_source_manifest = source_manifest.read_bytes()
        if digest_bytes(raw_source_manifest) != metadata['benchmark_manifest_sha256']:
            raise ValueError('Source suite inventory changed after execution.')
        source_inventory = inventory_rows(json.loads(raw_source_manifest))
        if set(source_inventory) != set(inventory) or any(
                any(source_inventory[i][key] != row[key] for key in IDENTITY) for i, row in inventory.items()):
            raise ValueError('Suite inventory identity does not match the declared cohort.')
        complete, records, ignored = complete_records((suite / 'records.jsonl').read_bytes())
        prefix = f'snapshots/suite_{number:02d}/'
        sources.append({'suite': str(suite), 'source_commit': metadata['source_commit'],
                        'metadata': metadata, 'live_metadata_path': str(metadata_path),
                        'metadata_snapshot': _snapshot(output, prefix + 'run.json', raw_metadata),
                        'inventory_snapshot': _snapshot(output, prefix + 'manifest.json', raw_source_manifest),
                        'records_snapshot': _snapshot(output, prefix + 'records.jsonl', complete),
                        'ignored_incomplete_tail_bytes': ignored, 'records': records})
    selection = select_attempts(inventory, sources)
    artifacts = {str(attempt['attempt']): _check_case(attempt, sources[attempt['source']])
                 for attempt in selection['attempts']}
    versions = Counter(sources[attempt['source']]['source_commit'] for attempt in selection['selected_attempts'])
    report = {'schema_version': 1, 'policy': POLICY, 'input_manifest': manifest_snapshot,
              'input_manifest_source': str(Path(manifest_path).resolve()),
              'sources': [{key: value for key, value in source.items() if key not in {'metadata', 'records'}}
                          for source in sources],
              **selection, 'artifacts': artifacts, 'selected_source_versions': dict(versions),
              'heterogeneous_source_versions': len(versions) > 1,
              'scope': 'Declared suite order selects the latest terminal attempt, never the best metric. '
                       'All attempts and pending declarations remain visible. Coverage is not an all-PASS or timing-improvement claim.'}
    with (output / 'cohort.json').open('x') as stream:
        json.dump(report, stream, indent=2)
    return report


def read_cohort(path):
    """Validate frozen sources and referenced results before using a selection."""
    path = Path(path).resolve()
    report = json.loads(path.read_text())
    if report['schema_version'] != 1 or report['policy'] != POLICY:
        raise ValueError('Unknown cohort policy/schema.')
    inventory = inventory_rows(json.loads(_read_snapshot(path.parent, report['input_manifest'])))
    sources = []
    for source in report['sources']:
        metadata = json.loads(_read_snapshot(path.parent, source['metadata_snapshot']))
        source_inventory = inventory_rows(json.loads(_read_snapshot(path.parent, source['inventory_snapshot'])))
        if set(source_inventory) != set(inventory) or any(
                any(source_inventory[i][key] != row[key] for key in IDENTITY) for i, row in inventory.items()):
            raise ValueError('Frozen source inventory does not match the cohort.')
        complete, records, ignored = complete_records(_read_snapshot(path.parent, source['records_snapshot']))
        if ignored or source['source_commit'] != metadata['source_commit']:
            raise ValueError('Incomplete frozen record or source-commit mismatch.')
        sources.append(dict(source, metadata=metadata, records=records))
    expected = select_attempts(inventory, sources)
    if any(report[key] != value for key, value in expected.items()):
        raise ValueError('Cohort selection/counts differ from the declared policy and frozen records.')
    versions = Counter(sources[attempt['source']]['source_commit'] for attempt in expected['selected_attempts'])
    if (report['selected_source_versions'] != dict(versions) or
            report['heterogeneous_source_versions'] != (len(versions) > 1)):
        raise ValueError('Cohort source-version counts differ from the frozen selection.')
    for attempt in expected['attempts']:
        actual = _check_case(attempt, sources[attempt['source']])
        if actual != report['artifacts'][str(attempt['attempt'])]:
            raise ValueError('Saved case configuration or result changed after cohort creation.')
    return report, sources


def analysis_sources(suites=(), cohort=None):
    """Provide original-suite identities for either live snapshots or a cohort."""
    if cohort is not None:
        if suites:
            raise ValueError('Choose a declared cohort or raw suites, not both.')
        report, sources = read_cohort(cohort)
        batches = []
        for number, source in enumerate(sources):
            records = [attempt['record'] for attempt in report['selected_attempts'] if attempt['source'] == number]
            raw = ''.join(json.dumps(row, sort_keys=True) + '\n' for row in records).encode()
            batches.append(dict(source, records=records, raw_complete=raw,
                                metadata_path=str(Path(cohort).resolve().parent / source['metadata_snapshot']['snapshot'])))
        return batches, report
    batches = []
    for suite in suites:
        suite = Path(suite).resolve()
        metadata_path = suite.parent.parent / 'run.json'
        metadata = json.loads(metadata_path.read_text())
        complete, records, ignored = complete_records((suite / 'records.jsonl').read_bytes())
        batches.append({'suite': str(suite), 'metadata': metadata, 'records': records,
                        'raw_complete': complete, 'ignored_incomplete_tail_bytes': ignored,
                        'metadata_path': str(metadata_path), 'source_commit': metadata['source_commit']})
    return batches, None
