"""Audit actual completed measurements before deriving delay/runtime curves."""
from collections import Counter
import json
import hashlib
import math
from pathlib import Path

from .candidates import Catalogue
from .common import digest, read_jsonl
from .evaluation import cec_passed, parse_timing


def read(path):
    return json.loads(Path(path).read_text())


def audit_measurement(observation, request, catalogue, hashes, execution_contract=None):
    result_path = request / 'evaluation/result.json'
    measured = read(result_path)
    metrics = observation['metrics']
    candidate = Path(observation['aig'])
    if (observation['status'] != 'PASS' or measured['status'] != 'PASS'
            or measured['cec'] != 'PASS' or measured['timing_model'] != 'ABC_SCL_LIBERTY'):
        raise ValueError('Only complete CEC/Liberty successes are measurement points.')
    if digest(candidate) != measured['input_sha256']:
        raise ValueError('Evaluated AIG bytes differ from the reported candidate.')
    cec_log, mapping_log = request / 'evaluation/cec.log', request / 'evaluation/mapping.log'
    if not cec_passed(cec_log.read_text()):
        raise ValueError('CEC raw log does not explicitly prove equivalence.')
    actual = parse_timing(mapping_log.read_text())
    for name in ('delay_ps', 'area', 'po_arrivals_ps'):
        if measured[name] != actual[name] or measured[name] != metrics[name]:
            raise ValueError('Raw Liberty log, measured file and observation disagree: ' + name)
    netlist = request / 'evaluation/mapped.v'
    if digest(netlist) != measured['mapped_sha256']:
        raise ValueError('Mapped netlist bytes changed.')
    files = [result_path, candidate, cec_log, mapping_log, netlist]
    for name in ('abc', 'library', 'original'):
        identity = measured['provenance'][name]
        path = Path(identity['path'])
        if str(path) not in hashes:
            hashes[str(path)] = digest(path)
        if hashes[str(path)] != identity['sha256']:
            raise ValueError('Measurement asset identity changed: ' + name)
    expected = ('read_lib ' + measured['provenance']['library']['path'] + '; read_aiger ' + str(candidate)
                + '; map; topo; stime -p; write_verilog ' + str(netlist))
    command = request / 'evaluation/mapping.command.json'
    if read(command)['argv'] != [measured['provenance']['abc']['path'], '-c', expected]:
        raise ValueError('Mapping command differs from the common fixed protocol.')
    cec_command = request / 'evaluation/cec.command.json'
    if read(cec_command)['argv'] != [measured['provenance']['abc']['path'], '-c',
        'cec -n ' + measured['provenance']['original']['path'] + ' ' + str(candidate)]:
        raise ValueError('CEC command differs from the common positional protocol.')
    files += [command, cec_command]
    if execution_contract is not None:
        if execution_contract['abc']['sha256'] != measured['provenance']['abc']['sha256']:
            raise ValueError('Measured ABC differs from the child frozen tool report.')
        if catalogue is None:
            method = observation['method']
            if method == 'original':
                if str(candidate) != measured['provenance']['original']['path']:
                    raise ValueError('Original control did not map the original input.')
            else:
                native_command = request / 'rewrite.command.json'
                original = measured['provenance']['original']['path']
                if method == 'native_abc':
                    expected_native = [execution_contract['abc']['path'], '-c',
                        f'read_aiger {original}; strash; rewrite; write_aiger {candidate}']
                elif method == 'native_culs':
                    expected_native = [execution_contract['gpuls']['path'], '-c',
                        f'read {original}; rewrite; write {candidate}']
                else:
                    raise ValueError('Unknown standalone native control.')
                if read(native_command)['argv'] != expected_native:
                    raise ValueError('Native rewrite command differs from the fixed baseline.')
                files.append(native_command)
    accepted, regions = None, None
    if catalogue is not None:
        requested = set(catalogue.requested(observation['choices']))
        trace_path = request / 'rewrite/raw/traces/choice.jsonl'
        trace = read_jsonl(trace_path)
        selected = [r for r in trace if r['preselected']]
        if (len(selected) != len(requested) or
                {(r['root'], r['cut_slot'], r['selected_subgraph']) for r in selected} != requested or
                any(r['final_status'] != 'ACCEPTED' for r in selected) or
                sum(r['final_status'] == 'ACCEPTED' for r in trace) != len(requested)):
            raise ValueError('A measured request was not installed exactly and completely.')
        accepted = len(requested)
        regions = len({d['decision_owner_region'] for d, c in zip(catalogue.decisions, observation['choices']) if c})
        files.append(trace_path)
    for path in files:
        hashes[str(path)] = digest(path)
    return {'accepted_rewrites': accepted, 'selected_original_mffcs': regions,
            'original_sha256': measured['provenance']['original']['sha256'],
            'abc_sha256': measured['provenance']['abc']['sha256'],
            'library_sha256': measured['provenance']['library']['sha256']}


def audit_profiles(profiles):
    rows, hashes = [], {}
    for profile in map(Path, profiles):
        parent_path = profile / 'output/profile_summary.json'
        parent, run = read(parent_path), read(profile / 'run.json')
        if run['state'] not in ('COMPLETED', 'COMPLETED_WITH_EXPLICIT_FAILURES'):
            raise ValueError('Cannot analyze a nonterminal profile run.')
        if parent['status'] not in ('PASS', 'PARTIAL'):
            raise ValueError('Unrecognized profile terminal status.')
        hashes[str(parent_path)] = digest(parent_path)
        for record in parent['records']:
            command = record['command']
            seed = record.get('seed', int(command[command.index('--seed') + 1]))
            row = {k: record.get(k) for k in ('index', 'variant', 'full_process_wall_s', 'input_staging_s',
                                             'end_to_end_with_input_staging_s')}
            row.update(seed=seed, run=str(profile), source_commit=run['source_commit'],
                       slurm_job_id=run.get('slurm_job_id'), input=record['input_name'], status=record['status'])
            if record['status'] != 'PASS' or record['returncode'] != 0:
                row.update(status='FAIL', points=[], error='Declared failed/incomplete cold child; no final QoR imputed.')
                rows.append(row)
                continue
            detail_path = Path(record['detail'])
            if digest(detail_path) != record['detail_sha256']:
                raise ValueError('Profile detail hash mismatch.')
            detail = read(detail_path)
            if detail['status'] != 'PASS':
                raise ValueError('Parent and child success statuses disagree.')
            case = detail_path.parent / f'case_{record["index"]:04d}'
            tool_run = Path(command[command.index('--tool-run') + 1])
            tool_report_path = tool_run / 'build/framework_tools/build_report.json'
            contract = read(tool_report_path)['binaries']
            hashes[str(tool_report_path)] = digest(tool_report_path)
            for tool in contract.values():
                if tool['path'] not in hashes:
                    hashes[tool['path']] = digest(Path(tool['path']))
                if hashes[tool['path']] != tool['sha256']:
                    raise ValueError('Frozen child tool binary changed.')
            source_root = Path(detail['source_root'])
            source_metadata_path = source_root.parent / 'run.json'
            source_metadata = read(source_metadata_path)
            hashes[str(source_metadata_path)] = digest(source_metadata_path)
            native = detail.get('record_kind') == 'native_control'
            summary_path = Path(detail.get('search_summary', case / 'gradient_area_constraint/summary.json'))
            search = read(summary_path)
            if search['best'] != detail['selected']:
                raise ValueError('Profile selected observation differs from search result.')
            hashes.update({str(detail_path): digest(detail_path), str(summary_path): digest(summary_path)})
            catalogue, domain_hash = None, None
            if not native:
                cat_path = case / 'cut_validation/verified_catalogue.json'
                cat = read(cat_path)
                catalogue = Catalogue(cat['decisions'], [tuple(c) for c in cat['conflicts']],
                                      {int(k): v for k, v in cat['region_decisions'].items()})
                hashes[str(cat_path)] = digest(cat_path)
                # Context proof file paths may differ; physical identities may not.
                domain = [[d[k] for k in ('root', 'cut_slot', 'truth', 'cut_leaves', 'removable_nodes',
                           'decision_owner_region', 'candidate_subgraph_indices')] for d in cat['decisions']]
                domain_hash = hashlib.sha256(json.dumps([domain, cat['conflicts']], sort_keys=True).encode()).hexdigest()
            observations = search['observations']
            if len(observations) != detail['requests'] or len({o['index'] for o in observations}) != len(observations):
                raise ValueError('Observation request count or indices disagree.')
            timeline_path = Path(detail['timeline'])
            if digest(timeline_path) != detail['timeline_sha256']:
                raise ValueError('Timeline hash mismatch.')
            events = read(timeline_path)['events']
            start, end = record['subprocess_started_monotonic'], record['subprocess_started_monotonic'] + record['full_process_wall_s']
            if not all(start <= e['started_monotonic'] <= e['completed_monotonic'] <= end for e in events):
                raise ValueError('Timeline event is outside the actual cold child.')
            requests = [e for e in events if e.get('kind') == 'request_completed']
            if len(requests) != len(observations) or len({e['index'] for e in requests}) != len(requests):
                raise ValueError('Actual request completion events are missing or duplicated.')
            event_by_index = {e['index']: e for e in requests}
            points, identities = [], []
            best_install = {}
            for observation in observations:
                event = event_by_index[observation['index']]
                if event['status'] != observation['status']:
                    raise ValueError('Completion event has a different request status.')
                if observation['status'] != 'PASS':
                    if event.get('metrics'):
                        raise ValueError('Failed request acquired a timing label.')
                    continue
                request = case if native else Path(observation['aig']).parents[3]
                installed = audit_measurement(observation, request, catalogue, hashes, contract)
                if installed['original_sha256'] != detail['input_sha256']:
                    raise ValueError('Measured original input differs from the profiled input.')
                identities.append((installed['abc_sha256'], installed['library_sha256']))
                if observation['index'] == search['best']['index']:
                    best_install = installed
                for key in ('delay_ps', 'area', 'cec', 'timing_model'):
                    if event['metrics'][key] != observation['metrics'][key]:
                        raise ValueError('Completion event timing differs from the audited measurement.')
                points.append({'index': observation['index'], 'completed_s': event['completed_monotonic'] - start + record.get('input_staging_s', 0),
                    'delay_ps': observation['metrics']['delay_ps'], 'area': observation['metrics']['area'],
                    'source': (observation.get('prediction_before_evaluation') or {}).get('source', 'replay_noop')})
            if not points or len(set(identities)) != 1:
                raise ValueError('Missing validated points or mixed ABC/library identities.')
            baseline = detail.get('no_op', {}).get('metrics')
            if baseline:
                if baseline != observations[0]['metrics'] or any(observations[0]['choices']):
                    raise ValueError('Timing/area reference is not the actual audited no-op.')
                feasible = [o for o in observations if o['status'] == 'PASS' and o['metrics']['area'] <= baseline['area'] * 1.05 + 1e-8]
                if search['best']['metrics']['delay_ps'] != min(o['metrics']['delay_ps'] for o in feasible):
                    raise ValueError('Selected point does not minimize measured area-feasible delay.')
            counts = Counter(p.name.removesuffix('.command.json') for p in case.rglob('*.command.json'))
            direct = [o for o in observations if (o.get('prediction_before_evaluation') or {}).get('source') == 'joint_selection_gradient']
            row.update(delay_ps=search['best']['metrics']['delay_ps'], area=search['best']['metrics']['area'],
                timing_model='ABC_SCL_LIBERTY', detail_path=str(detail_path),
                gpuls_sha256=contract['gpuls']['sha256'],
                gpuls_used=(not native or record['variant'] == 'native_culs'),
                imported_source_root=str(source_root), imported_source_commit=source_metadata['source_commit'],
                algorithm_sha256=detail['algorithm_sha256'],
                record_kind='native_control' if native else 'candidate_search',
                preparation_runtime_s=detail.get('preparation_runtime_s'),
                method_runtime_s=detail.get('method_runtime_s'), stages=detail.get('stages', {}),
                torch_num_threads=detail.get('torch_num_threads'), cuda_cache_regime=detail.get('cuda_cache_regime'),
                torch_version=detail.get('torch_version'),
                python_cuda_device=detail.get('python_cuda_device'),
                joint_proposal_config=detail.get('joint_proposal_config'),
                ablation=detail.get('ablation'),
                regions=detail.get('regions'), decisions=detail.get('decisions'), candidates=detail.get('candidates'),
                input_sha256=detail['input_sha256'], domain_sha256=domain_hash,
                baseline=baseline, points=points, requests=len(observations),
                failed_requests=sum(o['status'] != 'PASS' for o in observations),
                direct_requests=len(direct), direct_successes=sum(o['status'] == 'PASS' for o in direct),
                final_source=(search['best'].get('prediction_before_evaluation') or {}).get('source', 'replay_noop'),
                native_replay_commands=counts['rewrite'], cec_commands=counts['cec'], mapping_commands=counts['mapping'],
                litecut_cache_hit=detail.get('litecut_cache_hit'), torch_loaded=detail.get('torch_loaded'),
                proposal_devices=detail.get('proposal_devices_used'), proposal_engines=detail.get('proposal_engines_used'),
                installation_gain_policy=detail.get('installation_gain_policy', 'positive_gain'), **best_install)
            if catalogue is not None:
                signature = [(o['choices'], o['status'],
                    {k: o['metrics'][k] for k in ('delay_ps', 'area', 'po_arrivals_ps')} if o['status'] == 'PASS' else None)
                    for o in observations]
                row.update(best_choices_sha256=hashlib.sha256(json.dumps(search['best']['choices']).encode()).hexdigest(),
                    request_sequence_sha256=hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest(),
                    checkpoint_sha256=read(case / 'config.json').get('checkpoint_sha256'))
                if row['checkpoint_sha256'] is not None:
                    checkpoint_path = profile / 'input/student_checkpoint'
                    if str(checkpoint_path) not in hashes:
                        hashes[str(checkpoint_path)] = digest(checkpoint_path)
                    if hashes[str(checkpoint_path)] != row['checkpoint_sha256']:
                        raise ValueError('Used Student checkpoint changed or differs from the recorded model.')
            hashes[str(timeline_path)] = digest(timeline_path)
            rows.append(row)
    return rows, hashes


def add_runtime_metrics(rows, cuts=(30, 120, 600)):
    groups = {}
    for row in rows:
        groups.setdefault((row['index'], row['seed']), []).append(row)
    for group in groups.values():
        successes = [r for r in group if r['status'] == 'PASS']
        references = [r['baseline'] for r in successes if r.get('baseline')]
        if not references:
            for row in group:
                row['reference_status'] = 'MISSING_NOOP; no quality threshold inferred'
            continue
        reference = references[0]
        if any((r['delay_ps'], r['area']) != (reference['delay_ps'], reference['area']) for r in references):
            raise ValueError('Fresh no-op measurements disagree within an input/seed.')
        if len({(r['input_sha256'], r['abc_sha256'], r['library_sha256']) for r in successes}) != 1:
            raise ValueError('Compared methods use different input, ABC or library.')
        if len({r['domain_sha256'] for r in successes if r['domain_sha256'] is not None}) > 1:
            raise ValueError('Candidate methods use different physical candidate domains.')
        native = [r['delay_ps'] for r in successes if r['variant'] == 'native_abc'
                  and r['area'] <= 1.05 * reference['area'] + 1e-8]
        target = 1.01 * min([reference['delay_ps']] + native)
        for row in group:
            points = [p for p in row['points'] if p['area'] <= reference['area'] * 1.05 + 1e-8]
            hits = [p['completed_s'] for p in points if p['delay_ps'] <= target]
            row.update(noop_delay_ps=reference['delay_ps'], noop_area=reference['area'],
                       quality_target_ps=target, time_to_quality_s=min(hits) if hits else None,
                       reference_status='MEASURED_SAME_INPUT_SEED',
                       area_feasible_1p05=(row.get('area', math.inf) <= 1.05 * reference['area'] + 1e-8))
            for cut in cuts:
                available = [p['delay_ps'] for p in points if p['completed_s'] <= cut]
                row[f'best_at_{cut}s_ps'] = min(available) if available else None
    return rows
