#!/usr/bin/env python3
"""Package saved ecc85e6 evidence only. No EDA, ML imports or experiments."""
import argparse
from collections import Counter
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import zipfile


SOURCE_COMMIT = 'ecc85e6ac82571caf9f1a6004098dbe976536256'
ARCHIVE = 'MFFC_GPU_GRADIENT_905_ANALYSIS_BUNDLE.zip'
REF = '$bundle_ref'
FEATURES = ['R_root', 'SlackProxy_root', 'Root_Global_Fanout', 'Max_L_leaf',
            'Max_Leaf_Fanout', 'Mean_Leaf_Fanout', 'Std_Leaf_Fanout', 'Frac_Leaf_PI',
            'Weighted_Depth', 'Topo_Depth', 'Inv_Penalty', 'Node_Count', 'Avg_Width',
            'Max_Internal_Fanout', 'Inv_Ratio', 'Root_Inv_Count', 'Chain_Density',
            'Fork_Density', 'Avg_Curvature']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def dump(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def ref(file, pointer=''):
    return {REF: {'file': file, 'pointer': pointer}}


READER = '''"""Resolve lossless JSON references from the extracted ZIP; standard library only."""
import json
from pathlib import Path

def resolve_file(bundle_root, relative_path):
    root = Path(bundle_root).resolve()
    cache = {}
    active = set()
    def load(name):
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Reference escapes bundle')
        if name not in cache:
            cache[name] = json.loads(path.read_text())
        return cache[name]
    def resolve(value):
        if isinstance(value, dict) and set(value) == {'$bundle_ref'}:
            target = value['$bundle_ref']
            key = (target['file'], target['pointer'])
            if key in active:
                raise ValueError('Reference cycle')
            active.add(key)
            obj = load(target['file'])
            for token in target['pointer'].split('/')[1:]:
                obj = obj[int(token)] if isinstance(obj, list) else obj[token.replace('~1', '/').replace('~0', '~')]
            answer = resolve(obj)
            active.remove(key)
            return answer
        if isinstance(value, dict):
            return {k: resolve(v) for k, v in value.items()}
        if isinstance(value, list):
            return [resolve(v) for v in value]
        return value
    return resolve(load(relative_path))
'''


class Bundle:
    def __init__(self, stage):
        self.stage = stage
        stage.mkdir(exist_ok=False)
        self.sources = {}
        self.origins = {}

    def read(self, path):
        path = Path(path).resolve(strict=True)
        before = path.stat()
        raw = path.read_bytes()
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError('Source changed during collection: ' + str(path))
        entry = {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(),
                 'size_bytes': len(raw), 'mtime_ns': after.st_mtime_ns}
        if str(path) in self.sources and self.sources[str(path)] != entry:
            raise ValueError('Source changed between reads: ' + str(path))
        self.sources[str(path)] = entry
        return raw

    def json(self, path):
        return json.loads(self.read(path))

    def target(self, name):
        p = self.stage / name
        if not p.resolve().is_relative_to(self.stage.resolve()):
            raise ValueError('Invalid bundle destination: ' + name)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def write(self, name, value, sources=(), rule='Structured extraction; JSON values preserved'):
        p = self.target(name)
        with p.open('x') as f:
            f.write(dump(value) + '\n')
        self.origins[name] = {'sources': list(map(str, sources)), 'rule': rule}

    def text(self, name, value, sources=(), rule='Generated documentation'):
        with self.target(name).open('x') as f:
            f.write(value)
        self.origins[name] = {'sources': list(map(str, sources)), 'rule': rule}

    def copy(self, source, name):
        raw = self.read(source)
        with self.target(name).open('xb') as f:
            f.write(raw)
        self.origins[name] = {'sources': [str(Path(source).resolve())], 'rule': 'Byte-identical copy'}
        return raw

    def jsonl(self, name, sources=(), rule='One identified object per line'):
        self.origins[name] = {'sources': list(map(str, sources)), 'rule': rule}
        return self.target(name).open('x')

    def csv(self, name, columns, sources=(), rule='Identified numeric/string fields; blanks are missing, not zero'):
        stream = self.target(name).open('x', newline='')
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator='\n')
        writer.writeheader()
        self.origins[name] = {'sources': list(map(str, sources)), 'rule': rule}
        return stream, writer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--delivery', type=Path, required=True)
    args = parser.parse_args()
    run, experiment, delivery = [p.resolve() for p in (args.run, args.experiment, args.delivery)]
    if not os.environ.get('SLURM_JOB_ID') or not all(p.is_relative_to('/data/wushangge') for p in [run, experiment, delivery]):
        parser.error('Collect saved files in a new /data Slurm workspace.')
    source = Path(__file__).resolve().parents[1]
    bundle = Bundle(run / 'output/bundle')
    preflight = bundle.json(experiment / 'input/mffc905_preflight.json')
    assert preflight['source_commit'] == SOURCE_COMMIT
    inputs = {r['index']: r for r in preflight['rows']}
    assert set(inputs) == set(range(905))
    original_delivery = experiment / 'output/benchmark905_delivery'
    completions = [(original_delivery, 'analysis_completion.json'), (delivery, 'delivery_completion.json')]
    for directory, name in completions:
        complete = bundle.json(directory / name)
        assert complete['formal'] and complete['method_points'] == 2715
        for relative, expected in complete['artifacts_sha256'].items():
            assert sha(directory / relative) == expected, relative
    summary = bundle.json(delivery / 'BENCHMARK905_SUMMARY.json')
    assert summary['source_commit'] == SOURCE_COMMIT and summary['designs'] == 905
    audited = bundle.json(original_delivery / 'audited_records.json')
    points = {(r['index'], r['method']): r for r in audited}
    assert len(audited) == len(points) == 2715
    assert set(points) == {(i, m) for i in inputs for m in ['A', 'B', 'Ours']}
    for r in audited:
        assert r['source_commit'] == SOURCE_COMMIT and r['input_sha256'] == inputs[r['index']]['sha256']
        assert Path(r['detail_path']).is_relative_to(experiment)

    for name in ['BENCHMARK905_RESULTS.csv', 'BENCHMARK905_SUMMARY.json', 'BENCHMARK905_RESULTS.md']:
        bundle.copy(delivery / name, 'benchmark/' + name)
    for p in sorted((delivery / 'figures').iterdir()):
        bundle.copy(p, 'benchmark/figures/' + p.name)
    for directory, name in completions:
        bundle.copy(directory / name, 'audit/' + name)
    for directory, name in [(original_delivery, 'audited_records.json'),
                            (original_delivery, 'evidence_hashes.json'),
                            (delivery, 'installation_details.json'),
                            (delivery, 'supplementary_evidence_hashes.json')]:
        bundle.copy(directory / name, 'audit/' + name)
    for relative in ['BENCHMARK905_PROTOCOL.md', 'config/mffc905_protocol.json',
                     'benchmark905_manifest.json', 'provenance/mffc_gpu_freeze_audit.json']:
        bundle.copy(experiment / 'src' / relative, 'config/frozen/' + relative)
    for relative in ['docs/mffc_gpu/IMPLEMENTATION.md', 'GRADIENT_OPTIMIZATION_STATUS.md',
                     'docs/mffc_gpu/FAILURE_NOTES.md', 'docs/mffc_gpu/DELIVERY_AUDIT.md']:
        bundle.copy(source / relative, 'config/reference_docs/' + relative)
    for relative in ['run.json', 'input/freeze_identity.json', 'input/formal-command.json',
                     'input/mffc905_preflight.json', 'input/session_65245.json', 'input/submission.json']:
        bundle.copy(experiment / relative, 'config/experiment/' + relative)
    bundle.copy(delivery.parent.parent / 'run.json', 'config/postprocessing_run.json')
    for name in ['global_timing/mffc_experiment.py', 'global_timing/mffc_model.py',
                 'global_timing/mffc_search.py', 'global_timing/mffc_gpu.py',
                 'global_timing/surrogate.py', 'global_timing/litecut.py',
                 'framework/mffc_gpu/region_lse.cu',
                 'projects/litecut/src/data_preprocess/feat_extraction_v3.py']:
        bundle.copy(experiment / 'src' / name, 'config/frozen_source/' + name)
    bundle.copy(Path(__file__), 'config/package_mffc905_analysis.py')
    bundle.text('config/bundle_reader.py', READER)
    tools = preflight['tools']
    for name, item in tools.items():
        if name == 'native_library':
            report = Path(item['path']).parent / 'build_report.json'
            if report.is_file():
                bundle.copy(report, 'config/native_build_report.json')
    accounting = subprocess.check_output(['sacct', '-j', '65245,65247,65249',
        '--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocTRES', '-n', '-P'], text=True)
    bundle.text('audit/experiment_slurm_accounting.txt', accounting)
    bundle.write('config/versions.json', {'experiment_source_commit': SOURCE_COMMIT,
        'analysis_source_commit': summary['delivery_supplement']['analysis_source_commit'],
        'packaging_source_commit': json.loads((run / 'run.json').read_text())['source_commit'],
        'experiment': str(experiment), 'delivery': str(delivery), 'tools': tools,
        'scope': 'Original GPU environment belongs to job 65245; this CPU-only packaging job is not a performance observation.'})

    streams = []
    def table(name, columns, **kwargs):
        stream, writer = bundle.csv(name, columns, **kwargs)
        streams.append(stream)
        return writer
    def lines(name, **kwargs):
        stream = bundle.jsonl(name, **kwargs)
        streams.append(stream)
        return stream
    def emit(stream, row):
        stream.write(dump(row) + '\n')
    identity = ['circuit_id', 'input', 'input_sha256']
    circuits = table('optimization/circuit_index.csv', identity + ['summary_file', 'decisions', 'candidates',
        'regions', 'device', 'request_count', 'gradient_records', 'calibration_records', 'final_source', 'selected_observation_index'])
    domains = table('optimization/decisions.csv', ['circuit_id', 'decision_index', 'decision_id', 'root',
        'cut_slot', 'region_id', 'truth', 'truth_class', 'candidate_domain_id', 'cut_leaves_json',
        'removable_node_count', 'saved_count', 'classification'])
    scores = table('prediction/litecut_scores.csv', ['circuit_id', 'decision_index', 'choice_index',
        'subgraph_index', 'region_id', 'feature_id', 'predicted_local_abc_delay',
        'ground_truth_local_abc_delay', 'original_cut_truth_verified', 'truth_verified',
        'output_phase_complement', 'representation', 'extra_metadata_json'],
        rule='All saved Student scores including no-op; exact 19D vectors deduplicated by feature_hash. Only cut_validation and replacement_dag_info proof/materialization payloads omitted; no local ground truth exists.')
    features = table('prediction/feature_vectors.csv', ['feature_id', 'feature_hash'] + FEATURES)
    pairs = table('prediction/global_request_pairs.csv', identity + ['request_index', 'request_method',
        'status', 'prediction_source', 'has_prediction', 'valid_global_prediction_pair',
        'predicted_delay_ps', 'actual_mapping_delay_ps', 'predicted_area', 'actual_mapping_area',
        'predicted_objective', 'anchor_objective', 'training_observation_indices_json',
        'noop_mapping_delay_ps', 'noop_mapping_area', 'original_mapping_delay_ps',
        'final_selected_request', 'active_choices_json', 'request_inclusive_runtime_s',
        'method_cold_runtime_s', 'observation_file', 'prediction_file', 'actual_metrics_file', 'error'])
    stages = table('runtime/stages.csv', ['circuit_id', 'stage_index', 'stage', 'elapsed_s'])
    runtimes = table('runtime/method_runtime.csv', ['circuit_id', 'method', 'status', 'cold_wall_s',
        'device', 'device_reason', 'cec_commands', 'mapping_commands', 'replay_commands',
        'abc_native_rewrite_commands', 'tool_seconds_inclusive_json', 'stage_seconds_json',
        'native_forward_calls', 'native_backward_calls', 'native_kernel_ms', 'regions',
        'region_dag_layers', 'max_layer_width', 'decisions', 'candidates'])
    profiles = lines('runtime/native_gpu_profiles.jsonl')
    terminals = lines('benchmark/method_records.jsonl')
    failures = lines('audit/method_failures.jsonl')
    request_failures = lines('audit/request_failures.jsonl')
    score_metadata = lines('prediction/litecut_cache_metadata.jsonl')
    counts = Counter()
    states = Counter()
    feature_ids = {}
    candidate_domains = {}
    domain_values = []
    missing = []
    namespace = {}
    exec(READER, namespace)
    decode_file = namespace['resolve_file']

    for i, input_row in sorted(inputs.items()):
        ident = {'circuit_id': i, 'input': input_row['relative_path'], 'input_sha256': input_row['sha256']}
        ours_point = points[(i, 'Ours')]
        w = Path(ours_point['detail_path'])
        prefix = f'optimization/cases/{i:04d}/work'
        source_summary = w / 'summary.json'
        original_summary = bundle.json(source_summary)
        assert original_summary['status'] == 'PASS'
        assert len(original_summary['observations']) == original_summary['requests_including_noop_failed_repair']
        config = bundle.json(w / 'config.json')
        assert config['input_sha256'] == input_row['sha256']
        for field, name in [('gpuls_sha256', 'gpuls'), ('checkpoint_sha256', 'checkpoint'), ('native_library_sha256', 'native_library')]:
            assert config[field] == tools[name]['sha256']
        for name in ['config.json', 'structure_meta.json', 'litecut.json']:
            bundle.copy(w / name, prefix + '/' + name)
        original_file = prefix + '/original/result.json'
        measured_original = json.loads(bundle.copy(w / 'original/result.json', original_file))
        assert measured_original == original_summary['original']
        counts['ours_original_measurements'] += 1
        observations = original_summary['observations']
        catalogue = bundle.json(w / 'cut_validation/verified_catalogue.json')
        decisions = catalogue['decisions']
        assert len(decisions) == original_summary['decisions']
        for j, d in enumerate(decisions):
            domain = tuple(d['candidate_subgraph_indices'])
            if domain not in candidate_domains:
                candidate_domains[domain] = len(domain_values)
                domain_values.append(list(domain))
            domains.writerow({'circuit_id': i, 'decision_index': j, 'decision_id': d['decision_id'],
                'root': d['root'], 'cut_slot': d['cut_slot'], 'region_id': d['decision_owner_region'],
                'truth': d['truth'], 'truth_class': d['truth_class'],
                'candidate_domain_id': candidate_domains[domain], 'cut_leaves_json': dump(d['cut_leaves']),
                'removable_node_count': d['removable_node_count'], 'saved_count': d['saved_count'],
                'classification': d['classification']})
        counts['decisions'] += len(decisions)
        counts['replacement_candidates'] += sum(d['candidate_count'] for d in decisions)
        assert sum(d['candidate_count'] for d in decisions) == original_summary['candidates']

        observation_files = {}
        metric_files = {}
        source_observations = list((w / 'replay_noop').glob('eval_*/observation.json')) + list((w / 'optimization').glob('eval_*/observation.json'))
        assert len(source_observations) == len(observations)
        for observed in observations:
            n = observed['index']
            request_method = 'replay_noop' if n == 0 else 'optimization'
            rel = f'{request_method}/eval_{n:04d}'
            old_path = w / rel / 'observation.json'
            on_disk = bundle.json(old_path)
            assert on_disk == observed
            assert len(observed['choices']) == len(decisions)
            for j, c in enumerate(observed['choices']):
                assert 0 <= c <= decisions[j]['candidate_count']
            name = prefix + '/' + rel + '/observation.json'
            observation_files[n] = name
            normalized = copy.deepcopy(observed)
            metric_name = None
            if observed['status'] == 'PASS':
                metric_name = prefix + '/' + rel + '/evaluation/result.json'
                metrics = json.loads(bundle.copy(w / rel / 'evaluation/result.json', metric_name))
                assert metrics == observed['metrics'] and metrics['cec'] == 'PASS'
                assert metrics['timing_model'] == 'ABC_SCL_LIBERTY'
                assert metrics['provenance']['original']['sha256'] == ident['input_sha256']
                normalized['metrics'] = ref(metric_name)
                counts['request_success'] += 1
            else:
                assert observed.get('metrics') is None
                emit(request_failures, {**ident, 'observation_file': name, 'data': observed})
                counts['request_fail'] += 1
            metric_files[n] = metric_name
            bundle.write(name, normalized, [old_path], 'Lossless JSON: metrics replaced by a bundle reference to the exact saved result.json')
            bundle.copy(w / rel / 'rewrite/replay.json', prefix + '/' + rel + '/rewrite/replay.json')
            predicted = observed.get('prediction_before_evaluation')
            proof_name = None
            if predicted is not None:
                counts['predicted_requests'] += 1
                assert predicted['choices'] == observed['choices']
                assert all(t < n for t in predicted['training_observation_indices'])
                proof_path = w / 'optimization/predictions' / f'eval_{n:04d}.json'
                if proof_path.is_file():
                    proof = bundle.json(proof_path)
                    assert proof['prediction'] == predicted and proof['choices'] == observed['choices']
                    assert proof['phase'] == 'before_native_replay'
                    proof['prediction'] = ref(name, '/prediction_before_evaluation')
                    proof['choices'] = ref(name, '/choices')
                    proof_name = prefix + f'/optimization/predictions/eval_{n:04d}.json'
                    bundle.write(proof_name, proof, [proof_path], 'Lossless JSON references remove duplicate choices/prediction; original pre-replay phase/time retained')
                    counts['pre_replay_prediction_proofs'] += 1
                else:
                    missing.append({'circuit_id': i, 'request_index': n, 'path': str(proof_path), 'kind': 'missing_pre_replay_proof'})
            metrics = observed.get('metrics') or {}
            p = predicted or {}
            valid = (observed['status'] == 'PASS' and predicted is not None
                     and isinstance(p.get('delay_ps'), (int, float)) and math.isfinite(p['delay_ps'])
                     and isinstance(metrics.get('delay_ps'), (int, float)) and math.isfinite(metrics['delay_ps']))
            counts['valid_global_prediction_pairs'] += int(valid)
            if valid:
                assert len(p['po_arrivals_ps']) == len(metrics['po_arrivals_ps']) == input_row['po_count']
                counts['valid_po_prediction_pairs'] += input_row['po_count']
            active = [[j, c] for j, c in enumerate(observed['choices']) if c]
            baseline = original_summary['replay_noop']['metrics']
            pairs.writerow({**ident, 'request_index': n, 'request_method': observed['method'],
                'status': observed['status'], 'prediction_source': p.get('source'), 'has_prediction': predicted is not None,
                'valid_global_prediction_pair': valid, 'predicted_delay_ps': p.get('delay_ps'),
                'actual_mapping_delay_ps': metrics.get('delay_ps'), 'predicted_area': p.get('area'),
                'actual_mapping_area': metrics.get('area'), 'predicted_objective': p.get('predicted_objective'),
                'anchor_objective': p.get('anchor_objective'),
                'training_observation_indices_json': dump(p.get('training_observation_indices')),
                'noop_mapping_delay_ps': baseline['delay_ps'], 'noop_mapping_area': baseline['area'],
                'original_mapping_delay_ps': measured_original['delay_ps'],
                'final_selected_request': original_summary['selected']['observation_index'] == n,
                'active_choices_json': dump(active), 'request_inclusive_runtime_s': observed['runtime_s'],
                'method_cold_runtime_s': ours_point['end_to_end_runtime_s'], 'observation_file': name,
                'prediction_file': proof_name, 'actual_metrics_file': metric_name, 'error': observed.get('error')})
            counts['requests'] += 1
            counts['noop_requests'] += int(n == 0)
            assert decode_file(bundle.stage, name) == observed

        normalized = copy.deepcopy(original_summary)
        normalized['original'] = ref(original_file)
        normalized['observations'] = [ref(observation_files[o['index']]) for o in observations]
        normalized['replay_noop'] = ref(observation_files[0])
        selected = normalized['selected']
        selected['metrics'] = ref(original_file if selected['source'] == 'original_fallback'
                                  else metric_files[selected['observation_index']])
        summary_name = prefix + '/summary.json'
        bundle.write(summary_name, normalized, [source_summary],
                     'Lossless JSON: duplicate original/observations/no-op/selected metrics replaced by bundle references')
        assert decode_file(bundle.stage, summary_name) == original_summary
        counts['lossless_summary_roundtrips_verified'] += 1

        gradients = sorted((w / 'optimization').glob('gradient_*.json'))
        calibrations = sorted((w / 'optimization').glob('calibration_*.json'))
        assert len(gradients) == original_summary['gradient_rounds']
        for path in gradients + calibrations:
            bundle.copy(path, prefix + '/optimization/' + path.name)
        counts['gradient_records'] += len(gradients)
        counts['calibration_records'] += len(calibrations)
        emit(profiles, {**ident, 'backend': original_summary['backend'], 'native_profiles': original_summary['native_profiles'],
                        'structure': original_summary['structure']})
        for j, stage in enumerate(original_summary['stages']):
            stages.writerow({'circuit_id': i, 'stage_index': j, **stage})
        circuits.writerow({**ident, 'summary_file': summary_name, 'decisions': len(decisions),
            'candidates': original_summary['candidates'], 'regions': original_summary['regions'],
            'device': original_summary['backend']['device'], 'request_count': len(observations),
            'gradient_records': len(gradients), 'calibration_records': len(calibrations),
            'final_source': original_summary['selected']['source'],
            'selected_observation_index': original_summary['selected']['observation_index']})

        litecut = bundle.json(w / 'litecut.json')
        if decisions:
            score_path = Path(litecut['scores_path']).resolve()
            assert score_path.is_relative_to(w.resolve())
            cache = bundle.json(score_path)
            assert cache['identity']['input'] == ident['input_sha256']
            assert cache['identity']['checkpoint'] == tools['checkpoint']['sha256']
            assert len(cache['scores']) == len(decisions) + original_summary['candidates'] == litecut['score_count']
            cache_identity = dict(cache['identity'])
            cache_identity.pop('decisions')
            emit(score_metadata, {**ident, 'original_cache_path': str(score_path),
                'source_sha256': bundle.sources[str(score_path)]['sha256'], 'identity_without_duplicate_decisions': cache_identity,
                'metric': cache['metric'], 'batch_size': cache['batch_size'], 'torch_version': cache['torch_version'],
                'score_rows': len(cache['scores']), 'omitted_identity_decisions_rule': 'Root/cut/domain identities retained in optimization/decisions.csv and candidate_domains.json'})
            expected_order = ((j, c) for j, d in enumerate(decisions) for c in range(d['candidate_count'] + 1))
            known = {'decision_index', 'choice_index', 'subgraph_index', 'region_id', 'continuous_features_19',
                     'feature_hash', 'predicted_local_abc_delay', 'original_cut_truth_verified', 'truth_verified',
                     'output_phase_complement', 'representation', 'cut_validation', 'replacement_dag_info'}
            for row, (j, c) in zip(cache['scores'], expected_order):
                assert (row['decision_index'], row['choice_index']) == (j, c)
                d = decisions[j]
                assert row['subgraph_index'] == (None if c == 0 else d['candidate_subgraph_indices'][c - 1])
                assert row['region_id'] == d['decision_owner_region']
                values = tuple(row['continuous_features_19'])
                h = row['feature_hash']
                if h not in feature_ids:
                    assert len(values) == 19 and hashlib.sha256(struct.pack('<19f', *values)).hexdigest() == h
                    fid = len(feature_ids)
                    feature_ids[h] = (fid, values)
                    features.writerow({'feature_id': fid, 'feature_hash': h, **dict(zip(FEATURES, values))})
                else:
                    fid, previous = feature_ids[h]
                    assert previous == values
                score = row['predicted_local_abc_delay']
                assert isinstance(score, (int, float)) and math.isfinite(score)
                scores.writerow({'circuit_id': i, 'decision_index': j, 'choice_index': c,
                    'subgraph_index': row['subgraph_index'], 'region_id': row['region_id'], 'feature_id': fid,
                    'predicted_local_abc_delay': score, 'ground_truth_local_abc_delay': None,
                    **{k: row.get(k) for k in ['original_cut_truth_verified', 'truth_verified', 'output_phase_complement', 'representation']},
                    'extra_metadata_json': dump({k: v for k, v in row.items() if k not in known})})
                counts['litecut_score_rows'] += 1
            counts['scored_designs'] += 1
            del cache
        else:
            assert litecut['status'] == 'NOT_REQUIRED_NO_CANDIDATES'
            counts['no_candidate_designs'] += 1

        for method in ['A', 'B', 'Ours']:
            p = points[(i, method)]
            terminal_path = experiment / 'output/records' / f'{i:04d}_{method}.json'
            terminal = bundle.json(terminal_path)
            assert terminal['input_sha256'] == ident['input_sha256'] and terminal['source_commit'] == SOURCE_COMMIT
            emit(terminals, {**ident, 'method': method, 'data': terminal})
            states[method + '/' + terminal['status']] += 1
            if terminal['status'] != 'PASS':
                emit(failures, {**ident, 'method': method, 'data': terminal})
                for name in ['native_rewrite.command.json', 'native_rewrite.log']:
                    path = Path(p['detail_path']) / name
                    bundle.copy(path, f'audit/failures/{i:04d}_{method}/' + name)
            elif method in ['A', 'B']:
                result_file = f'benchmark/cases/{i:04d}_{method}/evaluation/result.json'
                metrics = json.loads(bundle.copy(Path(p['detail_path']) / 'evaluation/result.json', result_file))
                assert metrics['delay_ps'] == p['delay_ps'] and metrics['cec'] == 'PASS'
                assert metrics['timing_model'] == 'ABC_SCL_LIBERTY'
                counts['baseline_measurements'] += 1
            runtimes.writerow({'circuit_id': i, 'method': method, 'status': p['status'],
                'cold_wall_s': p['end_to_end_runtime_s'], 'device': p.get('device'), 'device_reason': p.get('device_reason'),
                **{k: p.get(k) for k in ['cec_commands', 'mapping_commands', 'replay_commands',
                   'abc_native_rewrite_commands', 'native_forward_calls', 'native_backward_calls',
                   'native_kernel_ms', 'regions', 'region_dag_layers', 'max_layer_width', 'decisions', 'candidates']},
                'tool_seconds_inclusive_json': dump(p.get('tool_seconds_inclusive')),
                'stage_seconds_json': dump(p.get('stage_seconds'))})
        counts['designs'] += 1
        if i % 50 == 0 or i == 904:
            print(dump({'phase': 'collect', 'designs': counts['designs'], 'requests': counts['requests'],
                        'litecut_rows': counts['litecut_score_rows'], 'unique_feature_vectors': len(feature_ids)}), flush=True)

    # The shared failed GPU command must be counted once, although it failed two requests.
    failure_work = experiment / 'output/case_0104_Ours/work/optimization/_batch_0001'
    for name in ['rewrite.command.json', 'rewrite.log', 'validation.json', 'batch_replay.json']:
        path = failure_work / name
        if path.is_file():
            bundle.copy(path, 'audit/failures/0104_Ours_shared_batch/' + name)
    for stream in streams:
        stream.close()
    assert counts['designs'] == counts['lossless_summary_roundtrips_verified'] == 905
    assert counts['requests'] == 3041 and counts['request_success'] == 3039 and counts['request_fail'] == 2
    assert counts['valid_global_prediction_pairs'] == counts['request_success'] - counts['noop_requests']
    assert counts['baseline_measurements'] + counts['ours_original_measurements'] + counts['request_success'] == 5750
    assert counts['gradient_records'] == 1589 and counts['calibration_records'] == 777
    assert counts['scored_designs'] == 811 and counts['no_candidate_designs'] == 94
    assert states == {'A/PASS': 905, 'B/PASS': 901, 'B/FAIL': 4, 'Ours/PASS': 905}
    counts['unique_feature_vectors'] = len(feature_ids)
    counts['candidate_domains'] = len(domain_values)
    bundle.write('optimization/candidate_domains.json', domain_values,
        rule='Domain ID is array index; choice 0=no-op, choice k>0 maps to domain[k-1] subgraph_index')
    stats = {'counts': dict(counts), 'method_states': dict(states), 'three_method_terminal_designs': 905,
        'three_method_successful_quality_designs': 901, 'local_litecut_ground_truth_pairs': 0,
        'source_commit': SOURCE_COMMIT, 'original_measurement_count': 5750, 'unexpected_missing': missing,
        'scope': 'Coverage counts only; no new optimization, model evaluation, timing measurement or performance test.'}
    bundle.write('audit/coverage.json', stats)
    bundle.write('audit/MISSING_DATA.json', {'unexpected_missing': missing,
        'not_recorded_in_original_experiment': [
            'Candidate-local measured abc_delay labels: absent, so Student local prediction accuracy cannot be computed from global Mapping labels.',
            'Full per-candidate autograd tensors: only saved norms and capped ranked directional edits exist.',
            'Per-method peak GPU VRAM and an independent full same-algorithm CPU/GPU comparison: not recorded.',
            'Physical AIG/netlist/library/model/binary payloads intentionally not included; identities and hashes retained.'],
        'failed_quality_fields': 'Four B native rewrite failures and two Ours request failures have no successful timing label. No-op requests have no pre-evaluation proxy prediction.',
        'intentional_extraction_omissions': [
            'LiteCut cut_validation and replacement_dag_info repeated Boolean-proof/materialization payloads; all choice identities, truth flags, predictions, feature hashes and exact 19D features retained.',
            'Catalogue removable-node lists, conflict-edge lists and repeated Boolean proof payloads; all decision IDs, root/cut/region/domain identities and saved/removable counts retained.']})
    bundle.write('runtime/timing_semantics.json', {'cold_wall': 'Authoritative method point wall includes all required preparation and real validation; OS page cache not flushed.',
        'request_stage_tool_times': 'Inclusive, shared, nested or concurrent; never sum as exclusive wall fractions.',
        'native_cuda': 'kernel_ms_since_collect are disjoint event intervals; calls_cumulative is cumulative. Device-none and CPU paths are not CUDA observations.',
        'experiment_gpu': 'Original session_65245.json; CPU packaging hardware is unrelated.',
        'causal_gpu_speedup': 'Not independently established by A/B/Ours comparisons.'})

    readme = f'''# 最新 MFFC GPU 梯度优化：905 电路分析包

本包只整理已存在的正式实验及其后处理，不执行 CULS、ABC、CEC、Mapping、LiteCut、优化器或新性能测试。

- 算法冻结源码：`{SOURCE_COMMIT}`；正式作业65245。
- 结果来自：`{experiment}`。
- 含成本的后处理：`{delivery}`，后处理源码`{summary['delivery_supplement']['analysis_source_commit']}`。
- 905个电路均有A/B/Ours终态，共2715点；901个电路有三方法全部成功的真实Delay/Area/CEC，另外4个B失败而A/Ours成功。
- 方法状态：A 905 PASS，B 901 PASS/4 FAIL，Ours 905 PASS；没有方法超时。失败和no-op/回退全部保留。
- Ours候选组合请求{counts['requests']}个：成功{counts['request_success']}、失败{counts['request_fail']}，含{counts['noop_requests']}个no-op；另有905次Ours自付费original测量，不混作候选请求。
- 有评估前代理预测的请求{counts['predicted_requests']}个；有效global预测—真实Mapping配对{counts['valid_global_prediction_pairs']}条；有效逐PO配对{counts['valid_po_prediction_pairs']}条，可从对应JSON的数组按PO位置展开。失败预测也保留。
- 有Student局部分数的设计{counts['scored_designs']}个，无候选{counts['no_candidate_designs']}个；{counts['decisions']}个Decision、{counts['replacement_candidates']}个physical candidates，连同no-op共{counts['litecut_score_rows']}条局部分数。没有candidate-local真实abc_delay标签，局部预测准确性的有效真值配对为0；不能把global Mapping Delay当作局部标签。
- 已保存梯度记录{counts['gradient_records']}份、代理校准{counts['calibration_records']}份；仅保存过的梯度norm/方向排名可分析，不伪造完整梯度张量。

## 目录与推荐读取顺序

| 目录 | 内容 |
|---|---|
| `benchmark/` | 原始完整CSV/JSON/MD与三组图，2715终态JSONL；A/B成功的全部PO/真实测量JSON |
| `optimization/` | 905个`cases/NNNN/work/summary.json`、全部observation、精确replay、gradient/calibration和评估前预测存证；Decision表及candidate域字典 |
| `prediction/` | `global_request_pairs.csv`全部3041请求，`litecut_scores.csv`全部局部分数，去重的精确19维特征，以及Student缓存身份 |
| `runtime/` | 2715方法冷wall/工具调用、各阶段时钟、保存的原生GPU profiles及计时语义 |
| `config/` | 正式协议、冻结输入/工具/源代码小快照、原始GPU环境和实现说明、无ML依赖的JSON引用读取器 |
| `audit/` | 已有audited_records/installation_details、失败、原始证据hash索引、本次来源及文件清单、完整性与缺失说明 |

先读`audit/coverage.json`、`benchmark/BENCHMARK905_RESULTS.csv`与`prediction/global_request_pairs.csv`；再按circuit_id/request_index读取详细JSON。不要用CSV物理行数代替记录数，原始benchmark CSV含加引号的多行错误文本。

## 身份与指标

`circuit_id`就是冻结清单原始index（0–904），与input及SHA256联合核对。方法固定A/B/Ours。Ours的request_index=0为no-op，其余为同一原图上的独立组合，不是累计改图。最终source和observation_index原样保留；original_fallback的选择向量是null，不冒充no-op。

Decision由本电路内`decision_index`对应原始`(root,cut_slot)`和owner MFFC；choice=0是no-op，choice=k>0通过`candidate_domain_id`及`candidate_domains.json[domain_id][k-1]`找到subgraph_index。MFFC、Decision、cut-bounded region、candidate及不同节点命名空间不可混用。

严格区分：`predicted_local_abc_delay`是冻结Student局部代理；`predicted_delay_ps`是评估前多PO可微代理；`actual_mapping_delay_ps`才是真实`ABC_SCL_LIBERTY`。真实PO数组保存在对应evaluation/result.json，预测PO数组保存在observation的prediction_before_evaluation。失败真实值缺失不填0；no-op缺预测也不填0。

有效global配对要求请求PASS、有有限的评估前predicted delay，并有同一次实际CEC PASS、ABC_SCL_LIBERTY测量。所有training_observation_indices均小于当前request_index。此数据是在线选择/反馈后的样本，不是独立未见集。

## 压缩提取规则（不损失搜索记录的JSON字段）

原始summary/observation反复包含相同真实PO数组。包内保留每份原始JSON的全部语义字段，仅把重复的original、observations、replay_noop、selected.metrics、observation.metrics及评估前存证中的重复prediction/choices替换为`{{"$bundle_ref":{{"file":"相对包根路径","pointer":"JSON pointer"}}}}`。真实metrics JSON、gradient、calibration、replay、配置均逐字节复制。全部905份summary及3041份observation已逐一解引用，并与原始对象全字段相等比较通过。

解压后用：

```python
import sys
sys.path.insert(0, '/解压目录/config')
from bundle_reader import resolve_file
summary = resolve_file('/解压目录', 'optimization/cases/0047/work/summary.json')
```

局部分数不筛选：所有physical及no-op均保留。`feature_id`连接feature_vectors.csv，精确保存原19维数值和feature_hash；相同hash的数值逐项核对后才去重。仅省略重复的cut_validation/replacement_dag_info证明与物化图细节，以及catalogue的大节点/冲突列表；它们不是局部或全局真实timing标签。原路径、原字节SHA256和提取规则在audit清单。连续值不舍入，不重做归一化，不重新推理。

按要求原样附上的reference_docs状态说明含旧工作背景，只作实现与历史边界参考；其旧905/16电路数值不属于本包机器数据或统计样本。本包没有读取旧905或16电路运行目录来提取任何实验结果。

## 完整性、缺失与成本边界

已核对905×3唯一终态、全部905份Ours summary、请求序号/选择域、预测存证、全部已有梯度/校准文件及版本。意外缺失{len(missing)}项，详见`audit/MISSING_DATA.json`；未记录的局部真值、完整梯度和显存峰值明确列为未记录，不重新生成。

四项B失败为codec/loco_top/mrc3/mrc4的native rewrite断言。两个内部请求失败来自BRAM_SSP同一次gpuls CUDA heap分配错误；它们的inclusive耗时重叠，共享命令只计一次。方法最终905 PASS不等于内部请求零失败。runtime分项可重叠，冷wall优先；原生GPU事件不是整个GPU或方法耗时。最新实验没有独立的全量同算法CPU/GPU对照，不能由A/B/Ours时间比推断纯GPU因果加速。

## ZIP大小与SHA256

**完整ZIP的精确大小和SHA256见ZIP同目录的外部`README.md`、`{ARCHIVE}.metadata.json`和`{ARCHIVE}.sha256`。** 将ZIP自身的SHA256再写入ZIP内部会改变被校验文件，形成自引用，故内部README只说明外部校验位置，不填伪造或过期hash。内部`audit/CHECKSUMS.sha256`校验所有其余成员；`audit/FILE_MANIFEST.jsonl`和`SOURCE_FILES.jsonl`记录包内文件及服务器原路径/原hash。整个ZIP在形成后逐成员读取、CRC检查并核对SHA256，再生成外部大小/hash。
'''
    bundle.text('README.md', readme)
    with bundle.jsonl('audit/SOURCE_FILES.jsonl', rule='Each actually consumed original source, exact SHA256/size/mtime') as stream:
        for path, metadata in sorted(bundle.sources.items()):
            emit(stream, metadata)
    with bundle.jsonl('audit/FILE_MANIFEST.jsonl', rule='All members except self/checksum; exact hashes and extraction provenance') as stream:
        for p in sorted(bundle.stage.rglob('*')):
            if p.is_file() and p.name not in ['FILE_MANIFEST.jsonl', 'CHECKSUMS.sha256']:
                relative = str(p.relative_to(bundle.stage))
                emit(stream, {'path': relative, 'sha256': sha(p), 'size_bytes': p.stat().st_size,
                    **bundle.origins.get(relative, {'rule': 'Generated structured extraction; join source index by circuit ID'})})
    checksum = []
    for p in sorted(bundle.stage.rglob('*')):
        if p.is_file() and p.name != 'CHECKSUMS.sha256':
            checksum.append(sha(p) + '  ' + str(p.relative_to(bundle.stage)))
    bundle.text('audit/CHECKSUMS.sha256', '\n'.join(checksum) + '\n')
    archive = run / 'output' / ARCHIVE
    print(dump({'phase': 'zip', 'counts': dict(counts)}), flush=True)
    files = [p for p in sorted(bundle.stage.rglob('*')) if p.is_file()]
    for p in files:
        assert p.suffix.lower() not in {'.aig', '.aag', '.v', '.lib', '.pt', '.so', '.o', '.a'}
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
        for p in files:
            z.write(p, str(p.relative_to(bundle.stage)))
    print(dump({'phase': 'verify_archive', 'bytes': archive.stat().st_size}), flush=True)
    expected_hashes = {line.split('  ', 1)[1]: line.split('  ', 1)[0] for line in checksum}
    with zipfile.ZipFile(archive) as z:
        assert len(z.namelist()) == len(set(z.namelist())) == len(files)
        assert set(z.namelist()) == {str(p.relative_to(bundle.stage)) for p in files}
        for info in z.infolist():
            h = hashlib.sha256()
            with z.open(info) as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    h.update(chunk)
            if info.filename in expected_hashes:
                assert h.hexdigest() == expected_hashes[info.filename], info.filename
        assert z.testzip() is None
        archived_coverage = json.loads(z.read('audit/coverage.json'))
        assert archived_coverage == stats
    metadata = {'status': 'VERIFIED_ARCHIVE', 'path': str(archive), 'size_bytes': archive.stat().st_size,
        'size_MiB': archive.stat().st_size / 1024**2, 'sha256': sha(archive), 'members': len(files),
        'uncompressed_bytes': sum(p.stat().st_size for p in files), 'counts': stats,
        'zip_crc_and_all_member_sha256_checked': True, 'created_at': datetime.now(timezone.utc).isoformat(),
        'slurm_job_id': os.environ['SLURM_JOB_ID'], 'new_experiment_or_model_calls': 0}
    with Path(str(archive) + '.metadata.json').open('x') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
        f.write('\n')
    with Path(str(archive) + '.sha256').open('x') as f:
        f.write(metadata['sha256'] + '  ' + archive.name + '\n')
    with (run / 'output/README.md').open('x') as f:
        f.write('# 已核验ZIP交付\n\n' + f'ZIP：`{archive}`\n\n大小：{metadata["size_bytes"]} bytes（{metadata["size_MiB"]:.2f} MiB）。\n\nSHA256：`{metadata["sha256"]}`\n\n' + readme)
    with (run / 'output/packaging_completion.json').open('x') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(dump({k: metadata[k] for k in ['status', 'path', 'size_bytes', 'sha256', 'members']}), flush=True)


if __name__ == '__main__':
    main()
