"""Path comparisons preserve ties, serialization controls and failed-request gaps."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from global_timing.common import digest
from global_timing.evaluation import parse_timing
from global_timing.path_analysis import parse_paths
from global_timing.path_changes import (aggregate_cases, aggregate_prediction_changes, analyze_case,
                                       compare_states, endpoint_state, prediction_changes, references)
from scripts.analyze_path_changes import snapshot_reports, validate_measurements


def state(pi=0, po=0, phase=0, delay=10, near=(0, 1)):
    return {'delay_ps': delay, 'area': 20, 'critical_po': [po], 'critical_pairs': [[pi, po]],
            'critical_transitions': [[pi, po, phase]], 'near_critical_po': list(near)}


def fixture():
    metrics = {'input_sha256': 'baseline', 'area': 20, 'delay_ps': 10}
    base = {'status': 'PASS', 'index': 0, 'choices': [0], 'metrics': metrics, 'baseline_reused': True}
    changed = {'status': 'PASS', 'index': 2, 'choices': [1],
               'metrics': dict(metrics, input_sha256='changed', delay_ps=9),
               'prediction_before_evaluation': {'source': 'gradient'}}
    summary = {'original': dict(metrics, input_sha256='original'), 'native_abc': metrics,
        'replay_noop': base, 'random_search': [base],
        'greedy_litecut': {'observations': [base], 'best': base},
        'independent_regions': {'observations': [base], 'best': base},
        'gradient_search': {'joint': {'observations': [base, {'status': 'FAIL', 'index': 1}, changed], 'best': changed}}}
    states = {'original': state(pi=1), 'baseline': state(), 'changed': state(pi=1, delay=9)}
    links = []
    for item in references(summary):
        key = item['metrics']['input_sha256']
        links.append({'method': item['method'], 'index': item['index'], 'input_sha256': key,
            'path_evidence': key, 'critical_pairs': states[key]['critical_pairs'],
            'critical_source_endpoint_pairs_changed': key == 'changed'})
    return summary, {'status': 'PASS', 'selection_path_changes': links}, states


class PathChangeTests(unittest.TestCase):
    def test_predictions_join_only_same_successful_path_observations(self):
        from test_search_analysis import summary_fixture
        summary = summary_fixture()
        rows = summary['gradient_search']['area_constraint']['observations']
        rows[3]['metrics']['po_arrivals_ps'] = [95., 80.]
        paths = {'observations': []}
        for row in rows:
            if row['status'] == 'PASS' and not row.get('baseline_reused'):
                row['metrics']['input_sha256'] = 'fixture-' + str(row['index'])
                paths['observations'].append({'method': 'gradient_area_constraint', 'index': row['index'],
                    'input_sha256': row['metrics']['input_sha256'], 'baseline_anchor': False,
                    'vs_noop_delay_delta_ps': row['metrics']['delay_ps'] - 100.,
                    'vs_noop_critical_po_changed': row['index'] == 3,
                    'vs_noop_observed_critical_pairs_changed': row['index'] == 3,
                    'vs_noop_observed_critical_transitions_changed': row['index'] == 3})
        samples = prediction_changes(summary, paths)
        self.assertEqual([sample['index'] for sample in samples], [2, 3])
        self.assertEqual([sample['endpoint_change'] for sample in samples],
                         ['SAME_OBSERVED_ENDPOINT_PAIR_SET', 'CRITICAL_PO_SET_CHANGED'])
        self.assertEqual([sample['normalized_absolute_error'] for sample in samples], [.1, .1])
        for corruption in ['missing', 'duplicate', 'input', 'delta', 'baseline', 'future_label']:
            changed, altered = copy.deepcopy(paths), copy.deepcopy(summary)
            if corruption == 'missing':
                changed['observations'].pop()
            elif corruption == 'duplicate':
                changed['observations'].append(changed['observations'][0])
            elif corruption == 'input':
                changed['observations'][0]['input_sha256'] = 'another run'
            elif corruption == 'delta':
                changed['observations'][0]['vs_noop_delay_delta_ps'] = 100.
            elif corruption == 'baseline':
                changed['observations'][0]['baseline_anchor'] = True
            else:
                altered['gradient_search']['area_constraint']['observations'][2][
                    'prediction_before_evaluation']['training_observation_indices'] = [2]
            with self.subTest(corruption=corruption), self.assertRaises(ValueError):
                prediction_changes(altered, changed)

    def test_prediction_groups_weight_designs_and_exclude_zero_references(self):
        sample = {'mode': 'joint', 'endpoint_change': 'SAME_OBSERVED_ENDPOINT_PAIR_SET',
                  'normalized_absolute_error': .1, 'actual_change_exceeds_tolerance': True,
                  'direction_correct': True}
        cases = [{'analysis': {'prediction_samples': [sample] * 10}},
                 {'analysis': {'prediction_samples': [dict(sample, normalized_absolute_error=.9,
                                                          direction_correct=False)]}},
                 {'analysis': {'prediction_samples': [dict(sample, normalized_absolute_error=None,
                                                          actual_change_exceeds_tolerance=False)]}}]
        group, = aggregate_prediction_changes(cases)
        self.assertEqual((group['design_count'], group['sample_count'], group['normalizable_designs']), (3, 12, 2))
        self.assertAlmostEqual(group['mean_per_design_normalized_mae'], .5)
        self.assertEqual(group['designs_with_actual_delay_change'], 2)
        self.assertEqual(group['mean_per_design_direction_accuracy'], .5)

    def test_distinguishes_endpoint_source_phase_ties_and_zero_delay(self):
        base = state()
        source = compare_states(base, state(pi=1))
        self.assertTrue(source['observed_pair_change_with_same_po_set'])
        self.assertFalse(source['critical_po_changed'])
        endpoint = compare_states(base, state(po=1))
        self.assertTrue(endpoint['critical_po_changed'])
        self.assertFalse(endpoint['observed_pair_change_with_same_po_set'])
        phase = compare_states(base, state(phase=1))
        self.assertTrue(phase['observed_critical_transitions_changed'])
        self.assertFalse(phase['observed_critical_pairs_changed'])
        tied = dict(base, critical_po=[0, 1], critical_pairs=[[0, 0], [1, 1]],
                    critical_transitions=[[0, 0, 0], [1, 1, 1]])
        self.assertTrue(compare_states(base, tied)['critical_po_changed'])
        self.assertTrue(compare_states(base, state(near=(0,)))['near_critical_po_changed'])
        self.assertIsNone(compare_states(state(delay=0), state(delay=0))['delay_improvement_pct'])
        for value in [float('nan'), float('inf'), -1]:
            with self.assertRaises(ValueError):
                compare_states(base, state(delay=value))

    def test_failed_requests_remain_gaps_and_serialization_is_separate(self):
        summary, report, states = fixture()
        result = analyze_case(summary, report, states)
        self.assertTrue(result['serialization_control']['observed_critical_pairs_changed'])
        transition, = result['transitions']
        self.assertEqual((transition['previous_successful_index'], transition['index']), (0, 2))
        self.assertEqual(transition['failed_requests_since_previous_success'], 1)
        selected = [row for row in result['observations'] if row['is_saved_final_selection']]
        self.assertEqual(len(selected), 3)  # Includes two saved no-op deterministic results.
        self.assertFalse(any(row['method'] == 'random_search' for row in selected))
        aggregate = aggregate_cases([{'analysis': result}])
        self.assertEqual(aggregate['methods']['gradient_joint']['successful_nonanchor_observations'], 1)
        self.assertEqual(aggregate['methods']['gradient_joint']['transitions_spanning_failed_requests'], 1)
        self.assertEqual(aggregate['methods']['greedy_litecut']['successful_nonanchor_observations'], 0)
        self.assertEqual(aggregate['methods']['greedy_litecut']['saved_final_selections'], 1)

    def test_missing_duplicate_invented_and_inconsistent_evidence_rejected(self):
        for corruption in ['missing', 'duplicate', 'invented', 'metrics', 'flag', 'extra_trace', 'failed_audit', 'bad_best']:
            with self.subTest(corruption=corruption):
                summary, report, states = fixture()
                if corruption == 'missing':
                    report['selection_path_changes'].pop()
                elif corruption == 'duplicate':
                    report['selection_path_changes'].append(report['selection_path_changes'][0])
                elif corruption == 'invented':
                    report['selection_path_changes'][-1]['index'] = 1
                elif corruption == 'metrics':
                    states['changed']['delay_ps'] = 8
                elif corruption == 'flag':
                    report['selection_path_changes'][-1]['critical_source_endpoint_pairs_changed'] = False
                elif corruption == 'extra_trace':
                    states['extra'] = state()
                elif corruption == 'failed_audit':
                    report['status'] = 'FAIL'
                else:
                    summary['gradient_search']['joint']['best'] = dict(summary['replay_noop'], index=1)
                with self.assertRaises(ValueError):
                    analyze_case(summary, report, states)

    def test_critical_ties_are_canonical_and_must_match_timing(self):
        trace = {'near_critical_ratio': .95, 'paths': [
            {'po_index': 0, 'pi_index': 1, 'output_phase': 1, 'critical': True, 'arrival_reconstruction': 'PASS'},
            {'po_index': 1, 'pi_index': -1, 'output_phase': 0, 'critical': True, 'arrival_reconstruction': 'PASS'}]}
        result = endpoint_state(trace, {'delay_ps': 0, 'area': 0, 'critical_po_indices': [0, 1]})
        self.assertEqual(result['critical_pairs'], [[-1, 1], [1, 0]])
        with self.assertRaises(ValueError):
            endpoint_state(trace, {'delay_ps': 0, 'area': 0, 'critical_po_indices': [0]})

    def test_report_priority_preserves_failure_and_exact_source_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = [root / 'first', root / 'last']
            for number, run in enumerate(runs):
                directory = run / 'output/arrival_paths/case'
                directory.mkdir(parents=True)
                (run / 'run.json').write_text(json.dumps({'source_commit': str(number), 'state': 'RUNNING'}))
                (directory / 'summary.json').write_text(json.dumps({'status': 'PASS' if number == 0 else 'FAIL',
                    'source_benchmark': str(root / 'original_benchmark')}))
            other = runs[-1] / 'output/arrival_paths/other'
            other.mkdir()
            (other / 'summary.json').write_text(json.dumps({'status': 'PASS', 'source_benchmark': str(root / 'retry')}))
            chosen, _, invalid = snapshot_reports(runs, root / 'output')
            self.assertFalse(invalid)
            self.assertEqual(chosen[str(root / 'original_benchmark')]['report']['status'], 'FAIL')
            self.assertEqual(chosen[str(root / 'retry')]['report']['status'], 'PASS')
            # A partial live JSON file is retained and reported, never analyzed as PASS.
            (other / 'summary.json').write_text('{')
            _, _, invalid = snapshot_reports(runs, root / 'another_output')
            self.assertEqual(len(invalid), 1)

    def test_raw_measurement_validation_rejects_tampered_cec_paths_and_netlist(self):
        # Synthetic parser fixture only; no claim of executing ABC or an experiment.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            benchmark, work = root / 'benchmark', root / 'audit/graph'
            work.mkdir(parents=True)
            (benchmark / 'original').mkdir(parents=True)
            (benchmark / 'native_abc/evaluation').mkdir(parents=True)
            (benchmark / 'noop/evaluation').mkdir(parents=True)
            original = root / 'original.aig'
            original.write_bytes(b'aig 1 1 0 1 0\n2\n')
            native, noop = benchmark / 'native_abc/rewritten.aig', benchmark / 'noop/replay.aig'
            native.write_bytes(original.read_bytes())
            noop.write_bytes(original.read_bytes())
            provenance = {}
            for name in ['abc', 'library']:
                path = root / name
                path.write_text('parser fixture')
                provenance[name] = {'path': str(path), 'sha256': digest(path)}
            provenance['original'] = {'path': str(original), 'sha256': digest(original)}
            log = ('GT_META 20 10 1 1\nGT_PO 0 10 9\nGT_PATH_META 0.95 1\n'
                   'GT_PATH_BEGIN 0 0 10\nGT_PATH_NODE 0 0 5 0 10 CELL NAND\n'
                   'GT_PATH_ARC 0 0 5 1 0 1 0 0 10 0 10 0 2\n'
                   'GT_PATH_NODE 0 1 1 1 0 PI -\nGT_PATH_END 0 2 0\n')
            mapped = 'module fixture; endmodule'
            for path in [work / 'mapped.v', benchmark / 'original/mapped.v',
                         benchmark / 'native_abc/evaluation/mapped.v', benchmark / 'noop/evaluation/mapped.v']:
                path.write_text(mapped)
            metrics = dict(parse_timing(log), status='PASS', cec='PASS', provenance=provenance,
                           input_sha256=digest(original), mapped_sha256=digest(work / 'mapped.v'))
            (work / 'mapping.log').write_text(log)
            (work / 'cec.log').write_text('Networks are equivalent\n')
            (work / 'result.json').write_text(json.dumps(metrics))
            (work / 'paths.json').write_text(json.dumps(parse_paths(log, metrics, 1)))
            baseline = {'index': 0, 'status': 'PASS', 'metrics': metrics, 'choices': [0],
                        'aig': str(noop), 'baseline_reused': True}
            (benchmark / 'noop/observation.json').write_text(json.dumps(baseline))
            (benchmark / 'noop/evaluation/result.json').write_text(json.dumps(metrics))
            summary = {'original': metrics, 'native_abc': metrics, 'replay_noop': baseline,
                'random_search': [baseline], 'greedy_litecut': {'observations': [baseline], 'best': baseline},
                'independent_regions': {'observations': [baseline], 'best': baseline}, 'gradient_search': {},
                'config': {'timeout_s': 300}}
            measurement = {'status': 'PASS', 'paths': str(work / 'paths.json'), 'aig': str(original),
                'input_sha256': digest(original), 'source_benchmark': str(benchmark), 'reference_abc': provenance['abc'],
                'observer_abc_sha256': provenance['abc']['sha256'], 'fresh_cec': 'PASS',
                'measured_metrics_match': True, 'mapped_netlist_body_match': True,
                'near_critical_path_count': 1, 'critical_pairs': [[0, 0]],
                'mapped_reference_body_sha256': hashlib.sha256(mapped.encode()).hexdigest(),
                'timing': {key: metrics[key] for key in ['area', 'delay_ps', 'po_arrivals_ps']}}
            report = {'source_benchmark': str(benchmark), 'status': 'PASS', 'measurements': [measurement],
                'selection_path_changes': [{'method': row['method'], 'index': row['index'],
                    'path_evidence': measurement['paths'], 'input_sha256': digest(original),
                    'critical_pairs': [[0, 0]], 'critical_source_endpoint_pairs_changed': False}
                    for row in references(summary)]}
            report_path = work.parent / 'summary.json'
            states, artifacts = validate_measurements(summary, report, report_path)
            self.assertEqual(len(states), 1)
            self.assertIn(str(work / 'mapping.log'), artifacts)
            self.assertEqual(len(analyze_case(summary, report, states)['observations']), 6)
            for path, corrupt in [(work / 'cec.log', 'Networks are NOT EQUIVALENT\nNetworks are equivalent\n'),
                                  (work / 'paths.json', '{}'), (work / 'mapped.v', mapped + ' tampered')]:
                previous = path.read_text()
                path.write_text(corrupt)
                with self.assertRaises(ValueError):
                    validate_measurements(summary, report, report_path)
                path.write_text(previous)
