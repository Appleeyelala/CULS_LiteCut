"""Analytical final-summary fixtures, not actual circuit or timing evidence."""
import copy
import unittest

from global_timing.joint_results import matrix, ratio, summarize


def fixture():
    variants = ['legacy_cpu', 'adaptive_batch_cpu', 'adaptive_batch_auto', 'adaptive_no_inner_batch_auto',
                'random_batch', 'native_abc', 'native_culs']
    protocol = {'seeds': [11, 12], 'variants': variants, 'primary_variant': 'adaptive_batch_auto',
                'cutoffs_s': [30, 120, 600], 'qor_noninferiority': {
                    'geometric_mean_delay_ratio_to_legacy_max': 1.01, 'p90_delay_ratio_to_legacy_max': 1.05}}
    panel = {'rows': [{'index': i, 'relative_path': f'fixture{i}.aig', 'sha256': f'{i:064x}',
                       'and_bin': [128, 4096], 'and_count': 1000} for i in (1, 2)]}
    rows = []
    for design in panel['rows']:
        for seed in protocol['seeds']:
            for variant in variants:
                delay, wall = {'legacy_cpu': (100., 16.), 'adaptive_batch_cpu': (100.5, 20.),
                    'adaptive_batch_auto': (100.5, 10.), 'adaptive_no_inner_batch_auto': (101., 12.),
                    'random_batch': (101., 8.), 'native_abc': (102., 3.), 'native_culs': (99., 2.)}[variant]
                rows.append({'index': design['index'], 'input': design['relative_path'], 'seed': seed,
                    'variant': variant, 'input_sha256': design['sha256'], 'status': 'PASS',
                    'timing_model': 'ABC_SCL_LIBERTY', 'delay_ps': delay, 'area': 10.,
                    'area_feasible_1p05': True, 'end_to_end_with_input_staging_s': wall,
                    'slurm_job_id': 'analytical_fixture', 'requests': 8, 'failed_requests': 0,
                    'direct_requests': 0, 'direct_successes': 0,
                    'proposal_devices': ['cuda' if design['index'] == 2 and variant == 'adaptive_batch_auto' else 'cpu'],
                    'request_sequence_sha256': f'requests-{design["index"]}-{seed}',
                    'best_choices_sha256': f'choices-{design["index"]}-{seed}',
                    'algorithm_sha256': {'fixture': 'code'}, 'checkpoint_sha256': 'model', 'domain_sha256': 'domain',
                    'joint_proposal_config': {'inner_updates': True, 'device': 'cpu'}, 'torch_num_threads': 1,
                    'native_replay_commands': 5, 'cec_commands': 8, 'mapping_commands': 8,
                    'noop_delay_ps': 110., 'noop_area': 10., 'time_to_quality_s': wall,
                    'final_source': 'analytical_fixture',
                    'points': [{'index': 0, 'completed_s': wall, 'source': 'analytical_fixture',
                                'delay_ps': delay, 'area': 10.}],
                    'stages': {'nested_fixture': {'inclusive_s': 9999., 'calls': 1},
                               'gradient_proposals': {'inclusive_s': 9999., 'calls': 2}},
                    **{f'best_at_{t}s_ps': delay for t in (30, 120, 600)}})
    return rows, panel, protocol


class JointResultsTests(unittest.TestCase):
    def test_frozen_gates_keep_seed_and_design_denominators_separate(self):
        rows, panel, protocol = fixture()
        result = summarize(rows, panel, protocol)
        old = result['comparisons']['legacy_cpu']
        self.assertEqual(old['expected_seed_pairs'], 4)
        self.assertEqual(old['complete_designs'], 2)
        self.assertAlmostEqual(old['geomean_delay_ratio_seed_pairs'], 1.005)
        self.assertAlmostEqual(old['median_shared_allocation_speed'], 1.6)
        self.assertEqual(result['protocol_assessment']['qor_noninferiority_vs_legacy'], 'MET_ON_THIS_PANEL')
        self.assertEqual(result['protocol_assessment']['median_speed_at_least_1p5_vs_legacy'], 'MET_ON_THIS_PANEL')
        self.assertEqual(result['gpu']['actual_cuda_seed_pairs'], 2)
        self.assertEqual(result['gpu']['actual_cuda_designs'], 1)
        self.assertAlmostEqual(result['gpu']['median_end_to_end_speed_identical_cuda_pairs'], 2.)
        self.assertEqual(result['gpu']['cpu_or_no_proposal_seed_pairs'], 2)

    def test_failed_or_infeasible_old_control_does_not_pass_subset_gate(self):
        for failed in (True, False):
            rows, panel, protocol = fixture()
            row = next(r for r in rows if r['index'] == 1 and r['seed'] == 11 and r['variant'] == 'legacy_cpu')
            if failed:
                row.update(status='FAIL', points=[])
            else:
                row['area_feasible_1p05'] = False
            result = summarize(rows, panel, protocol)
            self.assertEqual(result['comparisons']['legacy_cpu']['measured_area_feasible_seed_pairs'], 3)
            self.assertEqual(result['protocol_assessment']['qor_noninferiority_vs_legacy'], 'INCOMPLETE')

    def test_final_prefix_duplicate_or_wrong_timing_contract_is_rejected(self):
        rows, panel, protocol = fixture()
        with self.assertRaises(ValueError):
            matrix(rows[:-1], panel, protocol)
        with self.assertRaises(ValueError):
            matrix(rows + [rows[0]], panel, protocol)
        for field, value in [('timing_model', 'MIO'), ('input_sha256', 'different')]:
            changed = copy.deepcopy(rows)
            changed[0][field] = value
            with self.assertRaises(ValueError):
                matrix(changed, panel, protocol)

    def test_gpu_attribution_excludes_unknown_allocation_and_changed_requests(self):
        rows, panel, protocol = fixture()
        row = next(r for r in rows if r['index'] == 2 and r['seed'] == 11 and r['variant'] == 'adaptive_batch_cpu')
        row['request_sequence_sha256'] = 'changed'
        result = summarize(rows, panel, protocol)
        self.assertEqual(result['gpu']['identical_sequence_shared_allocation_seed_pairs'], 1)
        for r in rows:
            r['slurm_job_id'] = None
        result = summarize(rows, panel, protocol)
        self.assertEqual(result['gpu']['identical_sequence_shared_allocation_seed_pairs'], 0)
        self.assertIsNone(result['gpu']['median_end_to_end_speed_identical_cuda_pairs'])
        self.assertEqual(result['protocol_assessment']['median_speed_at_least_1p5_vs_legacy'], 'INCOMPLETE')

    def test_equal_time_absence_never_acquires_a_noop_label(self):
        rows, panel, protocol = fixture()
        for r in rows:
            if r['index'] == 1 and r['seed'] == 11 and r['variant'] == 'adaptive_batch_auto':
                r['best_at_30s_ps'] = None
            if r['index'] == 1 and r['seed'] == 12 and r['variant'] == 'legacy_cpu':
                r['best_at_30s_ps'] = None
            if r['index'] == 2 and r['seed'] == 11 and r['variant'] in ('legacy_cpu', 'adaptive_batch_auto'):
                r['best_at_30s_ps'] = None
        result = summarize(rows, panel, protocol)['equal_time']['legacy_cpu'][0]
        self.assertEqual(result['outcomes'], {'control_only': 1, 'proposed_only': 1, 'neither_available': 1, 'loss': 1})
        self.assertEqual(result['expected_seed_pairs'], 4)

    def test_zero_reference_is_explicit_not_silently_omitted(self):
        self.assertEqual(ratio(0., 0.), 1.)
        self.assertEqual(ratio(0., 10.), 0.)
        self.assertIsNone(ratio(10., 0.))
        rows, panel, protocol = fixture()
        for r in rows:
            if r['variant'] == 'legacy_cpu':
                r['delay_ps'] = 0.
        result = summarize(rows, panel, protocol)
        self.assertEqual(result['comparisons']['legacy_cpu']['unbounded_delay_ratios'], 4)
        self.assertIsNone(result['comparisons']['legacy_cpu']['geomean_delay_ratio_seed_pairs'])
        self.assertEqual(result['protocol_assessment']['qor_noninferiority_vs_legacy'], 'NOT_MET_ON_THIS_PANEL')

    def test_gpu_speed_requires_same_computation_and_actual_eda_work(self):
        for field, value in [('checkpoint_sha256', 'different'), ('mapping_commands', 999),
                             ('joint_proposal_config', {'inner_updates': False})]:
            rows, panel, protocol = fixture()
            for row in rows:
                if row['variant'] == 'adaptive_batch_cpu' and row['index'] == 2:
                    row[field] = value
            result = summarize(rows, panel, protocol)
            self.assertEqual(result['gpu']['actual_cuda_seed_pairs'], 2)
            self.assertEqual(result['gpu']['identical_sequence_shared_allocation_seed_pairs'], 0)


if __name__ == '__main__':
    unittest.main()
