"""Delivery semantics for partial installations, missing clocks and grouping."""
import unittest

from global_timing.mffc_delivery import installation_details, named_tradeoffs, shape_summary, tool_cost_summary


class DeliveryTests(unittest.TestCase):
    def test_installation_pass_is_not_a_passed_quality_measurement(self):
        requests = [
            {'replay': {'status': 'PASS', 'requested': [], 'applied_rewrites': 0},
             'observation': {'status': 'PASS', 'runtime_s': 1}},
            {'replay': {'status': 'PASS', 'requested': [[10, 1, 2]], 'applied_rewrites': 1},
             'observation': {'status': 'FAIL', 'runtime_s': 2}},
            {'replay': {'status': 'FAIL', 'requested': [[10, 1, 2], [20, 1, 0]], 'applied_rewrites': 1,
                        'rejected_requests': [{'status': 'COMPROMISED'}]},
             'observation': {'status': 'FAIL', 'runtime_s': 3}},
            {'replay': None, 'observation': None},
        ]
        data = installation_details(requests)
        self.assertEqual(data['installation_states'], {'PASS': 2, 'FAIL': 1, 'UNKNOWN': 1})
        self.assertEqual(data['nonempty_complete_installations'], 1)
        self.assertEqual(data['accepted_rewrites_in_failed_partial_installations'], 1)
        self.assertEqual(data['rejection_statuses'], {'COMPROMISED': 1})
        self.assertEqual(data['failed_request_inclusive_s']['sum'], 5)
        self.assertEqual(data['missing_terminal_observations'], 1)
        with self.assertRaises(ValueError):
            installation_details([{'replay': {'status': 'PASS', 'requested': [[10, 1, 2]], 'applied_rewrites': 0}}])

    def test_missing_repair_clock_is_not_zero(self):
        data = installation_details([{'observation': {'status': 'FAIL', 'runtime_s': None,
                                     'prediction_before_evaluation': {'source': 'explicit_feasibility_repair'}}}])
        self.assertEqual(data['repair_request_inclusive_s']['n'], 0)
        self.assertIsNone(data['repair_request_inclusive_s']['sum'])

    def test_tradeoff_names_preserve_frozen_grid_and_overlap(self):
        result = named_tradeoffs({'better_delay__better_time': 2, 'better_delay__tie_time': 3,
                                 'tie_delay__better_time': 4, 'worse_delay__better_time': 5,
                                 'better_delay__worse_time': 6})
        self.assertEqual(list(result.values()), [5, 6, 5, 6])

    def test_only_actual_cuda_work_enters_cuda_shape_groups(self):
        rows = [
            {'method': 'Ours', 'status': 'PASS', 'device': 'cuda', 'native_forward_calls': 3,
             'regions': 100, 'region_dag_layers': 40, 'max_layer_width': 2, 'native_kernel_ms': 5.},
            {'method': 'Ours', 'status': 'PASS', 'device': 'none', 'regions': 10, 'decisions': 0},
            {'method': 'Ours', 'status': 'FAIL', 'regions': 10},
        ]
        result = shape_summary(rows)
        self.assertEqual(result['observed_cuda_method_points'], 1)
        self.assertEqual(result['devices']['unknown']['method_points'], 1)
        self.assertEqual(result['devices']['none']['recorded']['native_kernel_ms']['n'], 0)
        self.assertEqual(result['cuda_shape_bands']['region_dag_layers']['32_512']['method_points'], 1)

    def test_tool_costs_include_failed_methods_and_keep_missing_clocks_unknown(self):
        rows = [
            {'method': 'Ours', 'status': 'PASS', 'end_to_end_runtime_s': 10,
             'mapping_commands': 2, 'cec_commands': 3,
             'tool_seconds_inclusive': {'mapping': 4., 'cec': 1.}},
            {'method': 'Ours', 'status': 'FAIL', 'end_to_end_runtime_s': 20,
             'mapping_commands': 1, 'cec_commands': 2,
             'tool_seconds_inclusive': {'mapping': 7., 'cec': 2.}},
            {'method': 'Ours', 'status': 'TIMEOUT', 'end_to_end_runtime_s': None,
             'mapping_commands': 1, 'unreadable_partial_command_files': ['unfinished.json']},
        ]
        data = tool_cost_summary(rows)['methods']
        mapping = data['Ours']['tools']['mapping']
        self.assertEqual(mapping['recorded_command_counts']['sum'], 4)
        self.assertEqual(mapping['per_point_inclusive_s'], {'n': 2, 'median': 5.5, 'p90': 6.7, 'sum': 11.})
        self.assertEqual(mapping['points_with_recorded_commands_but_no_clock'], 1)
        self.assertEqual(data['Ours']['cold_wall_s']['sum'], 30)
        self.assertEqual(data['Ours']['unreadable_command_files'], 1)
        self.assertIsNone(data['A']['tools']['mapping']['per_point_inclusive_s']['sum'])


if __name__ == '__main__':
    unittest.main()
