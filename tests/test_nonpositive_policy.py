"""Research Mapping must retain a verified worsening candidate, never a fallback."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from global_timing.mffc_ste_experiment import SteExperiment, SteSearchConfig, NonpositiveSteConfig


class NonpositivePolicyTest(unittest.TestCase):
    def run_fake_observation(self, config, status='PASS', area=100):
        with tempfile.TemporaryDirectory() as directory:
            x = SteExperiment.__new__(SteExperiment)
            x.output = Path(directory)
            x.config, x.aig, x.noop = config, x.output/'original.aig', [0]
            x.original = {'area':100, 'delay_ps':10, 'cec':'PASS'}
            x.model = object()
            x.feasibility = SimpleNamespace(validate=lambda choices: None)
            x.partition = SimpleNamespace(regions=[0])
            x.catalogue = SimpleNamespace(decisions=[{'decision_owner_region':0,'candidate_count':1}])
            x.backend, x.structure_meta, x.native_profiles, x.stages = {}, {}, [], []
            best = {'choices':[1]}
            trace = {'steps':40,'logits_total_change_norm':1,'distinct_hard_combinations_including_noop':2}
            final = {'status':status,'choices':[1]}
            if status=='PASS':
                final.update(metrics={'area':area,'delay_ps':11,'cec':'PASS'},aig='candidate.aig',applied_rewrites=1)
            def stage(name, function, *args, **kwargs):
                return (best, trace) if name=='ste_adam_proxy_optimization' else final
            x.stage = stage
            for name in ['original','final']:
                folder=x.output/name;folder.mkdir()
                (folder/'mapping.command.json').write_text('{}')
            return x.run()

    def test_worse_delay_is_retained(self):
        r=self.run_fake_observation(NonpositiveSteConfig())
        self.assertEqual(r['status'],'PASS')
        self.assertEqual(r['selected']['metrics']['delay_ps'],11)
        self.assertEqual(r['selected']['choices'],[1])

    def test_failed_candidate_has_no_substitute_metrics(self):
        r=self.run_fake_observation(NonpositiveSteConfig(),'FAIL')
        self.assertEqual(r['status'],'FAIL')
        self.assertIsNone(r['selected']['metrics'])

    def test_area_limit_is_still_reported(self):
        r=self.run_fake_observation(NonpositiveSteConfig(),area=106)
        self.assertEqual(r['status'],'AREA_LIMIT_EXCEEDED')
        self.assertEqual(r['final_candidate']['metrics']['delay_ps'],11)

    def test_legacy_delay_fallback_unchanged(self):
        r=self.run_fake_observation(SteSearchConfig())
        self.assertEqual(r['selected']['metrics']['delay_ps'],10)
        self.assertEqual(r['selected']['source'],'original_fallback')

    def test_fixed_optimizer_settings(self):
        SteSearchConfig().validate()
        NonpositiveSteConfig().validate()
        for cfg in (SteSearchConfig(),NonpositiveSteConfig()):
            with self.assertRaises(ValueError):
                replace(cfg,learning_rate=.13).validate()

if __name__=='__main__':
    unittest.main()
