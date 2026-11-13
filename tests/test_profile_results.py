"""Analytical audit/timestamp fixtures, never real mapping-delay evidence."""
import json
from pathlib import Path
import tempfile
import unittest

from global_timing.candidates import Catalogue
from global_timing.common import digest
from global_timing.profile_results import add_runtime_metrics, audit_measurement


class ProfileResultsTests(unittest.TestCase):
    def test_time_cutoffs_exclude_late_or_infeasible_points_and_preserve_missing(self):
        def row(method, delay, area, points, baseline=None):
            return {'index': 9, 'seed': 7, 'variant': method, 'status': 'PASS', 'input_sha256': 'input',
                    'abc_sha256': 'abc', 'library_sha256': 'lib', 'domain_sha256': 'domain' if baseline else None,
                    'delay_ps': delay, 'area': area, 'points': points, 'baseline': baseline}
        def point(t, delay, area=100.):
            return {'completed_s': t, 'delay_ps': delay, 'area': area}
        rows = [row('native_abc',90.,100.,[point(2,90)]),
                row('adaptive_batch_auto',88.,100.,[point(1,100),point(29,95),point(31,89),point(200,88),point(10,50,120)],
                    {'delay_ps':100.,'area':100.}),
                {'index':9,'seed':7,'variant':'failed','status':'FAIL','points':[]}]
        result = add_runtime_metrics(rows)
        self.assertAlmostEqual(result[1]['quality_target_ps'],90.9)
        self.assertEqual(result[1]['time_to_quality_s'],31)
        self.assertEqual([result[1][f'best_at_{t}s_ps'] for t in (30,120,600)],[95,89,88])
        self.assertIsNone(result[2]['time_to_quality_s'])
        self.assertIsNone(result[2]['best_at_600s_ps'])
        self.assertFalse(result[2]['area_feasible_1p05'])

    def test_timing_model_or_candidate_domain_mismatch_is_rejected(self):
        base = {'index':1,'seed':1,'status':'PASS','variant':'a','points':[],
                'input_sha256':'i','abc_sha256':'a','library_sha256':'l','domain_sha256':'d',
                'baseline':{'delay_ps':1.,'area':1.},'delay_ps':1.,'area':1.}
        for field in ('input_sha256','abc_sha256','library_sha256','domain_sha256'):
            with self.subTest(field=field), self.assertRaises(ValueError):
                add_runtime_metrics([dict(base),dict(base, **{field:'DIFFERENT'})])

    def test_raw_audit_rejects_partial_installation_cec_failure_and_changed_measurement(self):
        with tempfile.TemporaryDirectory() as directory:
            work=Path(directory)
            candidate=work/'rewrite/raw/aigs/choice.aig'
            candidate.parent.mkdir(parents=True)
            candidate.write_text('analytical AIG byte fixture; never executed')
            evaluation=work/'evaluation';evaluation.mkdir()
            original,abc,library=[work/n for n in ('original.aig','abc','lib')]
            for p in (original,abc,library):p.write_text(p.name)
            netlist=evaluation/'mapped.v';netlist.write_text('analytical netlist fixture')
            provenance={k:{'path':str(p),'sha256':digest(p)} for k,p in [('original',original),('abc',abc),('library',library)]}
            metrics={'delay_ps':100.,'area':1.,'po_arrivals_ps':[100.]}
            measured=dict(metrics,status='PASS',cec='PASS',timing_model='ABC_SCL_LIBERTY',
                input_sha256=digest(candidate),mapped_sha256=digest(netlist),provenance=provenance)
            (evaluation/'result.json').write_text(json.dumps(measured))
            (evaluation/'cec.log').write_text('Networks are equivalent\n')
            mapping='GT_META 1 100 1 1\nGT_PO 0 100 90\n'
            (evaluation/'mapping.log').write_text(mapping)
            argv=[str(abc),'-c',f'read_lib {library}; read_aiger {candidate}; map; topo; stime -p; write_verilog {netlist}']
            (evaluation/'mapping.command.json').write_text(json.dumps({'argv':argv}))
            (evaluation/'cec.command.json').write_text(json.dumps({'argv':[str(abc),'-c',f'cec -n {original} {candidate}']}))
            trace=work/'rewrite/raw/traces/choice.jsonl';trace.parent.mkdir()
            selected={'root':10,'cut_slot':2,'selected_subgraph':5,'preselected':True,'final_status':'ACCEPTED'}
            trace.write_text(json.dumps(selected)+'\n')
            domain=Catalogue([{'root':10,'cut_slot':2,'candidate_count':1,'candidate_subgraph_indices':[5],
                               'decision_owner_region':2}],[],{2:[0]})
            observation={'status':'PASS','choices':[1],'metrics':metrics,'aig':str(candidate)}
            self.assertEqual(audit_measurement(observation,work,domain,{})['accepted_rewrites'],1)
            trace.write_text(json.dumps(dict(selected,final_status='REVERTED'))+'\n')
            with self.assertRaises(ValueError):audit_measurement(observation,work,domain,{})
            trace.write_text(json.dumps(selected)+'\n')
            (evaluation/'cec.log').write_text('Networks are NOT equivalent\n')
            with self.assertRaises(ValueError):audit_measurement(observation,work,domain,{})
            (evaluation/'cec.log').write_text('Networks are equivalent\n')
            (evaluation/'mapping.log').write_text(mapping.replace('100','101'))
            with self.assertRaises(ValueError):audit_measurement(observation,work,domain,{})


if __name__ == '__main__':
    unittest.main()
