import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('panel',ROOT/'scripts/joint_gradient_panel.py')
panel=importlib.util.module_from_spec(spec);spec.loader.exec_module(panel)


class ResumeTests(unittest.TestCase):
    def test_incomplete_mapping_is_not_rerun_or_promoted_to_pass(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);source=root/'source';source.mkdir();out=root/'case';out.mkdir();(out/'batch_replay').mkdir()
            (source/'input.aig').write_bytes(b'fixture');gpuls=root/'gpuls';gpuls.write_bytes(b'fixture')
            panel.save(source/'context.json',{'tools':{'abc':{'path':'abc'},'library':{'path':'lib'}}})
            combinations=[{'choices':[(mask>>i)&1 for i in range(4)]} for mask in range(16)]
            panel.save(out/'frozen_predictions.json',{'source':str(source),'input_sha256':panel.sha(source/'input.aig'),'combinations':combinations})
            panel.save(out/'baseline.json',{'area':10.})
            for mask in range(1,16):
                folder=out/'measurements'/f'mask_{mask:02d}'
                panel.save(folder/'rewrite/replay.json',{'choices':combinations[mask]['choices'],'status':'PASS'})
                if mask==1:
                    (folder/'evaluation').mkdir();(folder/'evaluation/mapping.log').write_text('truncated')
                else:panel.save(folder/'observation.json',{'mask':mask,'status':'PASS'})
            with patch('global_timing.evaluation.Evaluator') as evaluator:
                args=SimpleNamespace(output=out,gpuls=gpuls,circuit=1)
                panel.resume(args);panel.resume(args)
                evaluator.return_value.evaluate.assert_not_called()
            result=json.loads((out/'measurements.json').read_text())
            self.assertEqual(result['results'][1]['status'],'INCOMPLETE_RECORD')
            self.assertEqual(result['new_mapping_calls'],1)
            self.assertEqual(result['mapping_calls_with_missing_metadata'],1)


if __name__=='__main__':unittest.main()
