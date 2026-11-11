"""Shared no-op observations retain physical artifacts under their first run."""
import json
from pathlib import Path
import tempfile
import unittest

from scripts.analyze_timing_paths import evaluation_directory


class PathArtifactTests(unittest.TestCase):
    def test_reused_baseline_resolves_record_provenance_at_variable_depth(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, relative in [('first', 'rewrite/raw/aigs/choice.aig'),
                                   ('second', 'rewrite/aigs/choice.aig')]:
                work = root / name
                aig = work / relative
                aig.parent.mkdir(parents=True)
                aig.write_bytes(b'test-only artifact identity')
                (work / 'evaluation').mkdir()
                (work / 'observation.json').write_text(json.dumps({'status': 'PASS', 'aig': str(aig)}))
                (work / 'evaluation/result.json').write_text(json.dumps({'input_sha256': 'fixture_hash'}))
                self.assertEqual(evaluation_directory(aig, root, 'fixture_hash'), work / 'evaluation')
                with self.assertRaises(ValueError):
                    evaluation_directory(aig, root, 'wrong_hash')
