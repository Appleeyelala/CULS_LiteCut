import gzip
import json
from pathlib import Path
import tempfile
import unittest
from global_timing.saved_arrays import iter_saved_array


class SavedArrayTests(unittest.TestCase):
    def test_chunk_boundaries_and_archived_inputs(self):
        rows=[{'x':i,'text':'汉字[]\\"'+str(i)*13} for i in range(20)]
        text=json.dumps({'earlier':['a'*200],'scores':rows},indent=2,ensure_ascii=False)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'scores.json';p.write_text(text)
            for chunk in (1,7,64,65536):
                self.assertEqual(list(iter_saved_array(p,'scores',chunk)),rows)
            with gzip.open(str(p)+'.gz','wt') as f:f.write(text)
            p.unlink()
            self.assertEqual(list(iter_saved_array(p,'scores',11)),rows)

    def test_rejects_truncation_missing_field_and_nonobjects(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.json'
            for value in ['{\n  "scores": [{"x": 1}', '{\n  "scores": [1]}','{}']:
                p.write_text(value)
                with self.assertRaises(ValueError):list(iter_saved_array(p,'scores',7))


if __name__=='__main__':unittest.main()
