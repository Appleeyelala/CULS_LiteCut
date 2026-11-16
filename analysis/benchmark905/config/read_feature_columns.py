"""Read lossless columnar CSV features with standard Python only."""
import csv, hashlib, json, struct
from contextlib import ExitStack
from pathlib import Path

def iter_feature_vectors(bundle_root):
    root = Path(bundle_root)
    schema = json.loads((root / 'prediction/feature_vectors.schema.json').read_text())
    with ExitStack() as stack:
        readers = [csv.reader(stack.enter_context((root / c['file']).open(newline=''))) for c in schema['columns']]
        for reader, column in zip(readers, schema['columns']):
            assert next(reader) == [column['name']]
        count = 0
        for feature_id, rows in enumerate(zip(*readers, strict=True)):
            values = [float(row[0]) for row in rows]
            feature_hash = hashlib.sha256(struct.pack('<19f', *values)).hexdigest()
            yield feature_id, values, feature_hash
            count += 1
        assert count == schema['row_count']

def iter_terminal_records(bundle_root):
    # JSON references point directly to the already saved full real metrics.
    root = Path(bundle_root)
    with (root / 'benchmark/method_records.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            metrics = row['data'].get('metrics')
            if isinstance(metrics, dict) and set(metrics) == {'$bundle_ref'}:
                ref = metrics['$bundle_ref']
                assert ref['pointer'] == ''
                path = (root / ref['file']).resolve()
                assert path.is_relative_to(root.resolve())
                row['data']['metrics'] = json.loads(path.read_text())
            yield row
