"""Read logical CSVs from the GitHub publication without materializing large files."""
import csv, hashlib, json, struct
from pathlib import Path

def iter_lines(root, logical_path, circuit_id=None):
    root = Path(root)
    manifest = json.loads((root / 'github/sharding_manifest.json').read_text())
    table = manifest['tables'].get(logical_path)
    if table is None:
        with (root / logical_path).open(newline='') as stream:
            yield from stream
        return
    yield table['header']
    for part in table['parts']:
        if circuit_id is not None and table['by_circuit'] and part['circuit_id'] != circuit_id:
            continue
        with (root / part['file']).open(newline='') as stream:
            assert next(stream) == table['header']
            yield from stream

def iter_csv(root, logical_path, circuit_id=None):
    yield from csv.DictReader(iter_lines(root, logical_path, circuit_id))

def iter_feature_vectors(root):
    root = Path(root)
    schema = json.loads((root / 'prediction/feature_vectors.schema.json').read_text())
    readers = [iter_csv(root, column['file']) for column in schema['columns']]
    count = 0
    for feature_id, rows in enumerate(zip(*readers, strict=True)):
        values = [float(row[column['name']]) for row, column in zip(rows, schema['columns'])]
        yield feature_id, values, hashlib.sha256(struct.pack('<19f', *values)).hexdigest()
        count += 1
    assert count == schema['row_count']
