#!/usr/bin/env python3
"""Losslessly compact an already collected formal-905 bundle; no EDA or ML."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import zipfile


ARCHIVE = 'MFFC_GPU_GRADIENT_905_ANALYSIS_BUNDLE.zip'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def dump(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


FEATURE_READER = '''"""Read lossless columnar CSV features with standard Python only."""
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
'''


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', required=True, type=Path)
    ap.add_argument('--input-run', required=True, type=Path)
    args = ap.parse_args()
    run, old_run = args.run.resolve(), args.input_run.resolve()
    if not os.environ.get('SLURM_JOB_ID') or not all(p.is_relative_to('/data/wushangge') for p in [run, old_run]):
        ap.error('Use a new /data Slurm workspace.')
    previous = json.loads((old_run / 'output/packaging_completion.json').read_text())
    assert previous['status'] == 'VERIFIED_ARCHIVE' and previous['zip_crc_and_all_member_sha256_checked']
    assert sha(Path(previous['path'])) == previous['sha256']
    old = old_run / 'output/bundle'
    stage = run / 'output/bundle'
    stage.mkdir(exist_ok=False)
    origins = {j['path']: j for j in map(json.loads, (old / 'audit/FILE_MANIFEST.jsonl').open())}
    omitted = {'README.md', 'prediction/feature_vectors.csv', 'benchmark/method_records.jsonl',
               'audit/FILE_MANIFEST.jsonl', 'audit/CHECKSUMS.sha256'}
    for p in sorted(old.rglob('*')):
        if not p.is_file():
            continue
        name = str(p.relative_to(old))
        if name in omitted:
            continue
        dest = stage / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        with p.open('rb') as src, dest.open('xb') as dst:
            shutil.copyfileobj(src, dst)
    def save(name, data):
        path = stage / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x') as f:
            f.write(dump(data) + '\n')

    coverage = json.loads((stage / 'audit/coverage.json').read_text())
    counts = coverage['counts']
    assert coverage['source_commit'] == 'ecc85e6ac82571caf9f1a6004098dbe976536256'
    columns = []
    output_streams = []
    matrix = old / 'prediction/feature_vectors.csv'
    with matrix.open(newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header[:2] == ['feature_id', 'feature_hash'] and len(header) == 21
        for i, name in enumerate(header[2:]):
            relative = f'prediction/feature_columns/{i:02d}_{name}.csv'
            path = stage / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            stream = path.open('x')
            stream.write(name + '\n')
            output_streams.append(stream)
            columns.append({'name': name, 'file': relative})
        n = 0
        for row in reader:
            assert int(row[0]) == n and len(row) == 21
            values = row[2:]
            assert hashlib.sha256(struct.pack('<19f', *map(float, values))).hexdigest() == row[1]
            for stream, value in zip(output_streams, values):
                assert value and '\n' not in value and ',' not in value
                stream.write(value + '\n')
            n += 1
            if n % 1000000 == 0:
                print(dump({'phase': 'columnar_features', 'rows': n}), flush=True)
    for stream in output_streams:
        stream.close()
    assert n == counts['unique_feature_vectors']
    save('prediction/feature_vectors.schema.json', {'row_count': n, 'columns': columns,
        'feature_id': 'Zero-based data-row index, identical for every column file and unchanged from litecut_scores.csv.',
        'values': 'Exact original CSV numeric strings, no rounding/re-normalization; represent original float32 values.',
        'feature_hash': "hashlib.sha256(struct.pack('<19f', *values)).hexdigest(); validated against every original feature_hash before omission of the redundant hash column.",
        'original_table_sha256': sha(matrix), 'all_original_feature_hashes_verified': n})

    terminal_count = 0
    metrics_references = 0
    with (old / 'benchmark/method_records.jsonl').open() as src, (stage / 'benchmark/method_records.jsonl').open('x') as dst:
        for line in src:
            row = json.loads(line)
            data = row['data']
            if isinstance(data.get('metrics'), dict):
                i, method = row['circuit_id'], row['method']
                if method == 'Ours':
                    summary = json.loads((stage / f'optimization/cases/{i:04d}/work/summary.json').read_text())
                    target = summary['selected']['metrics']['$bundle_ref']
                    assert target['pointer'] == ''
                else:
                    target = {'file': f'benchmark/cases/{i:04d}_{method}/evaluation/result.json', 'pointer': ''}
                assert json.loads((stage / target['file']).read_text()) == data['metrics']
                data['metrics'] = {'$bundle_ref': target}
                metrics_references += 1
            dst.write(dump(row) + '\n')
            terminal_count += 1
    assert terminal_count == 2715 and metrics_references == 2711
    with (stage / 'config/read_feature_columns.py').open('x') as f:
        f.write(FEATURE_READER)
    with Path(__file__).open('rb') as src, (stage / 'config/compact_mffc905_analysis_bundle.py').open('xb') as dst:
        shutil.copyfileobj(src, dst)
    record = {'scope': 'Lossless packaging-format compaction only; no collection rerun or EDA/ML calls.',
        'input_packaging_run': str(old_run), 'input_archive_sha256': previous['sha256'],
        'input_archive_size_bytes': previous['size_bytes'], 'source_commit': coverage['source_commit'],
        'compaction_source_commit': json.loads((run / 'run.json').read_text())['source_commit'],
        'all_feature_rows_and_hashes_verified': n, 'terminal_records': terminal_count,
        'duplicated_terminal_metrics_replaced_by_verified_references': metrics_references,
        'coverage_unchanged': True, 'original_source_files_untouched': True,
        'rules': ['Split feature vectors into 19 exact-text CSV columns; row order preserves feature_id.',
                  'The removed random hash string is deterministically reconstructible from the 19 original float32 values, verified on every row.',
                  'Terminal-record metrics are exact references to already included real metrics, verified before replacement.',
                  'All other members except README/manifests are copied without byte changes.']}
    save('audit/compaction_provenance.json', record)
    readme = (old / 'README.md').read_text()
    old_text = '`feature_id`连接feature_vectors.csv，精确保存原19维数值和feature_hash；相同hash的数值逐项核对后才去重。'
    new_text = '`feature_id`为feature_columns各列CSV中相同的0基数据行号；19维原始数值文本原样保留。feature_hash由这一行19个float32的小端字节SHA256精确重算，全部4,998,334行均已与原hash核对通过；无需重复保存可推导的哈希字符串。'
    assert old_text in readme
    readme = readme.replace(old_text, new_text)
    readme += '''
## 无损列式特征与终态记录读取

`prediction/feature_vectors.schema.json`列出19个单列CSV；同一数据行就是同一feature_id，表头不计入行号。所有数值文本及行顺序逐项保留，未削减特征、候选或请求。读取例子：

```python
import sys
sys.path.insert(0, '/解压目录/config')
from read_feature_columns import iter_feature_vectors, iter_terminal_records
# 按需流式读取，避免一次载入全部499.8万组向量。
for feature_id, values_19, original_feature_hash in iter_feature_vectors('/解压目录'):
    pass
```

`benchmark/method_records.jsonl`中成功终态的data.metrics也采用同样的$bundle_ref，指向已包含的真实测量JSON。`iter_terminal_records`可还原原始对象；2711份替换前的metrics均逐字段核对相等。原始记录所有其他字段不变，包括失败、状态、耗时和身份。其余summary/observation引用仍由bundle_reader读取。

压缩整理只是编码变化；统计口径与coverage.json逐字节保持不变。第一版较大ZIP及其原始整理目录仍保留，没有覆盖或删除。转换来源、逐行核验与版本见audit/compaction_provenance.json。
'''
    with (stage / 'README.md').open('x') as f:
        f.write(readme)
    changed = {'README.md', 'benchmark/method_records.jsonl'}
    with (stage / 'audit/FILE_MANIFEST.jsonl').open('x') as f:
        for p in sorted(stage.rglob('*')):
            if not p.is_file() or p.name in ['FILE_MANIFEST.jsonl', 'CHECKSUMS.sha256']:
                continue
            name = str(p.relative_to(stage))
            digest = sha(p)
            original = origins.get(name)
            if original and name not in changed:
                assert digest == original['sha256'], name
            entry = {'path': name, 'sha256': digest, 'size_bytes': p.stat().st_size,
                     'original_extraction': original, 'compaction_input': str(old / name),
                     'rule': 'See compaction_provenance.json; unchanged members are byte-identical to verified first collection.'}
            f.write(dump(entry) + '\n')
    checksums = {}
    for p in sorted(stage.rglob('*')):
        if p.is_file() and p.name != 'CHECKSUMS.sha256':
            checksums[str(p.relative_to(stage))] = sha(p)
    with (stage / 'audit/CHECKSUMS.sha256').open('x') as f:
        f.write(''.join(h + '  ' + name + '\n' for name, h in checksums.items()))
    archive = run / 'output' / ARCHIVE
    files = [p for p in sorted(stage.rglob('*')) if p.is_file()]
    print(dump({'phase': 'zip_compact', 'feature_rows': n, 'members': len(files)}), flush=True)
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=9, allowZip64=True) as z:
        for p in files:
            z.write(p, str(p.relative_to(stage)))
    print(dump({'phase': 'verify_compact', 'size_bytes': archive.stat().st_size}), flush=True)
    with zipfile.ZipFile(archive) as z:
        assert len(z.namelist()) == len(set(z.namelist())) == len(files)
        for info in z.infolist():
            h = hashlib.sha256()
            with z.open(info) as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b''):
                    h.update(chunk)
            if info.filename in checksums:
                assert h.hexdigest() == checksums[info.filename], info.filename
        assert z.testzip() is None
        assert json.loads(z.read('audit/coverage.json')) == coverage
    result = {'status': 'VERIFIED_ARCHIVE', 'path': str(archive), 'size_bytes': archive.stat().st_size,
        'size_MiB': archive.stat().st_size / 1024**2, 'sha256': sha(archive), 'members': len(files),
        'uncompressed_bytes': sum(p.stat().st_size for p in files), 'counts': coverage,
        'zip_crc_and_all_member_sha256_checked': True, 'created_at': datetime.now(timezone.utc).isoformat(),
        'slurm_job_id': os.environ['SLURM_JOB_ID'], 'compaction': record, 'new_experiment_or_model_calls': 0}
    for name in [ARCHIVE + '.metadata.json', 'packaging_completion.json']:
        with (run / 'output' / name).open('x') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            f.write('\n')
    with Path(str(archive) + '.sha256').open('x') as f:
        f.write(result['sha256'] + '  ' + archive.name + '\n')
    with (run / 'output/README.md').open('x') as f:
        f.write('# 已核验ZIP交付\n\n' + f'ZIP：`{archive}`\n\n大小：{result["size_bytes"]} bytes（{result["size_MiB"]:.2f} MiB）。\n\nSHA256：`{result["sha256"]}`\n\n' + readme)
    print(dump({k: result[k] for k in ['status', 'path', 'size_bytes', 'sha256', 'members']}), flush=True)


if __name__ == '__main__':
    main()
