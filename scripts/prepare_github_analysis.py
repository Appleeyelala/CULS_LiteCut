#!/usr/bin/env python3
"""Stage existing formal-905 evidence as readable Git files, without EDA/ML calls."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


FROZEN = 'ecc85e6ac82571caf9f1a6004098dbe976536256'
LIMIT = 4 * 1024 * 1024


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def shard_csv(source, relative, target, expected):
    parts = []
    counts = Counter()
    original_hash = hashlib.sha256()
    current = None
    current_group = None
    original_bytes = 0
    rows = 0
    with source.open('rb') as incoming:
        header = incoming.readline()
        assert header.endswith(b'\n')
        by_circuit = header.startswith(b'circuit_id,')
        original_hash.update(header)
        original_bytes += len(header)

        def close_part():
            if current is not None:
                current['stream'].close()
                path = current.pop('path')
                current.pop('stream')
                current['sha256'] = digest(path)
                current['bytes'] = path.stat().st_size
                parts.append(current.copy())

        for line in incoming:
            assert line.endswith(b'\n'), relative
            group = int(line.partition(b',')[0]) if by_circuit else None
            if current is None or group != current_group or current['bytes'] + len(line) > LIMIT:
                close_part()
                suffix = f'{group:04d}/' if by_circuit else ''
                name = f'{Path(relative).with_suffix("")}/{suffix}part-{counts[group]:04d}.csv'
                counts[group] += 1
                destination = target / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                stream = destination.open('xb')
                stream.write(header)
                current = {'file': name, 'path': destination, 'stream': stream,
                           'first_row': rows, 'rows': 0, 'bytes': len(header),
                           'circuit_id': group}
                current_group = group
            current['stream'].write(line)
            current['bytes'] += len(line)
            current['rows'] += 1
            original_hash.update(line)
            original_bytes += len(line)
            rows += 1
            if rows % 2000000 == 0:
                print(json.dumps({'phase': 'shard', 'file': relative, 'rows': rows}), flush=True)
        close_part()
    assert original_hash.hexdigest() == expected, relative
    assert original_bytes == source.stat().st_size
    # Independently reconstruct each original byte stream from the saved shards.
    reconstructed = hashlib.sha256(header)
    verified_rows = 0
    for part in parts:
        with (target / part['file']).open('rb') as stream:
            assert stream.readline() == header
            for line in stream:
                reconstructed.update(line)
                verified_rows += 1
    assert reconstructed.hexdigest() == expected and verified_rows == rows
    return {'original_sha256': expected, 'original_bytes': original_bytes,
            'rows': rows, 'header': header.decode(), 'by_circuit': by_circuit,
            'parts': parts, 'roundtrip_sha256_verified': True}


READER = '''"""Read logical CSVs from the GitHub publication without materializing large files."""
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
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    args = parser.parse_args()
    run, bundle = args.run.resolve(), args.bundle.resolve()
    assert os.environ.get('SLURM_JOB_ID')
    assert all(p.is_relative_to('/data/wushangge/CULS_LiteCut') for p in [run, bundle])
    metadata = json.loads((run / 'run.json').read_text())
    coverage = json.loads((bundle / 'audit/coverage.json').read_text())
    assert coverage['source_commit'] == FROZEN
    assert coverage['counts']['designs'] == 905 and not coverage['unexpected_missing']
    source = metadata['source_repository']
    repo = run / 'output/repository'
    subprocess.run(['git', 'clone', '--no-hardlinks', '--no-checkout', source, str(repo)], check=True)
    subprocess.run(['git', '-C', str(repo), 'checkout', '-b', 'github-delivery', metadata['source_commit']], check=True)
    for key in ['user.name', 'user.email']:
        value = subprocess.check_output(['git', '-C', source, 'config', key], text=True).strip()
        subprocess.run(['git', '-C', str(repo), 'config', key, value], check=True)
    target = repo / 'analysis/benchmark905'
    target.mkdir(parents=True, exist_ok=False)
    expected = {}
    for line in (bundle / 'audit/CHECKSUMS.sha256').read_text().splitlines():
        checksum, name = line.split('  ', 1)
        expected[name] = checksum
    tables = {}
    copied = 0
    for source_file in sorted(bundle.rglob('*')):
        if not source_file.is_file():
            continue
        assert not source_file.is_symlink()
        relative = source_file.relative_to(bundle).as_posix()
        if source_file.suffix == '.csv' and source_file.stat().st_size > LIMIT:
            tables[relative] = shard_csv(source_file, relative, target, expected[relative])
            print(json.dumps({'phase': 'sharded', 'file': relative, 'rows': tables[relative]['rows'],
                              'parts': len(tables[relative]['parts'])}), flush=True)
        else:
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source_file.open('rb') as incoming, destination.open('xb') as outgoing:
                shutil.copyfileobj(incoming, outgoing)
            original_digest = expected[relative] if relative in expected else digest(source_file)
            assert digest(destination) == original_digest, relative
            copied += 1
    assert tables['prediction/litecut_scores.csv']['rows'] == coverage['counts']['litecut_score_rows']
    assert tables['optimization/decisions.csv']['rows'] == coverage['counts']['decisions']
    for name, table in tables.items():
        if name.startswith('prediction/feature_columns/'):
            assert table['rows'] == coverage['counts']['unique_feature_vectors']
    github = target / 'github'
    github.mkdir()
    manifest = {'created_at': datetime.now(timezone.utc).isoformat(),
                'experiment_source_commit': FROZEN, 'publication_source_commit': metadata['source_commit'],
                'original_bundle': str(bundle), 'source_files_untouched': True,
                'slurm_job_id': os.environ['SLURM_JOB_ID'], 'max_csv_part_bytes': LIMIT,
                'rule': 'All original CSV data lines retained in order. Repeat the exact header in every shard. '
                        'Concatenating the first header and all shard data lines restores original bytes and SHA256. '
                        'Circuit tables split at circuit boundaries and size; feature_id remains global row number. '
                        'Other original bundle files are byte-identical. Original audit manifests describe logical files.',
                'tables': tables}
    write_json(github / 'sharding_manifest.json', manifest)
    (github / 'read_data.py').write_text(READER)
    readme = '''# 最新正式 905 电路数据：GitHub 阅读入口

本目录来自已核验的 905 分析资料包；正式算法冻结在 `ecc85e6ac82571caf9f1a6004098dbe976536256`。
此次只改变文件组织，没有运行 CULS、ABC、CEC、LiteCut、优化器或新实验。
原资料包说明保留在 [README.md](README.md)，完整项目入口见 [../../docs/GITHUB_DATA_INDEX.md](../../docs/GITHUB_DATA_INDEX.md)。

| 目录/文件 | 内容 |
|---|---|
| `benchmark/BENCHMARK905_RESULTS.csv` | 905×A/B/Ours=2715 条终态，含全部失败；901 电路三方法均成功 |
| `benchmark/BENCHMARK905_SUMMARY.json` | 正式汇总、统计口径及质量/耗时结论 |
| `optimization/circuit_index.csv` | 电路 ID、候选和各文件路径索引 |
| `optimization/cases/0000/work/summary.json` | 每个电路完整搜索摘要；0000 至 0904 |
| `optimization/cases/0000/work/optimization/` | 实际请求 observation、预测、梯度记录、校准及真实测量 |
| `optimization/decisions/0000/part-0000.csv` | 按 circuit_id 分片的全部 4,296,273 decisions |
| `prediction/global_request_pairs.csv` | 3041 请求，3039 成功/2 失败；2134 有效预测—真实 Global Delay 配对 |
| `prediction/litecut_scores/0000/part-0000.csv` | 按 circuit_id 分片的全部 12,698,529 条局部分数记录 |
| `prediction/feature_columns/` | 全部 4,998,334 组精确 19 维特征的列分片 |
| `runtime/` | 端到端、阶段、工具调用和已有 GPU profile |
| `config/` | 实验协议、环境、实现说明及原有读取程序 |
| `audit/` | 原始核验、失败、缺失、来源及校验信息 |
| `github/` | 本次分片映射、读取程序及 GitHub 文件 SHA256 |

**计量边界**：LiteCut local abc_delay、优化代理预测与真实 `ABC_SCL_LIBERTY` delay 分开保存。
局部候选的真实 abc_delay ground truth 未记录，有效局部准确率配对为 0；不能用全局 delay 冒充局部标签。
GPU 记录不能单独证明整套算法相对 CPU 的因果加速；详见原始说明和 runtime/timing_semantics.json。

## CSV 分片与完整性

大于 4 MiB 的 CSV 拆为可直接阅读的带表头分片；不使用 Git LFS 指针或压缩文件。
`github/sharding_manifest.json` 给出每个逻辑原文件的 SHA256、全部分片顺序、全局行号范围和 circuit_id。
全部分片按顺序合并并去掉重复表头，已逐文件核验与原始 SHA256 完全一致。
feature_id 为每列拼接后的 0 基数据行号；分片 first_row 保持这一编号，不能在每片重新编号。
summary/observation 中 `$bundle_ref.file` 仍相对于本目录；JSON 引用目标均原样保留。
原 `audit/CHECKSUMS.sha256`、`FILE_MANIFEST.jsonl` 描述整理前的逻辑文件；本次物理文件清单见 `github/CHECKSUMS.sha256`。
原 `config/read_feature_columns.py` 适用于重建后的逻辑 CSV；直接读本仓库请用新增读取程序：

```python
import sys
sys.path.insert(0, 'analysis/benchmark905/github')
from read_data import iter_csv, iter_feature_vectors
root = 'analysis/benchmark905'
for row in iter_csv(root, 'prediction/litecut_scores.csv', circuit_id=0):
    print(row)
```

源码、汇总表与单个请求可以按文件读取。数百万条记录的全量统计仍需流式程序处理，不能假设一次聊天已读完全部数据。
服务器原数据和 ZIP 均保留，既有结果未覆盖或删除。
'''
    (target / 'README_GITHUB.md').write_text(readme)
    with (github / 'CHECKSUMS.sha256').open('x') as stream:
        for path in sorted(target.rglob('*')):
            if path.is_file() and path != github / 'CHECKSUMS.sha256':
                stream.write(digest(path) + '  ' + path.relative_to(target).as_posix() + '\n')
    largest = max((p.stat().st_size, str(p.relative_to(repo))) for p in repo.rglob('*')
                  if p.is_file() and '.git' not in p.relative_to(repo).parts)
    assert largest[0] < 50 * 1024 * 1024, largest
    subprocess.run(['git', '-C', str(repo), 'add', '--', 'analysis/benchmark905'], check=True)
    subprocess.run(['git', '-C', str(repo), '-c', 'gc.auto=0', 'commit', '-m',
                    'data: publish complete formal 905 evidence as verified readable CSV shards'], check=True,
                   stdout=(run / 'logs/data_commit.log').open('x'))
    subprocess.run(['git', '-C', str(repo), '-c', 'pack.threads=2', '-c', 'pack.window=0',
                    'repack', '-a', '-d'], check=True)
    result = {'status': 'READY_FOR_GITHUB', 'repository': str(repo), 'coverage': coverage,
              'data_commit': subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
              'copied_byte_identical_files': copied, 'sharded_tables': len(tables),
              'total_csv_shards': sum(len(t['parts']) for t in tables.values()),
              'largest_file_bytes': largest[0], 'largest_file': largest[1],
              'slurm_job_id': os.environ['SLURM_JOB_ID'], 'no_experiments_executed': True}
    write_json(run / 'output/github_preparation.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
