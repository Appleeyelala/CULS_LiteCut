#!/usr/bin/env python3
"""Export existing STE100 evidence for GitHub; never run EDA or optimization."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def compact(value):
    if isinstance(value, list):
        return [compact(x) for x in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, item in value.items():
        if key in {'po_arrivals_ps', 'po_rise_fall', 'abc_native_trace_lines'}:
            continue
        if key == 'choices' and isinstance(item, list):
            result['choices_sparse'] = {'length': len(item), 'default': 0,
                                       'nonzero': [[i, c] for i, c in enumerate(item) if c]}
        else:
            result[key] = compact(item)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--experiment', type=Path, required=True)
    ap.add_argument('--destination', type=Path, required=True)
    args = ap.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not all(p.resolve().is_relative_to('/data/wushangge') for p in [args.experiment, args.destination]):
        ap.error('Use a Slurm /data publication workspace.')
    source, dest = args.experiment / 'output', args.destination
    dest.mkdir(parents=True, exist_ok=False)
    records = []

    def put(p, relative, transform=False):
        target = dest / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if transform:
            data = compact(json.loads(p.read_text()))
            with target.open('x') as f:
                json.dump(data, f, indent=2, allow_nan=False)
                f.write('\n')
        else:
            with p.open('rb') as a, target.open('xb') as b:
                shutil.copyfileobj(a, b)
        if not transform and sha(p) != sha(target):
            raise ValueError('Byte-identical copy failed.')
        records.append(dict(path=relative, original_path=str(p), original_sha256=sha(p),
                            sha256=sha(target), bytes=target.stat().st_size,
                            transformation='compact_v1' if transform else 'byte_identical'))

    for name in ['STE100_SAMPLE_MANIFEST.json', 'STE100_RESULTS.csv', 'STE100_SUMMARY.md',
                 'STE100_STATUS.json', 'DELIVERY_CHECK.json']:
        put(source / name, name)
    status = json.loads((source / 'STE100_STATUS.json').read_text())
    assert status['all_300_terminal'] and status['terminal'] == 300
    sample = json.loads((source / 'STE100_SAMPLE_MANIFEST.json').read_text())['rows']
    assert len(sample) == 100
    for row in sample:
        i = row['index']
        work = source / f'case_{i:04d}_Ours/work'
        put(work / 'summary.json', f'cases/{i:04d}/summary.json', True)
        for name in ['trajectory.json', 'proxy_best.json']:
            p = work / 'optimization' / name
            if p.exists():
                put(p, f'cases/{i:04d}/optimization/{name}')
        for method in ['A', 'B', 'Ours']:
            put(source / 'records' / f'{i:04d}_{method}.json', f'records/{i:04d}_{method}.json', True)
    for name in ['command.json', 'session_65534.json']:
        put(args.experiment / 'input' / name, 'execution/' + name)
    with (dest / 'FILE_MANIFEST.json').open('x') as f:
        json.dump(dict(source_commit=status['source_commit'], files=records,
                       compact_v1={'omitted_keys': ['po_arrivals_ps','po_rise_fall','abc_native_trace_lines'],
                                   'choices': 'Losslessly encoded as length/default=0/nonzero pairs; other keys retained.',
                                   'not_lossless_overall': 'PO arrays and mapped path text remain in original server files.'}),f,indent=2)
        f.write('\n')
    (dest / 'README.md').write_text('''# STE100：正式结果与梯度记录

来源：冻结源码 `9a199bf0ea0dcf20bf55e40da8caa8b8bbff5fac`，Slurm65534。
固定从905抽取100电路（seed20261116）；优化seed20261117，40步STE＋Adam。
A/B/Ours共300终态全部PASS；67个最终候选完整ACCEPTED及CEC PASS，31个真实改善。

- `STE100_RESULTS.csv`：100行三方法真实global Delay与完整冷启动Runtime。
- `STE100_SUMMARY.md`：真实配对比较；`STE100_STATUS.json`：300终态和Mapping次数。
- `STE100_SAMPLE_MANIFEST.json`：样本ID、输入路径、SHA256。
- `cases/NNNN/optimization/trajectory.json`：83个有候选电路的40步原始梯度/更新/loss轨迹（逐字节复制）。
- `cases/NNNN/optimization/proxy_best.json`：原始代理最优组合；此处预测不是真实Mapping标签。
- `cases/NNNN/summary.json`：100个Ours摘要，含阶段时间、后端、原图/最终真实测量与最终来源。
- `records/NNNN_{A,B,Ours}.json`：300个方法终态和完整Runtime。
- `execution/`：实际命令、节点/GPU/线程记录；`FILE_MANIFEST.json`：每个来源、hash及提取规则。

精简规则：cases摘要及records递归省略`po_arrivals_ps`、`po_rise_fall`和`abc_native_trace_lines`；choices向量改成可无损恢复的length/default/nonzero编码。其余字段保留。顶层五份交付文件及optimization文件保持原始字节。不存在未提交候选的真实Timing，不能把预测当标签。

未上传AIG、netlist、模型、库、大型候选目录和重复日志。完整原件仍在服务器：
`/data/wushangge/CULS_LiteCut/runs/20261116T070435Z-ste-adam-random100-d532863a/output/`。
本数据仅属STE100，禁止与`analysis/benchmark905/`旧方向导数/多Mapping反馈实验混算。
''')
    print(json.dumps({'files_with_provenance':len(records),'bytes':sum(r['bytes'] for r in records),'status':'EXPORTED_EXISTING_EVIDENCE'}))


if __name__ == '__main__':
    main()
