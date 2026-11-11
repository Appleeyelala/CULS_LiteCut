#!/usr/bin/env python3
"""Link actual CULS solver object into a CPU-only serialization regression."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from global_timing.common import digest, run_tool, save_json
from global_timing.evaluation import Evaluator
from global_timing.structure import parse_aig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--tool-run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    if not os.environ.get('SLURM_JOB_ID') or not str(run).startswith('/data/wushangge/'):
        raise SystemExit('Use a prepared Slurm /data run.')
    relative = 'projects/culs/src/algorithms/rewrite.cu'
    if digest(run / 'src' / relative) != digest(args.tool_run / 'src' / relative):
        raise SystemExit('Native source differs from built object.')
    output = run / 'output/native_writer'
    output.mkdir()
    binary = output / 'test_native_writer'
    source = run / 'src/projects/culs'
    obj = args.tool_run / 'build/framework_tools/culs/CMakeFiles/gpuls.dir/src/algorithms/rewrite.cu.o'
    run_tool(['/usr/local/bin/g++', '-std=c++17', '-I' + str(source / 'src'), '-I' + str(source / 'include'),
              '-I/usr/local/cuda-11.8/include', str(run / 'src/tests/native/replay_constants.cpp'), str(obj),
              '-L/usr/local/cuda-11.8/lib64', '-Wl,-rpath,/usr/local/cuda-11.8/lib64', '-lcudart', '-o', str(binary)],
             output, 'link', 120)
    run_tool([binary, output / 'actual'], output, 'writer', 60)
    content = bytearray(b'aig 8 3 0 5 5\n0\n1\n8\n10\n16\n')
    for node, pair in enumerate([(0, 2), (1, 4), (2, 4), (8, 10), (12, 6)], 4):
        low, high = sorted(pair)
        content.extend([2 * node - high, high - low])
    original = output / 'reference.aig'
    original.write_bytes(content)
    report = []
    abc = args.tool_run / 'build/framework_tools/abc/abc'
    for repetition in range(2):
        candidate = output / f'actual_{repetition}.aig'
        if parse_aig(candidate).po_literals[:2] != (0, 1):
            raise AssertionError('Constant output literals changed.')
        metrics = Evaluator(abc, run / 'input/asap7', original).evaluate(candidate, output / f'check_{repetition}')
        report.append(metrics)
    save_json(output / 'summary.json', {'status': 'PASS', 'scope': 'Actual CPUSolver Reset/Reorder/WriteAig, twice; no GPU rewrite.',
                                       'source_sha256': digest(run / 'src' / relative),
                                       'driver_sha256': digest(run / 'src/tests/native/replay_constants.cpp'),
                                       'object_sha256': digest(obj), 'binary_sha256': digest(binary), 'checks': report})


if __name__ == '__main__':
    main()
