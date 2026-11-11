#!/usr/bin/env python3
"""Run upstream ABC unit tests and safe Gate1 utilities with real retained data."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from global_timing.common import digest, run_tool, save_json
from global_timing.structure import parse_aig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--tool-run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    if not os.environ.get('SLURM_JOB_ID') or not str(run).startswith('/data/wushangge/'):
        raise SystemExit('Imported program tests require a prepared Slurm /data run.')
    output = run / 'output/imported_tests'
    output.mkdir()
    records = []
    for name in ['gate1c_r_io_signature', 'gate1c_select', 'upstream_gia_test']:
        work = output / name
        work.mkdir()
        record = {'program': name, 'status': 'RUNNING'}
        try:
            if name == 'gate1c_r_io_signature':
                source = run / 'src/legacy/gate1_aigrewrite/gate1c_r_io_signature.py'
                text = run_tool([sys.executable, '-B', source, run / 'input/design.aig'], work, 'run')
                result = json.loads(text)
                aig = parse_aig(run / 'input/design.aig')
                assert (result['pi_count'], result['po_count'], result['and_count']) == (aig.pi_count, aig.po_count, aig.and_count)
                record.update(source_sha256=digest(source), result=result)
            elif name == 'gate1c_select':
                source = run / 'src/legacy/gate1_aigrewrite/gate1c_select.py'
                retained = Path('/data/wushangge/CULS_LiteCut/history/imports/gate1/gate1c_out/c1355_probe.json')
                (work / 'gate1c_out').mkdir()
                target = work / 'gate1c_out/c1355_probe.json'
                shutil.copyfile(retained, target)
                text = run_tool([sys.executable, '-B', source], work, 'run')
                result = json.loads(text)
                a, b = result['a'], result['b']
                assert set(a['saved_region_nodes']).isdisjoint(b['saved_region_nodes'])
                assert set(a['cut_leaves']).isdisjoint(b['saved_region_nodes'])
                assert set(b['cut_leaves']).isdisjoint(a['saved_region_nodes'])
                record.update(source_sha256=digest(source), input_sha256=digest(target), input_source=str(retained), result=result)
            else:
                source = run / 'src/third_party/abc/test/gia/gia_test.cc'
                abc = args.tool_run / 'build/framework_tools/abc'
                objects = sorted(p for p in abc.rglob('*.o') if p.relative_to(abc).as_posix() != 'src/base/main/main.o')
                if not objects:
                    raise FileNotFoundError('No compiled ABC translation units.')
                library = work / 'libabc_for_test.a'
                run_tool(['ar', 'rcs', library, *objects], work, 'archive', 120)
                conda = Path('/home/share/process/anaconda3')
                binary = work / 'gia_test'
                run_tool(['/usr/local/bin/g++', '-std=c++17', '-DABC_USE_STDINT_H',
                          '-I' + str(abc / 'src'), '-I' + str(conda / 'include'), source, library,
                          '-L' + str(conda / 'lib'), '-Wl,-rpath,' + str(conda / 'lib'),
                          '-lgtest_main', '-lgtest', '-pthread', '-ldl', '-lm', '-lrt', '-o', binary], work, 'link', 120)
                xml = work / 'tests.xml'
                run_tool([binary, '--gtest_output=xml:' + str(xml)], work, 'run', 120)
                result = ET.parse(xml).getroot()
                assert int(result.attrib['tests']) == 4 and int(result.attrib['failures']) == 0
                record.update(source_sha256=digest(source), tests=result.attrib, binary_sha256=digest(binary),
                              dependency='Existing shared GoogleTest 1.11.0; no download; tests unchanged from ABC snapshot.',
                              dependency_sha256=digest(conda / 'lib/libgtest.so.1.11.0'),
                              linked_objects=[{'path': str(p), 'sha256': digest(p)} for p in objects])
            record['status'] = 'PASS'
        except Exception as exc:
            record.update(status='FAIL', reason=f'{type(exc).__name__}: {exc}')
        save_json(work / 'result.json', record)
        records.append(record)
        print(json.dumps({k: v for k, v in record.items() if k not in {'linked_objects', 'result'}}), flush=True)
    save_json(output / 'summary.json', records)
    return 0 if all(r['status'] == 'PASS' for r in records) else 1


if __name__ == '__main__':
    raise SystemExit(main())
