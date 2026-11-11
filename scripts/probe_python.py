#!/usr/bin/env python3
"""Probe an imported Python entrypoint without allowing historical side effects.

This is an audit hook for cooperative local research programs, not an adversarial
security sandbox. Subprocesses/network are denied; writes stay in the probe dir.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import resource
import runpy
import sys
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--scratch', type=Path, required=True)
    parser.add_argument('--mode', choices=['import', 'help'], default='import')
    args = parser.parse_args()
    root, target, scratch = args.root.resolve(), args.target.resolve(), args.scratch.resolve()
    if not os.environ.get('SLURM_JOB_ID') or not str(scratch).startswith('/data/wushangge/'):
        raise SystemExit('Probes require Slurm and /data.')
    scratch.mkdir(parents=True, exist_ok=False)
    os.chdir(scratch)
    resource.setrlimit(resource.RLIMIT_CPU, (20, 22))
    sys.dont_write_bytecode = True
    os.environ.update(MPLCONFIGDIR=str(scratch / 'matplotlib'), XDG_CACHE_HOME=str(scratch / 'cache'),
                      TMPDIR=str(scratch))

    class GuardViolation(RuntimeError):
        pass

    def writable(value):
        if isinstance(value, int):
            return
        if value is not None and os.fsdecode(value) == '/dev/null':
            return  # Torch discards dynamic-loader query output here.
        if value is not None and not Path(os.fsdecode(value)).resolve().is_relative_to(scratch):
            raise GuardViolation('Write outside isolated probe directory: ' + os.fsdecode(value))

    def guard(event, values):
        if event in {'subprocess.Popen', 'os.system', 'os.fork', 'os.forkpty', 'os.posix_spawn'} or event.startswith('socket.'):
            raise GuardViolation('Import/help attempted external execution or networking: ' + event +
                                 (' ' + repr(values[:2]) if event == 'subprocess.Popen' else ''))
        if event == 'open':
            path, mode, flags = values
            if (mode and any(c in mode for c in 'wax+')) or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC):
                writable(path)
        elif event in {'os.mkdir', 'os.remove', 'os.rmdir', 'os.chmod', 'os.utime'}:
            writable(values[0])
        elif event in {'os.rename', 'os.link', 'os.symlink'}:
            writable(values[0]); writable(values[1])

    sys.addaudithook(guard)
    relative = target.relative_to(root)
    sys.path[:0] = [str(target.parent), str(root), str(root / 'projects/litecut'),
                    str(root / 'projects/litecut/src'), str(root / 'projects/culs/tools'), str(root / 'tests')]
    if relative.parts[:2] == ('projects', 'litecut'):
        module = '.'.join(relative.with_suffix('').parts[2:])
    elif relative.parts[0] in {'global_timing', 'tests', 'scripts'}:
        module = '.'.join(relative.with_suffix('').parts)
    else:
        module = '_probe_' + target.stem
    if module.endswith('.__init__'):
        module = module[:-9]
    result = {'target': str(target), 'mode': args.mode, 'status': 'RUNNING'}
    sys.argv = [str(target)] + (['--help'] if args.mode == 'help' else [])
    try:
        if args.mode == 'help':
            if module.startswith(('src.', 'global_timing.', 'scripts.')):
                runpy.run_module(module, run_name='__main__')
            else:
                runpy.run_path(str(target), run_name='__main__')
        else:
            spec = importlib.util.spec_from_file_location(module, target,
                submodule_search_locations=[str(target.parent)] if target.name == '__init__.py' else None)
            loaded = importlib.util.module_from_spec(spec)
            sys.modules[module] = loaded
            spec.loader.exec_module(loaded)
        result['status'] = 'PASS'
    except SystemExit as exc:
        result.update(status='PASS' if args.mode == 'help' and exc.code in {0, None} else 'FAIL',
                      reason=f'SystemExit({exc.code})')
    except GuardViolation as exc:
        result.update(status='SKIP', reason=str(exc), requirement='Isolated adapter for import-time side effects.')
    except BaseException as exc:
        result.update(status='FAIL', reason=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
    result['scope'] = 'Import or --help only; no functional/training/evaluation claim.'
    result['loaded_source_files'] = sorted({str(Path(m.__file__).resolve()) for m in list(sys.modules.values())
                                         if getattr(m, '__file__', None) and str(m.__file__).endswith('.py')})
    with (scratch / 'result.json').open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k not in {'loaded_source_files', 'traceback'}}))


if __name__ == '__main__':
    main()
