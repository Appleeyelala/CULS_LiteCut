"""One original AIG, independent candidate configurations, complete validation.

Native enumeration/context reuse is opt-in; each candidate still requires
exact requested acceptance and its own CEC/mapping. Parallel CPU validation
is an engineering contribution, separate from CUDA proposal acceleration.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import time

from .common import ToolError, read_jsonl, run_tool, save_json, tool_path
from .pipeline import Experiment


def replay_many(gpuls, aig, owners, catalogue, choices, output, destinations,
                timeout=300, prepared=True):
    if not choices or len(choices) != len(destinations):
        raise ValueError('Nonempty batch with one output per request required.')
    if len(set(map(Path, destinations))) != len(destinations):
        raise ValueError('Each request needs a distinct output directory.')
    requested = [catalogue.requested(vector) for vector in choices]
    output.mkdir(parents=True, exist_ok=False)
    names = [f'choice_{i:04d}' for i in range(len(choices))]
    config = output / 'choices.txt'
    config.write_text(''.join(''.join(f'{name} {r} {c} {s}\n' for r, c, s in items)
                              or f'{name} -1 -1 -1\n' for name, items in zip(names, requested)))
    raw = output / 'raw'
    env = dict(os.environ)
    env.pop('CULS_REPLAY_PREPARED', None)
    if prepared:
        env['CULS_REPLAY_PREPARED'] = '1'
    common_error = None
    try:
        run_tool([gpuls, '-c', f'read {tool_path(aig)}; gate3e_execution_replay -o {tool_path(raw)} '
                  f'--region-map {tool_path(owners)} --config {tool_path(config)}'], output, 'rewrite', timeout, env)
        manifest = read_jsonl(raw / 'replay_manifest.jsonl')
        if len(manifest) != len(names) or {r['config_id'] for r in manifest} != set(names):
            raise ToolError('Native batch did not finish exactly all requested configurations.')
        if prepared:
            provenance = json.loads((raw / 'prepared_replay.json').read_text())
            if provenance['enumeration_count'] != 1 or provenance['configuration_count'] != len(choices):
                raise ToolError('Prepared enumeration provenance is missing or inconsistent.')
    except Exception as exc:
        common_error = f'{type(exc).__name__}: {exc}'
    results = []
    for name, vector, items, destination in zip(names, choices, requested, destinations):
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=False)
        (destination / 'choices.txt').write_text(''.join(f'choice {r} {c} {s}\n' for r, c, s in items)
                                                or 'choice -1 -1 -1\n')
        result = {'status': 'RUNNING', 'choices': vector, 'requested': items,
                  'order': 'ascending_original_root', 'prepared_enumeration': prepared,
                  'shared_command': str(output / 'rewrite.command.json'), 'native_config_id': name}
        try:
            if common_error:
                raise ToolError(common_error)
            trace_path = raw / 'traces' / (name + '.jsonl')
            aig_path = raw / 'aigs' / (name + '.aig')
            trace = read_jsonl(trace_path)
            selected_rows = [r for r in trace if r['preselected']]
            selected = {(r['root'], r['cut_slot'], r['selected_subgraph']): r for r in selected_rows}
            if len(selected_rows) != len(items) or set(selected) != set(items):
                raise ToolError('Batch replay did not preselect exactly the requested candidates.')
            rejected = [{'root': k[0], 'cut_slot': k[1], 'subgraph_index': k[2], 'status': row['final_status']}
                        for k, row in selected.items() if row['final_status'] != 'ACCEPTED']
            applied = sum(row['final_status'] == 'ACCEPTED' for row in trace)
            result.update(applied_rewrites=applied, rejected_requests=rejected,
                          rejected={str((r['root'], r['cut_slot'], r['subgraph_index'])): r['status'] for r in rejected})
            # Alias actual raw evidence into each request's ordinary layout.
            for folder, filename, source in [('traces', 'choice.jsonl', trace_path), ('aigs', 'choice.aig', aig_path)]:
                alias = destination / 'raw' / folder
                alias.mkdir(parents=True)
                if source.is_file():
                    (alias / filename).symlink_to(source.resolve())
            if rejected or applied != len(items):
                raise ToolError('Batch combination not fully accepted; refusing partial installation.')
            candidate = destination / 'raw/aigs/choice.aig'
            if not candidate.is_file():
                raise ToolError('Batch replay did not produce its requested AIG.')
            result.update(status='PASS', aig=str(candidate))
        except Exception as exc:
            result.update(status='FAIL', error=f'{type(exc).__name__}: {exc}')
        save_json(destination / 'replay.json', result)
        results.append(result)
    save_json(output / 'batch_replay.json', {'requests': results, 'native_command_count': 1,
                                           'prepared_enumeration': prepared,
                                           'scope': 'One native command; independent configurations, exact acceptance required.'})
    return results


class BatchedExperiment(Experiment):
    def __init__(self, *args, validation_workers=2, on_completion=None, **kwargs):
        if type(validation_workers) is not int or not 1 <= validation_workers <= int(os.environ.get('SLURM_CPUS_PER_TASK', '1')):
            raise ValueError('Validation workers must fit the Slurm CPU allocation.')
        self.validation_workers = validation_workers
        self.on_completion = on_completion
        super().__init__(*args, **kwargs)

    def evaluate_many(self, choices, method, first_index, predictions):
        if len(choices) != len(predictions) or not choices:
            raise ValueError('A prediction is required before each batched request.')
        for i, (vector, prediction) in enumerate(zip(choices, predictions)):
            save_json(self.output / method / 'predictions' / f'eval_{first_index + i:04d}.json',
                      {'choices': vector, 'prediction': prediction,
                       'created_monotonic': time.perf_counter(), 'phase': 'before_native_replay'})
        if len(choices) == 1:
            return [self.evaluate_choice(choices[0], method, first_index, predictions[0])]
        started = time.perf_counter()
        works = [self.output / method / f'eval_{first_index + i:04d}' for i in range(len(choices))]
        for work in works:
            work.mkdir(parents=True, exist_ok=False)
        batch = self.output / method / f'_batch_{first_index:04d}'
        executions = replay_many(self.gpuls, self.aig, self.output / 'partition/owners.csv', self.catalogue,
                                 choices, batch, [w / 'rewrite' for w in works], self.config.timeout_s)

        def evaluate(index):
            result = {'method': method, 'index': first_index + index, 'choices': choices[index],
                      'prediction_before_evaluation': predictions[index], 'status': 'RUNNING',
                      'batch': str(batch), 'runtime_scope': 'Shared replay plus own validation; concurrent request times overlap.'}
            execution = executions[index]
            try:
                if execution['status'] != 'PASS':
                    result['execution_failure'] = execution
                    raise ToolError(execution['error'])
                metrics = self.evaluator.evaluate(Path(execution['aig']), works[index] / 'evaluation')
                result.update(status='PASS', metrics=metrics, applied_rewrites=execution['applied_rewrites'], aig=execution['aig'])
            except Exception as exc:
                result.update(status='FAIL', error=f'{type(exc).__name__}: {exc}')
            result['runtime_s'] = time.perf_counter() - started
            save_json(works[index] / 'observation.json', result)
            if self.on_completion:
                self.on_completion(result, started, time.perf_counter())
            return index, result

        results = [None] * len(choices)
        with ThreadPoolExecutor(max_workers=self.validation_workers) as pool:
            for future in as_completed([pool.submit(evaluate, i) for i in range(len(choices))]):
                index, result = future.result()
                results[index] = result
        save_json(batch / 'validation.json', {'workers': self.validation_workers, 'requests': len(choices),
                                             'wall_s': time.perf_counter() - started,
                                             'validation_attempts': sum(e['status'] == 'PASS' for e in executions),
                                             'cec_calls': sum((w / 'evaluation/cec.command.json').is_file() for w in works),
                                             'mapping_calls': sum((w / 'evaluation/mapping.command.json').is_file() for w in works)})
        return results
