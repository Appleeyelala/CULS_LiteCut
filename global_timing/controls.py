"""Standalone baselines with the same candidate proofs and real validation.

Random controls do not import Torch, infer Student scores, or fit a surrogate.
Native controls do not enumerate a candidate domain. These are real methods,
not zero-gradient surrogates, and their smaller costs must remain visible.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import random
import time

from .boolean_candidates import parse_library, verify_library_truth
from .candidates import enumerate_candidates
from .common import ROOT, ToolError, digest, run_tool, save_json, tool_path
from .cut_validation import validate_catalogue
from .evaluation import Evaluator
from .execution import execute_many, execute_one, repair_rejected
from .structure import make_partition, parse_aig


CONTROL_VARIANTS = ('original', 'native_abc', 'native_culs', 'random_cpu',
                    'random_batch', 'random_physical_batch', 'greedy_litecut',
                    'greedy_litecut_physical')


@dataclass
class ControlConfig:
    seed: int = 20261111
    evaluation_budget: int = 8
    warmup_evaluations: int = 2
    feedback_batch: int = 4
    timeout_s: int = 900
    area_limit_ratio: float = 1.05
    allow_nonpositive_gain: bool = False

    def validate(self):
        if (self.evaluation_budget < 1 or self.feedback_batch < 1 or self.timeout_s < 1
                or self.area_limit_ratio < 1 or self.warmup_evaluations < 0
                or type(self.allow_nonpositive_gain) is not bool):
            raise ValueError('Invalid standalone baseline configuration.')


class CandidateControl:
    def __init__(self, aig, gpuls, abc, library, checkpoint, output, config,
                 use_litecut=False, validation_workers=2, on_completion=None):
        config.validate()
        import os
        if not 1 <= validation_workers <= int(os.environ.get('SLURM_CPUS_PER_TASK', '1')):
            raise ValueError('Validation workers exceed the Slurm allocation.')
        started = time.perf_counter()
        self.config, self.aig, self.gpuls = config, Path(aig), Path(gpuls)
        self.output, self.validation_workers = Path(output), validation_workers
        self.on_completion = on_completion
        self.output.mkdir(parents=True, exist_ok=False)
        self.evaluator = Evaluator(Path(abc), Path(library), self.aig, config.timeout_s)
        save_json(self.output / 'config.json', {**asdict(config), 'input': str(self.aig),
            'input_sha256': digest(self.aig), 'gpuls_sha256': digest(self.gpuls),
            'toolchain': self.evaluator.provenance, 'uses_litecut': use_litecut,
            'checkpoint_sha256': digest(Path(checkpoint)) if use_litecut else None})
        self.partition = make_partition(parse_aig(self.aig))
        self.partition.write(self.output / 'partition')
        catalogue = enumerate_candidates(self.gpuls, self.aig, self.partition,
                                          self.output / 'candidates', config.timeout_s)
        self.catalogue, self.cut_audit = validate_catalogue(self.evaluator.abc, self.partition,
            catalogue, self.output / 'cut_validation', config.timeout_s)
        self.cache = {'cache_hit': None, 'reason': 'Random baseline does not use Student.'}
        self.scores = None
        if use_litecut:
            from .litecut import score_catalogue
            self.scores, self.cache = score_catalogue(self.aig, self.partition, self.catalogue,
                Path(checkpoint), self.output.parent / 'litecut_cache')
            save_json(self.output / 'litecut.json', {'scores': self.scores, 'cache': self.cache})
        else:
            library_path = ROOT / 'projects/culs/src/algorithms/rewrite_library.inc'
            rewrite_library = parse_library(library_path)
            proofs = []
            for index, decision in enumerate(self.catalogue.decisions):
                for candidate in decision['candidate_subgraph_indices']:
                    proofs.append({'decision_index': index, 'subgraph_index': candidate,
                                   **verify_library_truth(decision, candidate, rewrite_library)})
            save_json(self.output / 'candidate_truth.json', {
                'library_sha256': digest(library_path), 'proofs': proofs,
                'scope': 'Same exhaustive candidate Boolean checks as the Student adapter; no delay predictions.'})
        self.noop = [0] * len(self.catalogue.decisions)
        self.preparation_runtime_s = time.perf_counter() - started

    def evaluate_choice(self, choices, method, index, predicted=None):
        began = time.perf_counter()
        result = execute_one(self, choices, method, index, predicted)
        if self.on_completion:
            self.on_completion(result, began, time.perf_counter())
        return result

    evaluate_many = execute_many
    repair_choice = repair_rejected

    def rank(self, row, reference):
        if (row['status'] != 'PASS' or
                row['metrics']['area'] > reference['area'] * self.config.area_limit_ratio + 1e-8):
            return float('inf')
        return row['metrics']['delay_ps']

    def random_choice(self, rng):
        # Identical variable-density sampler to frozen Experiment.random_choice.
        density = rng.uniform(0.05, 0.8)
        scores = []
        for decision in self.catalogue.decisions:
            row = [0.0] + [-1.0] * decision['candidate_count']
            if rng.random() < density:
                row[rng.randrange(1, len(row))] = rng.random() + 0.01
            scores.append(row)
        return self.catalogue.project(scores)

    def fresh_random(self, rng, seen):
        for _ in range(500):
            choices = self.random_choice(rng)
            if tuple(choices) not in seen:
                return choices
        return None

    def run(self, method='random_search', batched=True):
        started = time.perf_counter()
        baseline = self.evaluate_choice(self.noop, 'replay_noop', 0)
        if baseline['status'] != 'PASS':
            raise ToolError('Required standalone no-op validation failed.')
        observations = [dict(baseline, method=method, baseline_reused=True)]
        seen, repairs = {tuple(self.noop)}, []
        rng = random.Random(self.config.seed)
        greedy = method == 'greedy_litecut'
        if greedy and self.scores is None:
            raise ValueError('Greedy control requires actual Student scores.')
        greedy_proposed = False
        stop_reason = 'evaluation_budget_reached'
        while len(observations) < self.config.evaluation_budget:
            remaining = self.config.evaluation_budget - len(observations)
            candidates = []
            if repairs:
                candidates, repairs = repairs[:remaining], repairs[remaining:]
            elif greedy:
                if greedy_proposed:
                    stop_reason = 'deterministic_greedy_point_and_explicit_repairs_finished'
                    break
                local = [[] for _ in self.catalogue.decisions]
                for row in self.scores:
                    local[row['decision_index']].append(-row['predicted_local_abc_delay'])
                candidates = [{'choices': self.catalogue.project(local), 'source': 'greedy_litecut'}]
                greedy_proposed = True
            else:
                # Same random warmup prefix and total request budget as gradient;
                # no unused ML fit or inner gradient steps are charged to random.
                count = 1 if not batched or len(observations) <= self.config.warmup_evaluations else self.config.feedback_batch
                known = set(seen)
                for _ in range(min(count, remaining)):
                    choice = self.fresh_random(rng, known)
                    if choice is None:
                        break
                    candidates.append({'choices': choice, 'source': 'random'})
                    known.add(tuple(choice))
            unique, known = [], set(seen)
            for candidate in candidates:
                if tuple(candidate['choices']) not in known:
                    unique.append(candidate)
                    known.add(tuple(candidate['choices']))
            candidates = unique
            if not candidates:
                stop_reason = 'no_unseen_point_found_by_sampling_or_repair'
                break
            if not batched:
                repairs = candidates[1:] + repairs
                candidates = candidates[:1]
            results = self.evaluate_many([c['choices'] for c in candidates], method,
                                        len(observations), candidates)
            if len(results) != len(candidates):
                raise ValueError('Standalone evaluator omitted a request.')
            for proposal, result in zip(candidates, results):
                if (result['choices'] != proposal['choices'] or result['index'] != len(observations)
                        or result['status'] not in ('PASS', 'FAIL')):
                    raise ValueError('Standalone evaluator substituted a request.')
                observations.append(result)
                seen.add(tuple(result['choices']))
                repair = self.repair_choice(result)
                if repair is not None and tuple(repair) not in seen:
                    repairs.append({'choices': repair, 'source': 'explicit_feasibility_repair',
                                    'parent_index': result['index'], 'parent_source': proposal['source']})
                best = min(observations, key=lambda r: self.rank(r, baseline['metrics']))
                save_json(self.output / method / f'progress_{len(observations):04d}.json', {
                    'best_choices': best['choices'], 'best_metrics': best['metrics'],
                    'evaluation_count': len(observations), 'last_status': result['status']})
            if greedy and any(r['status'] == 'PASS' for r in results):
                stop_reason = 'deterministic_greedy_point_and_explicit_repairs_finished'
                break
        best = min(observations, key=lambda r: self.rank(r, baseline['metrics']))
        summary = {'method': method, 'best': best, 'observations': observations,
            'stop_reason': stop_reason, 'evaluation_budget': self.config.evaluation_budget,
            'evaluations_including_baseline': len(observations),
            'gradient_proposal_evaluations': 0,
            'exclusive_method_runtime_s': time.perf_counter() - started,
            'scope': 'Real standalone control; no surrogate calibration. Failed requests and explicit repairs consume budget; no-op is a measured feasible fallback.'}
        save_json(self.output / method / 'summary.json', summary)
        return summary


def native_control(method, aig, gpuls, abc, library, output, timeout=900, on_completion=None):
    if method not in ('original', 'native_abc', 'native_culs'):
        raise ValueError('Unknown native control.')
    output.mkdir(parents=True, exist_ok=False)
    evaluator = Evaluator(abc, library, aig, timeout)
    started = time.perf_counter()
    candidate = aig if method == 'original' else output / 'rewritten.aig'
    row = {'method': method, 'index': 0, 'choices': None, 'status': 'RUNNING',
           'prediction_before_evaluation': {'source': method}, 'aig': str(candidate)}
    try:
        if method == 'native_abc':
            run_tool([abc, '-c', f'read_aiger {tool_path(aig)}; strash; rewrite; write_aiger {tool_path(candidate)}'],
                     output, 'rewrite', timeout)
        elif method == 'native_culs':
            run_tool([gpuls, '-c', f'read {tool_path(aig)}; rewrite; write {tool_path(candidate)}'],
                     output, 'rewrite', timeout)
        row.update(status='PASS', metrics=evaluator.evaluate(candidate, output / 'evaluation'))
    except (ToolError, OSError, ValueError) as exc:
        row.update(status='FAIL', error=f'{type(exc).__name__}: {exc}')
    row['runtime_s'] = time.perf_counter() - started
    save_json(output / 'observation.json', row)
    if on_completion:
        on_completion(row, started, time.perf_counter())
    return row
