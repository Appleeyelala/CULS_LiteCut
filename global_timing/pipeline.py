"""Executable AIG -> candidates -> gradient search -> CEC/Liberty feedback."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import time

import torch

from .candidates import enumerate_candidates, replay
from .common import ToolError, digest, run_tool, save_json, tool_path
from .evaluation import Evaluator, failed_comparisons
from .cut_validation import validate_catalogue
from .litecut import score_catalogue
from .optimization import gradient_proposals
from .structure import make_partition, parse_aig
from .surrogate import TimingSurrogate, scalar_objective


@dataclass
class SearchConfig:
    seed: int = 20261111
    evaluation_budget: int = 24
    warmup_evaluations: int = 6
    gradient_steps: int = 60
    gradient_restarts: int = 2
    calibration_steps: int = 80
    feedback_batch: int = 4
    area_limit_ratio: float = 1.05
    area_weight: float = 0.5
    timeout_s: int = 300
    modes: tuple[str, ...] = ('area_constraint', 'joint')

    def validate(self):
        if self.evaluation_budget < 1 or self.feedback_batch < 1 or self.timeout_s < 1:
            raise ValueError('Evaluation budget, batch size and timeout must be positive.')
        if min(self.gradient_steps, self.gradient_restarts, self.calibration_steps) < 1:
            raise ValueError('Optimizer steps/restarts must be positive.')
        if self.warmup_evaluations < 0 or self.area_limit_ratio < 1 or not 0 < self.area_weight < 1:
            raise ValueError('Invalid warmup, area limit or objective weight.')
        if not self.modes or any(m not in {'area_constraint', 'joint'} for m in self.modes):
            raise ValueError('Unknown optimization mode.')


class Experiment:
    def __init__(self, aig, gpuls, abc, library, checkpoint, output, config):
        started = time.monotonic()
        config.validate()
        self.config = config
        self.aig, self.gpuls, self.output = Path(aig), Path(gpuls), Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.checkpoint = Path(checkpoint)
        self.evaluator = Evaluator(Path(abc), Path(library), self.aig, config.timeout_s)
        save_json(self.output / 'config.json', {**asdict(config), 'input': str(self.aig),
                  'input_sha256': digest(self.aig), 'gpuls_sha256': digest(self.gpuls),
                  'checkpoint_sha256': digest(self.checkpoint), 'toolchain': self.evaluator.provenance})
        self.partition = make_partition(parse_aig(self.aig))
        self.partition.write(self.output / 'partition')
        self.catalogue = enumerate_candidates(self.gpuls, self.aig, self.partition,
                                               self.output / 'candidates', config.timeout_s)
        self.catalogue, self.cut_audit = validate_catalogue(self.evaluator.abc, self.partition, self.catalogue,
                                                           self.output / 'cut_validation', config.timeout_s)
        self.scores, self.cache = score_catalogue(self.aig, self.partition, self.catalogue,
                                                 self.checkpoint, self.output.parent / 'litecut_cache')
        save_json(self.output / 'litecut.json', {'scores': self.scores, 'cache': self.cache})
        self.noop = [0] * len(self.catalogue.decisions)
        self.preparation_runtime_s = time.monotonic() - started

    def evaluate_choice(self, choices, method, index, predicted=None):
        work = self.output / method / f'eval_{index:04d}'
        work.mkdir(parents=True, exist_ok=False)
        start = time.monotonic()
        result = {'method': method, 'index': index, 'choices': choices, 'status': 'RUNNING', 'prediction_before_evaluation': predicted}
        try:
            candidate, execution = replay(self.gpuls, self.aig, self.output / 'partition/owners.csv',
                                           self.catalogue, choices, work / 'rewrite', self.config.timeout_s)
            metrics = self.evaluator.evaluate(candidate, work / 'evaluation')
            result.update(status='PASS', metrics=metrics, applied_rewrites=execution['applied_rewrites'], aig=str(candidate))
        except Exception as exc:
            result.update(status='FAIL', error=f'{type(exc).__name__}: {exc}')
            trace_summary = work / 'rewrite/replay.json'
            if trace_summary.exists():
                result['execution_failure'] = json.loads(trace_summary.read_text())
        result['runtime_s'] = time.monotonic() - start
        save_json(work / 'observation.json', result)
        return result

    def repair_choice(self, observation):
        """A new, explicit request, never reinterpret a failed partial replay."""
        rejected = observation.get('execution_failure', {}).get('rejected_requests', [])
        if not rejected:
            return None
        failed = {(row['root'], row['cut_slot']) for row in rejected}
        choices = observation['choices'].copy()
        for index, decision in enumerate(self.catalogue.decisions):
            if (decision['root'], decision['cut_slot']) in failed:
                choices[index] = 0
        self.catalogue.validate(choices)
        return choices if choices != observation['choices'] else None

    def deterministic_baseline(self, choices, method, baseline, selection_runtime_s=0.0):
        started = time.monotonic()
        rows = []
        seen = set()
        while len(rows) < self.config.evaluation_budget and tuple(choices) not in seen:
            seen.add(tuple(choices))
            reason = None if not rows else {'source': 'explicit_feasibility_repair', 'parent_index': rows[-1]['index']}
            observation = self.evaluate_choice(choices, method, len(rows), reason)
            rows.append(observation)
            if observation['status'] == 'PASS':
                break
            choices = self.repair_choice(observation)
            if choices is None:
                break
        successful = [row for row in rows if row['status'] == 'PASS']
        summary = {'method': method, 'observations': rows, 'status': 'PASS' if successful else 'FAIL',
                   'best': successful[-1] if successful else None,
                   'fallback_noop': baseline if not successful else None,
                   'evaluations': len(rows), 'repair_is_new_replay': True,
                   'selection_runtime_s': selection_runtime_s,
                   'exclusive_method_runtime_s': time.monotonic() - started + selection_runtime_s}
        save_json(self.output / method / 'summary.json', summary)
        return summary

    def random_choice(self, rng):
        # Variable-density search includes sparse and joint configurations.
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
        # Do not claim exhaustion: random sampling can fail to find an unseen
        # point in a large constrained space. The reason is reported explicitly.
        return None

    def rank_actual(self, observation, reference, mode):
        if observation['status'] != 'PASS':
            return float('inf')
        metrics = observation['metrics']
        if mode == 'area_constraint' and metrics['area'] > reference['area'] * self.config.area_limit_ratio + 1e-8:
            return float('inf')
        if mode == 'area_constraint':
            return metrics['delay_ps']
        po = torch.tensor(metrics['po_arrivals_ps'], dtype=torch.float64)
        area = torch.tensor(metrics['area'], dtype=torch.float64)
        return float(scalar_objective(po, area, reference['delay_ps'], reference['area'], mode,
                                      self.config.area_limit_ratio, area_weight=self.config.area_weight))

    def gradient_search(self, baseline, mode):
        started = time.monotonic()
        method = 'gradient_' + mode
        model = TimingSurrogate(self.partition, self.catalogue, self.scores, baseline['metrics'])
        observations = [dict(baseline, method=method, baseline_reused=True)]
        seen = {tuple(self.noop)}
        rng = random.Random(self.config.seed)
        iteration, stop_reason = 0, 'evaluation_budget_reached'
        repair_pending = []
        fitted_indices = []
        while len(observations) < self.config.evaluation_budget:
            remaining = self.config.evaluation_budget - len(observations)
            candidates = []
            if repair_pending:
                candidates = repair_pending[:min(remaining, self.config.feedback_batch)]
                repair_pending = repair_pending[len(candidates):]
            elif len(observations) <= self.config.warmup_evaluations:
                choices = self.fresh_random(rng, seen)
                if choices is not None:
                    candidates = [{'choices': choices, 'source': 'calibration_warmup'}]
            else:
                calibration = model.fit_observations(observations, steps=self.config.calibration_steps)
                fitted_indices = [row['index'] for row in observations if row['status'] == 'PASS']
                calibration['training_observation_indices'] = fitted_indices.copy()
                save_json(self.output / method / f'calibration_{iteration:04d}.json', calibration)
                best = min(observations, key=lambda row: self.rank_actual(row, baseline['metrics'], mode))
                proposals, trace = gradient_proposals(model, seen, self.config.seed + iteration,
                    incumbent=best['choices'], steps=self.config.gradient_steps, restarts=self.config.gradient_restarts,
                    mode=mode, area_limit_ratio=self.config.area_limit_ratio, area_weight=self.config.area_weight)
                save_json(self.output / method / f'gradients_{iteration:04d}.json', trace)
                candidates = [dict(p, source='joint_selection_gradient') for p in proposals[:min(remaining, self.config.feedback_batch)]]
            if not candidates:
                choices = self.fresh_random(rng, seen)
                if choices is None:
                    stop_reason = 'no_unseen_feasible_point_found_by_sampling'
                    break
                candidates = [{'choices': choices, 'source': 'exploration_after_duplicate_projection'}]
            for candidate in candidates:
                choices = candidate['choices']
                if tuple(choices) in seen:
                    continue
                with torch.no_grad():
                    po, area = model(model.one_hot(choices))
                prediction = dict(candidate, po_arrivals_ps=po.tolist(), area=float(area),
                                  delay_ps=float(po.max()), training_observations=len(fitted_indices),
                                  training_observation_indices=fitted_indices.copy())
                observation = self.evaluate_choice(choices, method, len(observations), prediction)
                observations.append(observation)
                seen.add(tuple(choices))
                repaired = self.repair_choice(observation)
                if repaired is not None and tuple(repaired) not in seen:
                    repair_pending.append({'choices': repaired, 'source': 'explicit_feasibility_repair',
                                           'parent_index': observation['index'], 'parent_source': candidate['source']})
                best = min(observations, key=lambda row: self.rank_actual(row, baseline['metrics'], mode))
                save_json(self.output / method / f'progress_{len(observations):04d}.json',
                          {'best_choices': best['choices'], 'best_metrics': best.get('metrics'),
                           'evaluation_count': len(observations), 'last_status': observation['status']})
            iteration += 1
        best = min(observations, key=lambda row: self.rank_actual(row, baseline['metrics'], mode))
        summary = {'method': method, 'observations': observations, 'best': best, 'stop_reason': stop_reason,
                   'exclusive_method_runtime_s': time.monotonic() - started,
                   'shared_baseline_runtime_s': baseline['runtime_s'],
                   'evaluation_budget': self.config.evaluation_budget, 'evaluations_including_baseline': len(observations),
                   'gradient_proposal_evaluations': sum(row.get('prediction_before_evaluation', {}).get('source') == 'joint_selection_gradient'
                       for row in observations if row.get('prediction_before_evaluation'))}
        save_json(self.output / method / 'summary.json', summary)
        return summary

    def independent_choices(self, baseline):
        model = TimingSurrogate(self.partition, self.catalogue, self.scores, baseline)
        combined = self.noop.copy()
        neighbors = {i: set() for i in range(len(combined))}
        for left, right in self.catalogue.conflicts:
            neighbors[left].add(right)
            neighbors[right].add(left)
        # Optimize each region with every other region frozen at no-rewrite.
        # Coordinate choices allow multiple compatible decisions per region.
        with torch.no_grad():
            for rid, indices in self.catalogue.region_decisions.items():
                if not indices:
                    continue
                state = model.independent_region_state(rid)
                active = set()
                for _ in range(2):
                    for index in indices:
                        if neighbors[index] & (active - {index}):
                            choice = 0  # the only feasible option
                        else:
                            costs = state.options(index)
                            choice = min((float(cost), candidate) for candidate, cost in enumerate(costs))[1]
                        state.set_choice(index, choice)
                        active.discard(index)
                        if choice:
                            active.add(index)
                for index, choice in state.choices.items():
                    combined[index] = choice
        # Cross-region conflicts require a final global legal projection.
        logits = [[1.0 if c == choice else 0.0 for c in range(d['candidate_count'] + 1)]
                  for d, choice in zip(self.catalogue.decisions, combined)]
        return self.catalogue.project(logits)

    def native_baseline(self):
        """Measure an independent control without substituting another result."""
        native = self.output / 'native_abc'
        native.mkdir()
        native_aig = native / 'rewritten.aig'
        started = time.monotonic()
        metrics = None
        attempt = {'status': 'RUNNING', 'stage': 'rewrite', 'aig': str(native_aig)}
        try:
            command = f'read_aiger {tool_path(self.aig)}; strash; rewrite; write_aiger {tool_path(native_aig)}'
            run_tool([self.evaluator.abc, '-c', command], native, 'rewrite', self.config.timeout_s)
            attempt['stage'] = 'cec_and_mapping'
            metrics = self.evaluator.evaluate(native_aig, native / 'evaluation')
            attempt['status'] = 'PASS'
        except Exception as exc:
            attempt.update(status='FAIL', error=f'{type(exc).__name__}: {exc}')
            if not isinstance(exc, (ToolError, OSError, ValueError)):
                raise
        finally:
            attempt['runtime_s'] = time.monotonic() - started
            save_json(native / 'attempt.json', attempt)
        return metrics, attempt

    def run(self):
        started = time.monotonic()
        original = self.evaluator.evaluate(self.aig, self.output / 'original')
        baseline = self.evaluate_choice(self.noop, 'replay_noop', 0)
        if baseline['status'] != 'PASS':
            raise RuntimeError('No-rewrite replay baseline failed; search cannot continue.')
        native_metrics, native_attempt = self.native_baseline()
        selecting_started = time.monotonic()
        local_scores = [[] for _ in self.catalogue.decisions]
        for row in self.scores:
            local_scores[row['decision_index']].append(-row['predicted_local_abc_delay'])
        choices = self.catalogue.project(local_scores)
        greedy = self.deterministic_baseline(choices, 'greedy_litecut', baseline,
                                             time.monotonic() - selecting_started)
        selecting_started = time.monotonic()
        choices = self.independent_choices(baseline['metrics'])
        independent = self.deterministic_baseline(choices, 'independent_regions', baseline,
                                                  time.monotonic() - selecting_started)
        rng = random.Random(self.config.seed)
        random_started = time.monotonic()
        random_rows, seen = [dict(baseline, baseline_reused=True)], {tuple(self.noop)}
        repair = None
        while len(random_rows) < self.config.evaluation_budget:
            choices = repair if repair is not None and tuple(repair) not in seen else self.fresh_random(rng, seen)
            if choices is None:
                break
            random_rows.append(self.evaluate_choice(choices, 'random_search', len(random_rows),
                               {'source': 'explicit_feasibility_repair' if repair == choices else 'random'}))
            seen.add(tuple(choices))
            repair = self.repair_choice(random_rows[-1])
        save_json(self.output / 'random_search/observations.json', random_rows)
        random_runtime = time.monotonic() - random_started
        searches = {mode: self.gradient_search(baseline, mode) for mode in self.config.modes}
        summary = {'status': 'COMPLETED', 'original': original, 'replay_noop': baseline,
                   'native_abc': native_metrics, 'native_abc_attempt': native_attempt,
                   'greedy_litecut': greedy, 'independent_regions': independent,
                   'random_search': random_rows, 'gradient_search': searches,
                   'candidate_count': sum(d['candidate_count'] for d in self.catalogue.decisions),
                   'cut_validation_counts': {k: self.cut_audit[k] for k in
                       ['raw_executable_decisions', 'verified_decisions', 'excluded_decisions']},
                   'region_count': len(self.partition.regions), 'config': asdict(self.config),
                   'preparation_runtime_s': self.preparation_runtime_s,
                   'exclusive_native_runtime_s': native_attempt['runtime_s'], 'exclusive_random_runtime_s': random_runtime,
                   'total_runtime_including_preparation_s': time.monotonic() - started + self.preparation_runtime_s,
                   'serialization_control': 'Original and zero-choice replay are measured separately; graph serialization may alter mapping.',
                   'scope': 'Measured benchmark outcomes; this run does not establish generalization to other designs.'}
        summary['failed_comparisons'] = failed_comparisons(summary)
        if summary['failed_comparisons']:
            summary['status'] = 'COMPLETED_WITH_FAILURES'
        save_json(self.output / 'summary.json', summary)
        return summary
