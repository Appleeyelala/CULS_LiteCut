"""Paid original/no-op, full-proxy joint gradients and exact real feedback.

This entry runs only Ours. Its internal measurements are not extra comparison
methods; all costs and failed/repair requests stay in its cold method runtime.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import time

import torch

from .candidates import enumerate_candidates
from .common import ROOT, digest, save_json
from .cut_validation import validate_catalogue
from .evaluation import Evaluator
from .execution import execute_many, execute_one, repair_rejected
from .feasibility import CandidateFeasibility
from .litecut import score_catalogue
from .mffc_model import MffcGpuSurrogate
from .mffc_gpu import graph_layout
from .mffc_search import gradient_blocks, prepare_selection_mask
from .structure import make_partition, parse_aig
from .surrogate import TimingSurrogate


@dataclass(frozen=True)
class MffcSearchConfig:
    seed: int = 20261114
    evaluation_budget: int = 4  # Includes no-op, failure and explicit repair.
    gradient_rounds: int = 2
    proposals_per_round: int = 2
    maximum_edits: int = 16
    calibration_steps: int = 12
    area_limit_ratio: float = 1.05
    timeout_s: int = 900
    allow_nonpositive_gain: bool = False
    cpu_small_decisions: int = 32
    cpu_small_regions: int = 1024

    def validate(self):
        if min(self.evaluation_budget, self.gradient_rounds, self.proposals_per_round,
               self.maximum_edits, self.calibration_steps, self.timeout_s) < 1:
            raise ValueError('Positive request/update/calibration limits required.')
        if self.allow_nonpositive_gain or self.area_limit_ratio != 1.05:
            raise ValueError('This revision fixes positive gain and no-op area ratio 1.05.')


class MffcExperiment:
    def __init__(self, aig, gpuls, abc, library, checkpoint, native_library, output,
                 config=MffcSearchConfig(), validation_workers=2):
        config.validate()
        self.config, self.aig, self.gpuls, self.output = config, Path(aig), Path(gpuls), Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.validation_workers, self.on_completion = validation_workers, None
        self.stages, self.native_profiles = [], []
        self.model = None
        self.evaluator = Evaluator(Path(abc), Path(library), self.aig, config.timeout_s)
        save_json(self.output / 'config.json', {'config': asdict(config), 'input': str(self.aig),
                  'input_sha256': digest(self.aig), 'gpuls_sha256': digest(self.gpuls),
                  'checkpoint_sha256': digest(Path(checkpoint)), 'native_library': str(native_library),
                  'native_library_sha256': digest(Path(native_library)), 'toolchain': self.evaluator.provenance})
        self.original = self.stage('original_cec_mapping', self.evaluator.evaluate, self.aig, self.output / 'original')
        self.partition = self.stage('mffc_partition', lambda: make_partition(parse_aig(self.aig)))
        self.partition.write(self.output / 'partition')
        sources, targets = [], []
        for rid, ports in enumerate(self.partition.transfers):
            for node in ports:
                sources.append(self.partition.owner.get(node, -1))
                targets.append(rid)
        layout = graph_layout(sources, targets, len(self.partition.regions)) if self.partition.regions else {}
        self.structure_meta = {'regions': len(self.partition.regions), 'region_edges': len(self.partition.edges),
                               'region_ports': len(sources), 'region_dag_layers': layout.get('layers', 0),
                               'max_layer_width': layout.get('max_layer_width', 0)}
        save_json(self.output / 'structure_meta.json', self.structure_meta)
        self.catalogue = self.stage('candidate_enumeration', enumerate_candidates, self.gpuls, self.aig,
                                    self.partition, self.output / 'candidates', config.timeout_s)
        self.catalogue, self.cut_audit = self.stage('boolean_validation', validate_catalogue,
            self.evaluator.abc, self.partition, self.catalogue, self.output / 'cut_validation', config.timeout_s)
        self.noop = [0] * len(self.catalogue.decisions)
        self.baseline = self.evaluate_choice(self.noop, 'replay_noop', 0)
        if self.baseline['status'] != 'PASS':
            raise RuntimeError('The required paid replay-no-op calibration failed.')
        if self.noop:
            self.feasibility = self.stage('candidate_reuse_preflight', CandidateFeasibility,
                self.partition.aig, self.catalogue, ROOT / 'projects/culs/src/algorithms/rewrite_library.inc')
            save_json(self.output / 'candidate_feasibility.json', self.feasibility.payload())
            self.scores, self.cache = self.stage('litecut_preparation_inference', score_catalogue,
                self.aig, self.partition, self.catalogue, Path(checkpoint), self.output / 'litecut_cache')
            save_json(self.output / 'litecut.json', {'cache': self.cache, 'scores_path': self.cache['path'],
                                                   'score_count': len(self.scores)})
            small = len(self.noop) < config.cpu_small_decisions and len(self.partition.regions) < config.cpu_small_regions
            if small:
                self.backend = {'device': 'cpu', 'reason': 'Fixed small-decision and small-region thresholds.',
                                'small_decisions': config.cpu_small_decisions, 'small_regions': config.cpu_small_regions}
                self.model = self.stage('global_model_preparation', TimingSurrogate, self.partition,
                                       self.catalogue, self.scores, self.baseline['metrics'])
            else:
                if not torch.cuda.is_available():
                    raise RuntimeError('Nontrivial MFFC GPU workload requires CUDA; refusing silent fallback.')
                self.backend = {'device': 'cuda', 'reason': 'Candidate/Region workload exceeds fixed CPU thresholds.'}
                self.model = self.stage('global_model_preparation', MffcGpuSurrogate, self.partition,
                    self.catalogue, self.scores, self.baseline['metrics'], native_library)
            prepare_selection_mask(self.model, self.feasibility)
            self.collect_native('initialization')
        else:
            self.backend = {'device': 'none', 'reason': 'No verified executable decisions; Student/proxy not loaded.'}
            self.cache = {'cache_hit': False, 'status': 'NOT_REQUIRED_NO_CANDIDATES'}
            save_json(self.output / 'litecut.json', self.cache)

    def stage(self, name, function, *args, **kwargs):
        began = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            if torch.cuda.is_initialized():
                torch.cuda.synchronize()
            self.stages.append({'stage': name, 'elapsed_s': time.perf_counter() - began})

    def collect_native(self, label):
        if isinstance(self.model, MffcGpuSurrogate):
            self.native_profiles.append({'stage': label, **self.model.native_plan.collect_profile()})

    def evaluate_choice(self, choices, method, index, predicted=None):
        return self.stage('real_replay_cec_mapping', execute_one, self, choices, method, index, predicted)

    def actual_rank(self, row):
        if row['status'] != 'PASS' or row['metrics']['area'] > self.baseline['metrics']['area'] * self.config.area_limit_ratio + 1e-8:
            return float('inf')
        return row['metrics']['delay_ps']

    def run(self):
        observations, seen = [self.baseline], {tuple(self.noop)}
        pending, round_index, stop_reason = [], 0, 'evaluation_budget_reached'
        fitted = [0]
        while self.model is not None and len(observations) < self.config.evaluation_budget:
            remaining = self.config.evaluation_budget - len(observations)
            pending = [p for p in pending if tuple(p['choices']) not in seen]
            if pending:
                candidates, pending = pending[:remaining], pending[remaining:]
            elif round_index < self.config.gradient_rounds:
                usable = [o for o in observations if o['status'] == 'PASS']
                if len(usable) > 1:
                    calibration = self.stage('gpu_or_cpu_proxy_calibration', self.model.fit_observations,
                                              observations, steps=self.config.calibration_steps)
                    fitted = calibration['training_observation_indices'] if 'training_observation_indices' in calibration else [o['index'] for o in usable]
                    save_json(self.output / 'optimization' / f'calibration_{round_index:02d}.json', calibration)
                    self.collect_native('calibration')
                incumbent = min(observations, key=self.actual_rank)
                candidates, trace = self.stage('global_gradient_and_joint_blocks', gradient_blocks,
                    self.model, self.feasibility, incumbent['choices'], seen,
                    self.config.maximum_edits, min(self.config.proposals_per_round, remaining))
                trace.update(round=round_index, seed=self.config.seed, training_observation_indices=fitted)
                save_json(self.output / 'optimization' / f'gradient_{round_index:02d}.json', trace)
                round_index += 1
                self.collect_native('global_gradient_and_hard_scoring')
            else:
                stop_reason = 'gradient_round_limit'
                break
            candidates = [c for c in candidates if tuple(c['choices']) not in seen][:remaining]
            if not candidates:
                stop_reason = 'no_unseen_legal_gradient_block'
                break
            for candidate in candidates:
                self.feasibility.validate(candidate['choices'])
                # Repair candidates need their own current prediction, saved by
                # execute_many before any native installation starts.
                if 'delay_ps' not in candidate:
                    with torch.no_grad():
                        po, area = self.model(self.model.one_hot(candidate['choices']))
                    candidate.update(po_arrivals_ps=po.cpu().tolist(), area=float(area), delay_ps=float(po.max()))
                candidate['training_observation_indices'] = fitted.copy()
            results = self.stage('real_batch_replay_cec_mapping', execute_many, self,
                [c['choices'] for c in candidates], 'optimization', len(observations), candidates)
            for candidate, row in zip(candidates, results):
                observations.append(row)
                seen.add(tuple(row['choices']))
                repaired = repair_rejected(self, row)
                if repaired is not None and tuple(repaired) not in seen and not any(p['choices'] == repaired for p in pending):
                    self.feasibility.validate(repaired)
                    pending.append({'choices': repaired, 'source': 'explicit_feasibility_repair',
                                    'parent_index': row['index'], 'parent_source': candidate['source']})
        best = min(observations, key=self.actual_rank)
        source = (best.get('prediction_before_evaluation') or {}).get('source', 'replay_noop')
        selected = {'metrics': best['metrics'], 'aig': best['aig'], 'choices': best['choices'],
                    'source': source, 'observation_index': best['index'], 'applied_rewrites': best['applied_rewrites']}
        if (self.original['area'] <= self.baseline['metrics']['area'] * self.config.area_limit_ratio + 1e-8
                and self.original['delay_ps'] < best['metrics']['delay_ps']):
            selected = {'metrics': self.original, 'aig': str(self.aig), 'choices': None,
                        'source': 'original_fallback', 'observation_index': None, 'applied_rewrites': 0}
        active_regions = set()
        for i, c in enumerate(selected['choices'] or []):
            if c:
                active_regions.add(self.catalogue.decisions[i]['decision_owner_region'])
        summary = {'status': 'PASS', 'selected': selected, 'original': self.original,
                   'replay_noop': self.baseline, 'observations': observations, 'config': asdict(self.config),
                   'stop_reason': stop_reason, 'gradient_rounds': round_index,
                   'requests_including_noop_failed_repair': len(observations),
                   'failed_requests': sum(o['status'] != 'PASS' for o in observations),
                   'direct_requests': sum((o.get('prediction_before_evaluation') or {}).get('source') == 'mffc_global_gradient' for o in observations),
                   'repair_requests': sum((o.get('prediction_before_evaluation') or {}).get('source') == 'explicit_feasibility_repair' for o in observations),
                   'selected_original_mffcs': len(active_regions), 'regions': len(self.partition.regions),
                   'decisions': len(self.noop), 'candidates': sum(d['candidate_count'] for d in self.catalogue.decisions),
                   'backend': self.backend, 'structure': self.structure_meta,
                   'native_profiles': self.native_profiles, 'stages': self.stages,
                   'original_is_separate_paid_measurement': True,
                   'scope': 'Actual exact installation/CEC/Liberty final; proxy direction is not true mapping gradient.'}
        save_json(self.output / 'summary.json', summary)
        return summary
