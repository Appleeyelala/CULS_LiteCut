"""One paid original anchor, proxy-only STE Adam, at most one final rewrite."""
from dataclasses import asdict, dataclass
from pathlib import Path
import time

import torch

from .candidates import enumerate_candidates
from .common import ROOT, digest, save_json
from .cut_validation import validate_catalogue
from .evaluation import Evaluator
from .execution import execute_one
from .feasibility import CandidateFeasibility
from .litecut import score_catalogue
from .mffc_model import MffcGpuSurrogate
from .mffc_gpu import graph_layout
from .mffc_ste_search import ste_adam_search
from .structure import make_partition, parse_aig
from .surrogate import TimingSurrogate


@dataclass(frozen=True)
class SteSearchConfig:
    seed: int = 20261117
    steps: int = 40
    learning_rate: float = 0.12
    maximum_edits: int = 16
    area_limit_ratio: float = 1.05
    timeout_s: int = 900
    allow_nonpositive_gain: bool = False
    cpu_small_decisions: int = 32
    cpu_small_regions: int = 1024

    def validate(self):
        if self != SteSearchConfig():
            raise ValueError('STE100 uses the predeclared fixed settings; no tuning.')


class SteExperiment:
    def __init__(self, aig, gpuls, abc, library, checkpoint, native_library, output,
                 config=SteSearchConfig(), validation_workers=2):
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
        self.baseline = {'metrics': self.original, 'choices': self.noop, 'status': 'PASS'}
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

    def run(self):
        best, trace, final = None, None, None
        selected = {'metrics': self.original, 'aig': str(self.aig), 'choices': self.noop,
                    'source': 'original_noop', 'applied_rewrites': 0}
        if self.model is not None:
            best, trace = self.stage('ste_adam_proxy_optimization', ste_adam_search,
                self.model, self.feasibility, self.config.seed, self.config.steps,
                self.config.learning_rate, self.config.maximum_edits)
            self.collect_native('ste_adam_optimization')
            save_json(self.output / 'optimization/trajectory.json', trace)
            save_json(self.output / 'optimization/proxy_best.json', best)
            if any(best['choices']):
                self.feasibility.validate(best['choices'])
                final = self.stage('final_replay_cec_mapping', execute_one, self,
                                  best['choices'], 'final', 0, best)
                selected['source'] = 'original_fallback'
                if (final['status'] == 'PASS'
                    and final['metrics']['area'] <= self.original['area'] * self.config.area_limit_ratio
                    and final['metrics']['delay_ps'] < self.original['delay_ps']):
                    selected = {'metrics': final['metrics'], 'aig': final['aig'],
                                'choices': final['choices'], 'source': 'ste_adam_verified_improvement',
                                'applied_rewrites': final['applied_rewrites']}
        mappings = list(self.output.rglob('mapping.command.json'))
        if len(mappings) > 2 or (not any((best or {}).get('choices', [])) and len(mappings) != 1):
            raise RuntimeError('Original plus at most one final mapping contract violated.')
        summary = {'status': 'PASS', 'selected': selected, 'original': self.original,
                   'final_candidate': final, 'proxy_best': best, 'config': asdict(self.config),
                   'mapping_calls': len(mappings), 'backend': self.backend,
                   'regions': len(self.partition.regions), 'decisions': len(self.noop),
                   'candidates': sum(d['candidate_count'] for d in self.catalogue.decisions),
                   'selected_original_mffcs': len({self.catalogue.decisions[i]['decision_owner_region']
                       for i, c in enumerate(selected['choices']) if c}),
                   'structure': self.structure_meta, 'native_profiles': self.native_profiles,
                   'stages': self.stages, 'adam_steps': trace['steps'] if trace else 0,
                   'logits_total_change_norm': trace['logits_total_change_norm'] if trace else None,
                   'distinct_hard_combinations_including_noop': trace['distinct_hard_combinations_including_noop'] if trace else 1,
                   'original_is_separate_paid_measurement': True,
                   'noop_semantics': 'Unmodified original AIG; no serialized replay-noop.',
                   'scope': 'One final exact acceptance + CEC + actual Liberty timing, no retries or feedback.'}
        save_json(self.output / 'summary.json', summary)
        return summary
