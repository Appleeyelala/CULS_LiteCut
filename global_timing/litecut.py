"""Canonical 19D Student inference with batch evaluation and content cache."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import networkx as nx
import numpy as np
import torch

from .common import ROOT, digest, legacy, save_json
from .truth import ContextProofRequired, support_cut_truth


def original_cone(aig, decision):
    boundary = set(decision['cut_leaves']) | {0}
    root = decision['root']
    graph = {}
    pending = [root]
    seen = set()
    while pending:
        node = pending.pop()
        graph.setdefault(node, [])
        if node in boundary or node <= aig.pi_count or node in seen:
            continue
        seen.add(node)
        for literal in aig.fanin_literals[node]:
            source = literal >> 1
            graph.setdefault(source, []).append((node, literal & 1))
            pending.append(source)
    return graph


def verify_library_truth(decision, subgraph, library):
    """Exhaustive four-input truth check, allowing the engine's output phase.

    Engine ReplaceSubgr aligns zero-input phases when installing a DAG. The
    cut truth and complement are the only admissible library functions.
    End-to-end CEC independently checks actual installation and global phase.
    """
    truth = int(decision['truth']) & 0xffff
    phase = library['p_phases'][truth]
    permutation = library['perms4'][library['p_perms'][truth]]
    data = library['classes'][decision['truth_class']]
    patterns = [0xaaaa, 0xcccc, 0xf0f0, 0xff00]
    values = [(patterns[permutation[i]] ^ (0xffff if (phase >> i) & 1 else 0)) for i in range(4)]
    for a, b, ca, cb in data['fanins']:
        values.append((values[a] ^ (0xffff if ca else 0)) & (values[b] ^ (0xffff if cb else 0)))
    actual = values[data['roots'][subgraph]]
    if actual not in (truth, truth ^ 0xffff):
        raise ValueError(f'Library candidate function mismatch at {decision["root"]}:{decision["cut_slot"]}:{subgraph}')
    return {'truth_verified': True, 'output_phase_complement': actual != truth}


def verify_cut_truth(aig, decision):
    """Validate the exported cut's boundary and truth against original AIG IDs."""
    leaves = decision['cut_leaves']
    if len(leaves) not in {3, 4} or len(set(leaves)) != len(leaves):
        raise ValueError('Expected distinct 3/4-leaf rewrite cut.')
    values = {node: pattern for node, pattern in zip(leaves, [0xaaaa, 0xcccc, 0xf0f0, 0xff00])}
    values.setdefault(0, 0)
    pending, interior, external_pis = [decision['root']], set(), set()
    while pending:
        node = pending.pop()
        if node in values or node in interior or node in external_pis:
            continue
        if node not in aig.fanin_literals:
            if 0 < node <= aig.pi_count:
                external_pis.add(node)
                continue
            raise ValueError('Cut contains a node absent from original AIG.')
        interior.add(node)
        pending.extend(literal >> 1 for literal in aig.fanin_literals[node])
    if external_pis:
        return support_cut_truth(aig, decision, interior, external_pis)
    for node in sorted(interior):
        left, right = aig.fanin_literals[node]
        values[node] = (values[left >> 1] ^ (0xffff if left & 1 else 0)) & (values[right >> 1] ^ (0xffff if right & 1 else 0))
    if values[decision['root']] != (int(decision['truth']) & 0xffff):
        raise ContextProofRequired('Independent cut truth differs; original cut-leaf relationships require CEC.')
    return {'cut_semantics': 'STRUCTURAL_BOUNDARY', 'boolean_validator': 'EXHAUSTIVE_16_BIT_TRUTH'}


def score_catalogue(aig_path, partition, catalogue, checkpoint, cache_dir, batch_size=128):
    if batch_size <= 0:
        raise ValueError('batch_size must be positive.')
    library_path = ROOT / 'projects/culs/src/algorithms/rewrite_library.inc'
    canonical_paths = [ROOT / 'projects/litecut' / p for p in [
        'gate2a_candidate_inference.py', 'src/data_preprocess/feat_extraction_v3.py',
        'src/extract_cutlib_student/modeling.py', 'src/extract_cutlib_teacher/feature_branch.py',
        'src/validate/eval_prediction_accuracy.py']]
    identity = {'input': digest(aig_path), 'checkpoint': digest(checkpoint), 'library': digest(library_path),
                'canonical_code': [digest(p) for p in canonical_paths], 'adapter': digest(Path(__file__)),
                'bridge': digest(ROOT / 'projects/culs/tools/gate3e_run.py'),
                'environment': {'torch': str(torch.__version__), 'numpy': np.__version__, 'networkx': nx.__version__},
                'batch_size': batch_size,
                'truth_validator': digest(ROOT / 'global_timing/truth.py'),
                'context_validator': digest(ROOT / 'global_timing/cut_validation.py'),
                'decisions': [[d['root'], d['cut_slot'], d['truth'], d['cut_leaves'], d['decision_owner_region'],
                               d['truth_class'], d['candidate_subgraph_indices']] for d in catalogue.decisions]}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / (key + '.json')
    if path.exists():
        payload = json.loads(path.read_text())
        if payload['identity'] != identity:
            raise ValueError('LiteCut cache identity mismatch.')
        return payload['scores'], {'cache_hit': True, 'path': str(path), 'key': key}
    predictor = legacy('gate3e_run').LiteCutPredictor(ROOT / 'projects/litecut', aig_path, checkpoint, library_path)
    if not np.isfinite(predictor.y_inv) or predictor.y_inv == 0:
        raise ValueError('Invalid Student output normalization.')
    rows, features = [], []
    for index, decision in enumerate(catalogue.decisions):
        cut_validation = decision.get('cut_validation') or verify_cut_truth(partition.aig, decision)
        options = [None] + decision['candidate_subgraph_indices']
        for choice, subgraph in enumerate(options):
            row = {'decision_index': index, 'choice_index': choice, 'subgraph_index': subgraph,
                   'region_id': decision['decision_owner_region'], 'original_cut_truth_verified': True,
                   'cut_validation': cut_validation}
            if subgraph is None:
                graph = original_cone(partition.aig, decision)
                row['representation'] = 'original_structural_cone_to_cut_or_PI'
            else:
                row.update(verify_library_truth(decision, subgraph, predictor.lib))
                record = dict(decision, target_root=decision['root'], subgraph_index=subgraph,
                              phase=predictor.lib['p_phases'][decision['truth']],
                              permutation=predictor.lib['perms4'][predictor.lib['p_perms'][decision['truth']]])
                graph, info = predictor.materialize(record, predictor.lib,
                                                     synthetic_base=partition.aig.maximum_variable + 1)
                row['replacement_dag_info'] = info
                row['representation'] = 'canonical_library_materialization'
            full, names = predictor.extractor.extract_all(graph)
            values = np.asarray(full[:19], dtype=np.float32)
            if names != predictor.feature_names or values.shape != (19,) or not np.isfinite(values).all():
                raise ValueError('Canonical feature extraction contract violated.')
            row['continuous_features_19'] = values.tolist()
            row['feature_hash'] = hashlib.sha256(values.tobytes()).hexdigest()
            rows.append(row)
            features.append(values)
    if features:
        matrix = np.stack(features)
        scaled = (matrix - predictor.x_mu) * predictor.x_inv
        if not np.isfinite(scaled).all():
            raise ValueError('Non-finite normalized Student input.')
        with torch.inference_mode():
            predicted = torch.cat([predictor.model(torch.from_numpy(scaled[start:start + batch_size])).prediction.reshape(-1)
                                   for start in range(0, len(matrix), batch_size)]).cpu().numpy()
        delays = np.expm1(np.clip(predicted.astype(np.float64) / predictor.y_inv + predictor.y_mu, -10, 10))
        if not np.isfinite(delays).all():
            raise ValueError('Non-finite local delay prediction.')
        for row, value in zip(rows, delays):
            row['predicted_local_abc_delay'] = float(value)
    payload = {'identity': identity, 'scores': rows, 'metric': 'local abc_delay proxy, not region/path/global timing',
               'batch_size': batch_size, 'torch_version': torch.__version__}
    save_json(path, payload)
    return rows, {'cache_hit': False, 'path': str(path), 'key': key}
