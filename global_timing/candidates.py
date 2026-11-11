"""Decision/candidate identities, conservative conflicts and actual replay."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .common import ToolError, legacy, read_jsonl, run_tool, save_json, tool_path


@dataclass
class Catalogue:
    decisions: list[dict]
    conflicts: list[tuple[int, int]]
    region_decisions: dict[int, list[int]]

    def validate(self, choices: list[int]):
        if len(choices) != len(self.decisions):
            raise ValueError('One selection per decision is required.')
        for choice, decision in zip(choices, self.decisions):
            if type(choice) is not int or choice < 0 or choice > decision['candidate_count']:
                raise ValueError('Candidate index is outside its decision domain.')
        for left, right in self.conflicts:
            if choices[left] and choices[right]:
                raise ValueError(f'Conflicting decisions selected: {left}, {right}')

    def project(self, scores: list[list[float]]) -> list[int]:
        """Deterministic greedy projection; no-op remains feasible for all groups.

        Scores are selection logits, not candidate IDs or measured delays.
        The projection policy is explicit and all discarded choices are logged
        by callers; it does not claim to solve the discrete subproblem exactly.
        """
        import math
        if len(scores) != len(self.decisions):
            raise ValueError('Selection score shape mismatch.')
        proposed = []
        for i, (values, decision) in enumerate(zip(scores, self.decisions)):
            if len(values) != decision['candidate_count'] + 1 or not all(math.isfinite(x) for x in values):
                raise ValueError('Non-finite or malformed selection scores.')
            choice = max(range(len(values)), key=lambda c: (values[c], -c))
            if choice:
                proposed.append((values[choice] - values[0], i, choice))
        neighbors = {i: set() for i in range(len(scores))}
        for left, right in self.conflicts:
            neighbors[left].add(right)
            neighbors[right].add(left)
        active = set()
        result = [0] * len(scores)
        for _, index, choice in sorted(proposed, key=lambda row: (-row[0], row[1])):
            if not neighbors[index] & active:
                result[index] = choice
                active.add(index)
        self.validate(result)
        return result

    def requested(self, choices):
        self.validate(choices)
        # Deterministic ascending original-root order. Conservative conflicts
        # exclude leaf deletion in either direction and removable overlap.
        return [(d['root'], d['cut_slot'], d['candidate_subgraph_indices'][c - 1])
                for d, c in zip(self.decisions, choices) if c]


def conflict_reason(left, right):
    if left['root'] == right['root']:
        return 'SAME_ROOT'
    a, b = set(left['removable_nodes']), set(right['removable_nodes'])
    if a & b:
        return 'REMOVABLE_OVERLAP'
    if a & set(right['cut_leaves']) or b & set(left['cut_leaves']):
        return 'CUT_LEAF_DELETION'
    return None


def catalogue_from_raw(rows, partition):
    normalize = legacy('gate3c_analyze').normalize_decision
    decisions = []
    for row in sorted(rows, key=lambda r: (r['root'], r['cut_slot'])):
        decision = normalize(row, f'd{len(decisions)}', partition.owner, set(range(len(partition.regions))))
        if decision['candidate_count'] == 0:
            continue
        if decision['classification'] != 'INTRA_REGION':
            raise ValueError('Cross-region removable cone requires an explicit multi-owner policy.')
        decision['noop'] = {'choice_index': 0, 'subgraph_index': None, 'semantics': 'ORIGINAL_AIG'}
        decisions.append(decision)
    conflicts = []
    by_region = {i: [] for i in range(len(partition.regions))}
    # Generate possible conflict pairs through node incidence, avoiding N^2
    # comparison on large candidate collections.
    removers, leaf_users = {}, {}
    pairs = set()
    for i, decision in enumerate(decisions):
        for node in decision['removable_nodes']:
            removers.setdefault(node, []).append(i)
        for node in decision['cut_leaves']:
            leaf_users.setdefault(node, []).append(i)
        by_region[decision['decision_owner_region']].append(i)
    # Shared leaves alone are not conflicts. Expanding all leaf-user pairs
    # would be quadratic for high-fanout PIs on large benchmark designs.
    for node, deleting in removers.items():
        for left in deleting:
            for right in deleting + leaf_users.get(node, []):
                if left != right:
                    pairs.add((min(left, right), max(left, right)))
    for left, right in sorted(pairs):
        if conflict_reason(decisions[left], decisions[right]):
            conflicts.append((left, right))
    return Catalogue(decisions, conflicts, by_region)


def enumerate_candidates(gpuls, aig, partition, output, timeout=300):
    output.mkdir(parents=True, exist_ok=False)
    raw = output / 'raw'
    run_tool([gpuls, '-c', f'read {tool_path(aig)}; gate3c_rewrite_decisions_dump -o {tool_path(raw)}'],
             output, 'enumeration', timeout)
    rows = read_jsonl(raw / 'gpu_rewrite_cuts.jsonl')
    catalogue = catalogue_from_raw(rows, partition)
    save_json(output / 'catalogue.json', {
        'decisions': catalogue.decisions, 'conflicts': [
            {'left': a, 'right': b, 'reason': conflict_reason(catalogue.decisions[a], catalogue.decisions[b])}
            for a, b in catalogue.conflicts], 'region_decisions': catalogue.region_decisions,
        'legal_cut_count': len(rows), 'executable_decisions': len(catalogue.decisions),
        'candidate_count': sum(d['candidate_count'] for d in catalogue.decisions),
        'policy': 'Positive node gain GPU masks; conservative static conflicts; exact replay acceptance required.'})
    return catalogue


def replay(gpuls, aig, owner_csv, catalogue, choices, output, timeout=300):
    requested = catalogue.requested(choices)  # reject conflicts BEFORE calling GPU
    output.mkdir(parents=True, exist_ok=False)
    config = output / 'choices.txt'
    config.write_text(''.join(f'choice {r} {c} {s}\n' for r, c, s in requested) or 'choice -1 -1 -1\n')
    result = {'choices': choices, 'requested': requested, 'status': 'RUNNING', 'order': 'ascending_original_root'}
    try:
        raw = output / 'raw'
        run_tool([gpuls, '-c', f'read {tool_path(aig)}; gate3e_execution_replay -o {tool_path(raw)} '
                  f'--region-map {tool_path(owner_csv)} --config {tool_path(config)}'], output, 'rewrite', timeout)
        trace = read_jsonl(raw / 'traces/choice.jsonl')
        selected = {(r['root'], r['cut_slot'], r['selected_subgraph']): r for r in trace if r['preselected']}
        if set(selected) != set(requested):
            raise ToolError('Replay did not preselect exactly the requested candidates.')
        rejected = {str(key): row['final_status'] for key, row in selected.items() if row['final_status'] != 'ACCEPTED'}
        result.update(applied_rewrites=sum(row['final_status'] == 'ACCEPTED' for row in trace), rejected=rejected,
                      rejected_requests=[{'root': key[0], 'cut_slot': key[1], 'subgraph_index': key[2],
                                          'status': row['final_status']}
                                         for key, row in selected.items() if row['final_status'] != 'ACCEPTED'])
        if rejected:
            raise ToolError('Candidate combination was not fully accepted; refusing a substituted result.')
        path = raw / 'aigs/choice.aig'
        if not path.is_file():
            raise ToolError('Replay did not produce the requested AIG.')
        result.update(status='PASS', aig=str(path))
    except Exception as exc:
        result.update(status='FAIL', error=str(exc))
        raise
    finally:
        save_json(output / 'replay.json', result)
    return path, result
