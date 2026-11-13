"""Candidate-aware original-structure reuse constraints before real replay.

This conservative preflight is not a prediction of sequential gain or CEC.
Only the native complete-acceptance check can certify actual installation.
"""
from .boolean_candidates import parse_library


class CandidateFeasibility:
    def __init__(self, aig, catalogue, library):
        library = parse_library(library) if not isinstance(library, dict) else library
        self.catalogue = catalogue
        self.static = [set() for _ in catalogue.decisions]
        for a, b in catalogue.conflicts:
            self.static[a].add(b)
            self.static[b].add(a)
        removers = {}
        for index, decision in enumerate(catalogue.decisions):
            for node in decision['removable_nodes']:
                removers.setdefault(node, set()).add(index)
        original = {}
        for node, literals in aig.fanin_literals.items():
            original.setdefault(tuple(sorted(literals)), []).append(node)
        self.blocked, self.references, self.forbidden = [], [], []
        programs = {}
        ambiguities = 0
        for i, decision in enumerate(catalogue.decisions):
            blocked, references, forbidden = [None], [set()], [set()]
            truth = int(decision['truth']) & 0xffff
            phase = library['p_phases'][truth]
            permutation = library['perms4'][library['p_perms'][truth]]
            leaves = list(decision['cut_leaves'])
            if len(leaves) == 3:
                leaves.append(0)
            if len(leaves) != 4:
                raise ValueError('Reuse analysis requires the verified three/four-input domain.')
            data = library['classes'][decision['truth_class']]
            for subgraph in decision['candidate_subgraph_indices']:
                key = (decision['truth_class'], subgraph)
                if key not in programs:
                    root = data['roots'][subgraph]
                    used, work = set(), [root]
                    while work:
                        v = work.pop()
                        if v < 4 or v in used:
                            continue
                        used.add(v)
                        a, b, _, _ = data['fanins'][v - 4]
                        work.extend((a, b))
                    programs[key] = (root, [(v, data['fanins'][v - 4]) for v in sorted(used)])
                root, program = programs[key]
                values = [(leaves[permutation[k]] << 1) ^ ((phase >> k) & 1) for k in range(4)]
                local, reused = {}, set()
                for v, (a, b, ca, cb) in program:
                    left, right = values[a] ^ ca, values[b] ^ cb
                    if left == 0 or right == 0 or left == (right ^ 1):
                        literal = 0
                    elif left == 1 or left == right:
                        literal = right
                    elif right == 1:
                        literal = left
                    else:
                        pair = tuple(sorted((left, right)))
                        if pair in original:
                            matches = original[pair]
                            reused.update(matches)
                            ambiguities += len(matches) > 1
                            literal = min(matches) << 1
                        else:
                            if pair not in local:
                                local[pair] = (aig.maximum_variable + len(local) + 1) << 1
                            literal = local[pair]
                    while len(values) <= v:
                        values.append(None)
                    values[v] = literal
                # Wire/constant roots may have no library gates at all.
                if values[root] is None:
                    raise ValueError('Library program is not topologically ordered.')
                nodes = set(reused)
                if (values[root] >> 1) in aig.fanin_literals:
                    nodes.add(values[root] >> 1)
                # Any original descendant reused by the candidate can create a
                # transitive self reference. Original binary AIG IDs are topo.
                work, seen, cycle = [n for n in nodes if n >= decision['root']], set(), False
                while work:
                    node = work.pop()
                    if node == decision['root']:
                        cycle = True
                        break
                    if node in seen or node < decision['root'] or node not in aig.fanin_literals:
                        continue
                    seen.add(node)
                    work.extend(lit >> 1 for lit in aig.fanin_literals[node])
                blocked.append('ORIGINAL_ROOT_OR_DESCENDANT_REUSED' if cycle else None)
                references.append(nodes)
                forbidden.append(set().union(*(removers.get(n, set()) for n in nodes)) - {i} if nodes else set())
            self.blocked.append(blocked)
            self.references.append(references)
            self.forbidden.append(forbidden)
        self.ambiguous_hash_hits = ambiguities

    def allowed(self):
        return [[reason is None for reason in row] for row in self.blocked]

    def validate(self, choices):
        self.catalogue.validate(choices)
        active = {i for i, c in enumerate(choices) if c}
        for i in active:
            c = choices[i]
            if self.blocked[i][c] or self.forbidden[i][c] & (active - {i}):
                raise ValueError('Candidate reuses original structure removed by this combination.')

    def project(self, anchor, edits, maximum):
        self.validate(anchor)
        result = list(anchor)
        active = {i for i, c in enumerate(result) if c}
        changes = 0
        for gain, index, choice in sorted(edits, key=lambda e: (-e[0], e[1], e[2])):
            if changes >= maximum:
                break
            if choice == result[index] or gain <= 0:
                continue
            others = active - {index}
            if self.blocked[index][choice]:
                continue
            if choice and (self.static[index] & others or self.forbidden[index][choice] & others
                           or any(index in self.forbidden[j][result[j]] for j in others)):
                continue
            result[index] = choice
            active.discard(index)
            if choice:
                active.add(index)
            changes += 1
        self.validate(result)
        return result

    def payload(self):
        return {'blocked_choices': [{'decision_index': i, 'choice_index': c, 'reason': reason}
                                   for i, row in enumerate(self.blocked) for c, reason in enumerate(row) if reason],
                'candidate_reuse': [{'decision_index': i, 'choice_index': c,
                                    'original_and_references': sorted(refs),
                                    'incompatible_active_decisions': sorted(self.forbidden[i][c])}
                                   for i, row in enumerate(self.references) for c, refs in enumerate(row) if refs],
                'ambiguous_original_hash_hits': self.ambiguous_hash_hits,
                'scope': 'Conservative original structural hashing, not complete dynamic gain/cycle prediction; actual full acceptance and CEC remain mandatory.'}
