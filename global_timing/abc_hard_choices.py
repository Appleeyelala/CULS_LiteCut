"""Prove structural correspondence from candidate literals to a serialized hard replay."""
from collections import Counter, defaultdict


class StructuralIds:
    """Exact commutative AIG interning; literal complement stays on each edge."""
    def __init__(self, pi_count):
        self.first = pi_count + 1
        self.gates = {}

    def gate(self, a, b):
        if a == 0 or b == 0 or a == (b ^ 1):
            return 0
        if a == 1 or a == b:
            return b
        if b == 1:
            return a
        pair = (min(a,b), max(a,b))
        if pair not in self.gates:
            self.gates[pair] = 2 * (self.first + len(self.gates))
        return self.gates[pair]

    def graph(self, graph):
        keys = [2*i for i in range(graph.pi_count+1)]
        for n,(a,b) in sorted(graph.fanin_literals.items()):
            if n != len(keys):
                raise ValueError("Non-dense AIGER")
            keys.append(self.gate(keys[a//2] ^ (a&1), keys[b//2] ^ (b&1)))
        return keys


def hard_annotations(original, installed, program, options):
    if (original.pi_count, original.po_count) != (installed.pi_count, installed.po_count):
        raise ValueError("Hard replay changed interface")
    ids = StructuralIds(original.pi_count)
    baseline = set(k//2 for k in ids.graph(original))
    gates, overrides = {}, {}
    for line in program:
        fields = line.split()
        if fields[0] == "g":
            gates[original.maximum_variable + 1 + len(gates)] = tuple(map(int, fields[1:]))
        else:
            root, lit = map(int, fields[3:5])
            overrides[root//2] = lit
    keys = {n:2*n for n in range(original.pi_count+1)}

    def resolve(lit):
        n = lit//2
        if n not in keys:
            left, right = gates[n]
            keys[n] = ids.gate(resolve(left), resolve(right))
        return keys[n] ^ (lit & 1)

    for n,(a,b) in sorted(original.fanin_literals.items()):
        keys[n] = resolve(overrides[n]) if n in overrides else ids.gate(resolve(a), resolve(b))
    actual = ids.graph(installed)
    if [resolve(l) for l in original.po_literals] != [actual[l//2] ^ (l&1) for l in installed.po_literals]:
        raise ValueError("Candidate-composed and installed signed PO structures differ")
    locations = defaultdict(list)
    for n, key in enumerate(actual):
        locations[key//2].append(n)
    multiplicity = Counter(keys[o["original_root"]]//2 for o in options)
    tagged = []
    for o in options:
        key = keys[o["original_root"]]
        places = locations[key//2]
        unique = (key//2 not in baseline and len(places) == 1 and multiplicity[key//2] == 1
                  and places[0] in installed.fanin_literals and o["ratio_valid"])
        tagged.append(dict(o, installed_node=places[0] if unique else None,
                           installed_polarity=((actual[places[0]] ^ key) & 1) if unique else None))
    return tagged
