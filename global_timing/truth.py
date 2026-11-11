"""Exact reduced ordered BDD checks for support-minimized rewrite cuts.

Unit fixtures are analytical Boolean functions, never timing experiment data.
Resource limits fail explicitly rather than asserting an unproved equivalence.
"""


class ContextProofRequired(ValueError):
    """Independent boundary variables cannot certify this cut; use real CEC."""


class BDD:
    def __init__(self, limit=200000):
        self.nodes = [None, None]  # constant false / true
        self.unique = {}
        self.and_cache = {}
        self.not_cache = {0: 1, 1: 0}
        self.limit = limit

    def node(self, variable, low, high):
        if low == high:
            return low
        key = variable, low, high
        if key not in self.unique:
            if len(self.nodes) >= self.limit:
                raise ContextProofRequired('Cut Boolean validation exceeded its explicit BDD node limit.')
            self.unique[key] = len(self.nodes)
            self.nodes.append(key)
        return self.unique[key]

    def negate(self, value):
        if value not in self.not_cache:
            variable, low, high = self.nodes[value]
            result = self.node(variable, self.negate(low), self.negate(high))
            self.not_cache[value] = result
            self.not_cache[result] = value
        return self.not_cache[value]

    def conjunction(self, left, right):
        left, right = min(left, right), max(left, right)
        if left == 0:
            return 0
        if left == 1 or left == right:
            return right
        if self.not_cache.get(left) == right:
            return 0
        key = left, right
        if key not in self.and_cache:
            if len(self.and_cache) >= self.limit * 4:
                raise ContextProofRequired('Cut Boolean validation exceeded its explicit BDD apply limit.')
            a, b = self.nodes[left], self.nodes[right]
            variable = min(a[0], b[0])
            low = self.conjunction(a[1] if a[0] == variable else left,
                                   b[1] if b[0] == variable else right)
            high = self.conjunction(a[2] if a[0] == variable else left,
                                    b[2] if b[0] == variable else right)
            self.and_cache[key] = self.node(variable, low, high)
        return self.and_cache[key]


def support_cut_truth(aig, decision, interior, external_pis):
    """Prove independence from omitted PIs and exact exported cut truth."""
    leaves = decision['cut_leaves']
    if len(external_pis) > 8 or len(interior) > 128:
        raise ContextProofRequired('Support-minimized cut requires original-context CEC, beyond small exact BDD scope.')
    order = leaves + sorted(external_pis)
    bdd = BDD()
    values = {node: bdd.node(index, 0, 1) for index, node in enumerate(order)}
    values.setdefault(0, 0)
    for node in sorted(interior):
        left, right = aig.fanin_literals[node]
        a, b = values[left >> 1], values[right >> 1]
        values[node] = bdd.conjunction(bdd.negate(a) if left & 1 else a,
                                       bdd.negate(b) if right & 1 else b)
    truth = int(decision['truth']) & 0xffff
    if len(leaves) == 3 and truth & 255 != truth >> 8:
        raise ValueError('Three-leaf truth depends on an absent fourth input.')

    def expected(index, assignment):
        if index == len(leaves):
            return (truth >> assignment) & 1
        return bdd.node(index, expected(index + 1, assignment),
                        expected(index + 1, assignment | (1 << index)))

    if values[decision['root']] != expected(0, 0):
        raise ContextProofRequired('Independent support check differs; prove with original cut-leaf relationships.')
    return {'cut_semantics': 'FUNCTIONAL_SUPPORT_MINIMIZED', 'boolean_validator': 'EXACT_ROBDD',
            'eliminated_external_pi_count': len(external_pis), 'bdd_nodes': len(bdd.nodes)}
