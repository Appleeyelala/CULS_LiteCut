"""Unit-AND structural levels for combinational AIGs. Node 0 is the constant."""
from dataclasses import dataclass


@dataclass
class Levels:
    level: list[int]
    required: list[int | None]
    depth: int

    def slack(self, node):
        req = self.required[node]
        return None if req is None else req - self.level[node]


def propagate_levels(aig):
    levels = [0] * (aig.maximum_variable + 1)
    # The binary AIG reader checks contiguous, topologically numbered ANDs.
    for node in range(aig.pi_count + 1, len(levels)):
        left, right = aig.fanin_literals[node]
        levels[node] = 1 + max(levels[left >> 1], levels[right >> 1])
    outputs = [lit >> 1 for lit in aig.po_literals]
    depth = max((levels[node] for node in outputs), default=0)
    required = [None] * len(levels)
    for node in outputs:
        required[node] = depth
    for node in range(len(levels) - 1, aig.pi_count, -1):
        if required[node] is None:
            continue
        bound = required[node] - 1
        for literal in aig.fanin_literals[node]:
            pred = literal >> 1
            if required[pred] is None or bound < required[pred]:
                required[pred] = bound
    return Levels(levels, required, depth)


def candidate_root_level(leaf_levels, depths):
    """Longest represented path. A missing depth denotes an unused leaf."""
    if len(leaf_levels) != len(depths):
        raise ValueError('Leaf levels and depths must share the ordered cut.')
    return max((level + depth for level, depth in zip(leaf_levels, depths)
                if depth is not None), default=0)
