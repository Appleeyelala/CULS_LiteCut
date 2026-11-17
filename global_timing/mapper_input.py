"""Deterministic AIGER encoding, without Boolean rewriting or structural merging."""
import hashlib
from types import SimpleNamespace


def canonical_graph(aig):
    hashes = {i: hashlib.sha256(f'PI:{i}'.encode()).digest() for i in range(aig.pi_count + 1)}
    hashes[0] = hashlib.sha256(b'CONST0').digest()
    ordered = {}
    for node, pair in sorted(aig.fanin_literals.items()):
        ordered[node] = tuple(sorted(pair, key=lambda l: (hashes[l >> 1], l & 1)))
        hashes[node] = hashlib.sha256(b'AND' + b''.join(hashes[l >> 1] + bytes([l & 1])
                                                     for l in ordered[node])).digest()
    translated = {i: 2 * i for i in range(aig.pi_count + 1)}
    fanins = {}
    # PO order and PI identity are invariant. Retain even disconnected gates.
    roots = [l >> 1 for l in aig.po_literals] + sorted(ordered, key=lambda n: hashes[n])
    for root in roots:
        stack = [(root, False)]
        while stack:
            node, done = stack.pop()
            if node in translated:
                continue
            if not done:
                stack.append((node, True))
                stack.extend((l >> 1, False) for l in reversed(ordered[node]))
            else:
                pair = tuple(sorted((translated[l >> 1] ^ (l & 1) for l in ordered[node]), reverse=True))
                target = aig.pi_count + len(fanins) + 1
                translated[node] = target * 2
                fanins[target] = pair
    outputs = [translated[l >> 1] ^ (l & 1) for l in aig.po_literals]
    return SimpleNamespace(pi_count=aig.pi_count, po_count=len(outputs), po_literals=outputs,
                           and_count=len(fanins), maximum_variable=aig.pi_count + len(fanins),
                           fanin_literals=fanins, and_nodes=set(fanins))


def aiger_bytes(aig):
    data = bytearray(f'aig {aig.pi_count + len(aig.fanin_literals)} {aig.pi_count} 0 '
                     f'{len(aig.po_literals)} {len(aig.fanin_literals)}\n'.encode())
    data.extend(''.join(f'{l}\n' for l in aig.po_literals).encode())
    def varint(value):
        if value < 0:
            raise ValueError('AIGER requires topological literals')
        while value >= 128:
            data.append((value & 127) | 128); value >>= 7
        data.append(value)
    for node, pair in sorted(aig.fanin_literals.items()):
        a, b = sorted(pair, reverse=True)
        varint(2 * node - a); varint(a - b)
    data.extend(b'c\nCULS deterministic mapper input v1\n')
    return bytes(data)


def normalize(aig, path):
    graph = canonical_graph(aig)
    path.write_bytes(aiger_bytes(graph))
    return graph
