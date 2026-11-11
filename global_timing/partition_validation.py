"""Independent structural certificates for strict maximal MFFC partitions.

Unlike construction, this check performs no reference-count dereferencing and
does not propagate a separate port map at every internal node. It checks fanout
closure and computes longest boundary-to-root distances backwards from roots.

Root reachability and internal fanout closure place every member in the root's
standard MFFC. Every outside boundary AND retains an outside use, so none can
be absorbed even after removing the entire region. The cone is therefore exact.
Any member-rooted MFFC stays inside that same closed region; a disjoint complete
cover by exact root cones consequently consists of inclusion-maximal MFFCs.
"""
from collections import Counter


def validate_partition(partition):
    aig, regions, owner = partition.aig, partition.regions, partition.owner
    nodes = set(aig.fanin_literals)
    if nodes != set(aig.and_nodes) or len(nodes) != aig.and_count:
        raise ValueError('AIG AND inventory disagrees with its fanins/header.')
    if len(partition.transfers) != len(regions):
        raise ValueError('Missing region transfer maps.')
    membership = {}
    for rid, region in enumerate(regions):
        members = region['internal_nodes']
        if (region['region_id'] != rid or region['root'] not in members or
                region['cone_size'] != len(members) or not members):
            raise ValueError('Invalid region identity, root or size.')
        for node in members:
            if node not in nodes or node in membership:
                raise ValueError('Invalid or repeated AND ownership.')
            membership[node] = rid
    if set(membership) != nodes or membership != owner:
        raise ValueError('Incomplete or inconsistent owner coverage.')

    uses = Counter(lit >> 1 for pair in aig.fanin_literals.values() for lit in pair)
    uses.update(lit >> 1 for lit in aig.po_literals)
    boundary_uses, expected_edges = Counter(), set()
    for target, pair in aig.fanin_literals.items():
        rid = owner[target]
        for lit in pair:
            source = lit >> 1
            if source >= target or source < 0:
                raise ValueError('AIG literals are not topologically ordered.')
            if source in nodes and owner[source] != rid:
                if regions[owner[source]]['root'] != source:
                    raise ValueError('Non-root member has an external AND fanout.')
                boundary_uses[(source, rid)] += 1
                expected_edges.add((owner[source], rid))
    for lit in aig.po_literals:
        source = lit >> 1
        if source in nodes and regions[owner[source]]['root'] != source:
            raise ValueError('Non-root member has a PO fanout.')
    for (source, _), inside_uses in boundary_uses.items():
        if inside_uses >= uses[source]:
            raise ValueError('Boundary AND can be absorbed: region is not a full MFFC.')
    if (len(partition.edges) != len(expected_edges) or
            set(map(tuple, partition.edges)) != expected_edges):
        raise ValueError('Region DAG edges disagree with the original AIG.')
    if sorted(partition.order) != list(range(len(regions))):
        raise ValueError('Region topological order is not a permutation.')
    position = {rid: i for i, rid in enumerate(partition.order)}
    if any(position[left] >= position[right] for left, right in expected_edges):
        raise ValueError('Region DAG order contains a backward edge.')

    for rid, region in enumerate(regions):
        distance = {region['root']: 0}
        boundary = {}
        for node in sorted(region['internal_nodes'], reverse=True):
            if node not in distance:
                raise ValueError('Region member has no internal path to its root.')
            for lit in aig.fanin_literals[node]:
                source = lit >> 1
                destination = distance if owner.get(source) == rid else boundary
                destination[source] = max(destination.get(source, 0), distance[node] + 1)
        if boundary != partition.transfers[rid]:
            raise ValueError('Boundary-to-root AND distances disagree with reverse traversal.')

    # A separate node-level recurrence checks complete PI/constant-to-PO depth.
    direct = {i: 0 for i in range(aig.pi_count + 1)}
    for node, pair in sorted(aig.fanin_literals.items()):
        direct[node] = 1 + max(direct[lit >> 1] for lit in pair)
    region_depth = {}
    for rid in partition.order:
        region_depth[rid] = max((region_depth[owner[port]] if port in owner else 0) + length
                                for port, length in partition.transfers[rid].items())
        if region_depth[rid] != direct[regions[rid]['root']]:
            raise ValueError('Region propagation differs from original node-level AND depth.')
    po_depths = [direct[lit >> 1] for lit in aig.po_literals]
    return {'status': 'PASS', 'and_count': len(nodes), 'region_count': len(regions),
            'edge_count': len(expected_edges), 'covered_and_nodes': len(membership),
            'boundary_port_count': sum(len(ports) for ports in partition.transfers),
            'po_count': len(po_depths), 'maximum_po_and_depth': max(po_depths, default=0),
            'checks': ['unique_complete_ownership', 'internal_fanout_closure', 'unabsorbable_boundary',
                       'root_reachability', 'exact_quotient_edges', 'acyclic_region_order',
                       'every_port_reverse_distance', 'node_vs_region_and_depth'],
            'scope': 'Structural maximal MFFC/AND-depth certificate only; no mapped timing or candidate execution.'}
