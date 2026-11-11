"""Validate optional read-only Liberty arrival traces in mapped ID space."""
import math
import struct

from .common import ToolError


def _f32(value):
    return struct.unpack('f', struct.pack('f', value))[0]


def _same(left, right):
    return math.isclose(left, right, rel_tol=2e-6, abs_tol=2e-5)


def parse_paths(text, timing, pi_count):
    paths, metadata = {}, []
    for line in text.splitlines():
        fields = line.split()
        if not fields or not fields[0].startswith('GT_PATH_'):
            continue
        tag = fields[0]
        try:
            if tag == 'GT_PATH_ERROR':
                raise ToolError('Liberty arrival traceback failed: ' + line)
            if tag == 'GT_PATH_META':
                if len(fields) != 3:
                    raise ValueError('Malformed path metadata.')
                metadata.append((float(fields[1]), int(fields[2])))
                continue
            po = int(fields[1])
            if tag == 'GT_PATH_BEGIN':
                if len(fields) != 4 or po in paths:
                    raise ValueError('Malformed or duplicate path start.')
                paths[po] = {'po_index': po, 'output_phase': int(fields[2]), 'arrival_ps': float(fields[3]),
                             'nodes': [], 'arcs': []}
            elif tag == 'GT_PATH_NODE':
                if len(fields) != 8:
                    raise ValueError('Malformed path node.')
                paths[po]['nodes'].append(dict(zip(['sequence', 'node_id', 'phase'], map(int, fields[2:5])),
                                               arrival_ps=float(fields[5]), kind=fields[6], cell=fields[7]))
            elif tag == 'GT_PATH_ARC':
                if len(fields) != 14:
                    raise ValueError('Malformed path arc.')
                row = dict(zip(['sequence', 'node_id', 'fanin_id', 'pin', 'input_phase', 'output_phase'],
                               map(int, fields[2:8])))
                row.update(zip(['input_arrival_ps', 'arc_ps', 'load_estimation_extra_ps', 'output_arrival_ps',
                                'input_slew_ps', 'output_load'], map(float, fields[8:])))
                paths[po]['arcs'].append(row)
            elif tag == 'GT_PATH_END':
                if len(fields) != 4 or 'pi_index' in paths[po]:
                    raise ValueError('Malformed or duplicate path end.')
                paths[po].update(declared_nodes=int(fields[2]), pi_index=int(fields[3]))
            else:
                raise ValueError('Unknown path record.')
        except (ValueError, KeyError, IndexError) as exc:
            raise ToolError(f'Invalid Liberty path record: {line}') from exc
    count = len(timing['po_arrivals_ps'])
    if metadata != [(0.95, count)]:
        raise ToolError('Expected one complete near-critical path report.')
    threshold = .95 * timing['delay_ps']
    required = {i for i, value in enumerate(timing['po_arrivals_ps']) if value > threshold + 2e-5}
    allowed = {i for i, value in enumerate(timing['po_arrivals_ps']) if value >= threshold - 2e-5}
    if not required <= paths.keys() or not paths.keys() <= allowed or not set(timing['critical_po_indices']) <= paths.keys():
        raise ToolError('Path coverage does not match critical and near-critical PO arrivals.')
    for po, path in paths.items():
        nodes, arcs = path['nodes'], path['arcs']
        if path.get('declared_nodes') != len(nodes) or not nodes or len(arcs) != len(nodes) - 1:
            raise ToolError('Incomplete arrival path.')
        if path['output_phase'] not in (0, 1) or nodes[0]['phase'] != path['output_phase']:
            raise ToolError('Invalid endpoint transition.')
        if not _same(path['arrival_ps'], timing['po_arrivals_ps'][po]) or not _same(nodes[0]['arrival_ps'], path['arrival_ps']):
            raise ToolError('Path endpoint does not reconstruct the measured PO arrival.')
        if len({node['node_id'] for node in nodes}) != len(nodes):
            raise ToolError('Cycle in arrival trace.')
        for index, node in enumerate(nodes):
            if node['sequence'] != index or node['node_id'] < 0 or node['phase'] not in (0, 1) or not math.isfinite(node['arrival_ps']):
                raise ToolError('Invalid path node identity, transition or arrival.')
        terminal = nodes[-1]
        if (terminal['kind'] == 'PI' and not 0 <= path['pi_index'] < pi_count) or (
                terminal['kind'] == 'CONST' and path['pi_index'] != -1) or terminal['kind'] not in {'PI', 'CONST'}:
            raise ToolError('Path does not terminate at a valid PI or constant.')
        for index, (arc, node, fanin) in enumerate(zip(arcs, nodes, nodes[1:])):
            if (arc['sequence'], arc['node_id'], arc['fanin_id'], arc['output_phase'], arc['input_phase']) != (
                    index, node['node_id'], fanin['node_id'], node['phase'], fanin['phase']) or arc['pin'] < 0:
                raise ToolError('Broken path arc or transition chain.')
            numbers = [arc[key] for key in ['input_arrival_ps', 'arc_ps', 'load_estimation_extra_ps',
                                          'output_arrival_ps', 'input_slew_ps', 'output_load']]
            if not all(math.isfinite(value) for value in numbers):
                raise ToolError('Non-finite path arc.')
            reconstructed = _f32(_f32(_f32(arc['input_arrival_ps']) + _f32(arc['arc_ps'])) + _f32(arc['load_estimation_extra_ps']))
            if not all([_same(arc['input_arrival_ps'], fanin['arrival_ps']),
                        _same(arc['output_arrival_ps'], node['arrival_ps']), _same(reconstructed, node['arrival_ps'])]):
                raise ToolError('Liberty path arc arithmetic does not reconstruct arrival.')
        path['critical'] = po in timing['critical_po_indices']
        path['arrival_reconstruction'] = 'PASS'
    return {'paths': [paths[index] for index in sorted(paths)], 'near_critical_ratio': .95,
            'scope': 'One maximizing transition path per PO within 5% of global delay. Mapped IDs are local; no cross-map logical correspondence is inferred.'}
