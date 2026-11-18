"""Verify printed gate connectivity and publish delay-only paired decisions."""
import argparse
import csv
import json
import math
from pathlib import Path
import re
from statistics import mean


def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def write(path, rows):
    with path.open('x', newline='') as stream:
        w = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def critical_arcs(folder):
    trace = json.loads((folder/'trace.json').read_text())
    timing = trace['timing']
    path = next(v for v in trace['paths'] if v['po_index'] == min(timing['critical_po_indices']))
    _, circuit, index, stage, _ = folder.name.split('_')
    # ABC write_verilog emits named input pins followed by the output pin.
    drivers, aliases = {}, {}
    for line in (folder/'mapped.v').read_text().splitlines():
        cell = re.match(r'\s*(\w+)\s+g\d+\(', line)
        if cell:
            ports = re.findall(r'\.(\w+)\((\w+)\)', line)
            out = ports[-1][1]
            assert out not in drivers
            drivers[out] = (cell[1], dict(ports[:-1]))
        assign = re.match(r'\s*assign\s+(\w+)\s*=\s*(\w+)\s*;', line)
        if assign:
            aliases[assign[1]] = assign[2]
    genlib = {}
    for line in (folder/'gates.genlib').read_text().splitlines():
        f = line.split()
        if f and f[0] == 'GATE':
            cell = f[1]
        if f and f[0] == 'PIN':
            genlib[cell, f[1]] = (float(f[5]), float(f[7]))
    net, = [n for n in drivers.keys() | aliases.keys()
            if re.fullmatch(r'po\d+', n) and int(n[2:]) == path['po_index']]
    rows = []
    for a in path['arcs']:
        while net in aliases:
            net = aliases[net]
        cell, ports = drivers[net]
        assert cell == a['cell'], (net, cell, a['cell'])
        pin = a['pin_name']
        inp = ports[pin]
        rows.append(dict(circuit=int(circuit), index=int(index), stage=stage,
                         pi=path['pi_index'], po=path['po_index'], step=len(path['arcs'])-a['sequence'],
                         cell=cell, pin=pin, inputs=','.join(f'{p}={v}' for p, v in ports.items()),
                         output_net=net, fanout=a['fanout'], input_rise=a['input_phase'],
                         output_rise=a['output_phase'], load_ff=a['output_load'], slew_ps=a['input_slew_ps'],
                         table_max_load_ff=a['table_max_load'], table_max_slew_ps=a['table_max_slew'],
                         fixed_genlib_arc=genlib[cell, pin][0 if a['output_phase'] else 1],
                         nominal_ps=a['nominal_ps'], load_only_ps=a['load_only_ps'], slew_only_ps=a['slew_only_ps'],
                         actual_arc_ps=a['arc_ps'], extra_ps=a['load_estimation_extra_ps']))
        net = inp
    assert net.startswith('pi') and int(net[2:]) == path['pi_index']
    assert math.isclose(sum(v['actual_arc_ps'] + v['extra_ps'] for v in rows), timing['delay_ps'],
                        abs_tol=0.003, rel_tol=2e-6)
    return list(reversed(rows))


def main(args):
    data = read(args.audit/'panel.csv')
    fan = {(int(r['circuit']), int(r['index']), int(r['factor']), float(r['alpha'])): float(r['score'])
           for r in read(args.fan/'predictions.csv')}
    assert len(data) == 96 and len(fan) == 324
    methods = ['native', 'initial_scl', 'assisted_scl', 'bank', 'bank_fanout', 'bank_fanout_litecut']
    scores, choices = [], []
    for i in sorted({int(r['circuit']) for r in data}):
        g = []
        for r in data:
            if int(r['circuit']) != i:
                continue
            k = int(r['index'])
            entry = dict(circuit=i, index=k, edits=int(r['edits']), final_ps=float(r['final_ps']),
                         area_ratio=float(r['area_ratio']))
            entry.update((m, float(r[m])) for m in methods[:3])
            for m, f, a in zip(methods[3:], (0,100,100), (0.,0.,.5)):
                entry[m] = fan[i, k, f, a]
            g.append(entry)
        assert {r['index'] for r in g} == set(range(1,9))
        scores.extend(g)
        fastest = min(r['final_ps'] for r in g)
        for m in methods:
            r = min(g, key=lambda row: (row[m], row['edits'], row['index']))
            choices.append(dict(circuit=i, method=m, index=r['index'], score=r[m], real_ps=r['final_ps'],
                                best_ps=fastest, regret_pct=100*(r['final_ps']/fastest-1),
                                top1=math.isclose(r['final_ps'], fastest, rel_tol=2e-6, abs_tol=2e-5),
                                area_ratio=r['area_ratio']))
    summary = []
    for m in methods:
        c = [r for r in choices if r['method'] == m]
        summary.append(dict(method=m, circuits=len(c), mean_regret_pct=mean(r['regret_pct'] for r in c),
                            top1=sum(r['top1'] for r in c), over_area_105=sum(r['area_ratio']>1.05 for r in c)))
    arcs = [a for p in sorted(args.audit.glob('case_*/trace.json')) for a in critical_arcs(p.parent)]
    assert len({(a['circuit'],a['index'],a['stage']) for a in arcs}) == 12
    for name, rows in [('critical_arcs',arcs),('scores',scores),('choices',choices),('summary',summary)]:
        write(args.output/(name+'.csv'),rows)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for n in ('audit','fan','output'):
        p.add_argument('--'+n, type=Path, required=True)
    main(p.parse_args())
