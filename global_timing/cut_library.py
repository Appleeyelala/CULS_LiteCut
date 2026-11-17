"""Small combinational Liberty matcher. No ABC calls or fitted parameters.

The supported subset is explicit: <=4 inputs, a single output, unconditional
combinational NLDM arcs, input_net_transition / total_output_net_capacitance.
Tables are interpolated at fixed 20 ps slew and estimated load, with clamping.
"""
from dataclasses import dataclass
from functools import lru_cache
from itertools import permutations
import re
import statistics

import numpy as np


def blocks(text, kind):
    pattern = re.compile(r'\b' + re.escape(kind) + r'\s*\(([^)]*)\)\s*\{')
    for match in pattern.finditer(text):
        depth, end = 1, match.end()
        quoted = False
        while depth:
            char = text[end]
            if char == '"' and text[end-1] != '\\': quoted = not quoted
            if not quoted:
                depth += (char == '{') - (char == '}')
            end += 1
        yield match.group(1).strip().strip('"'), text[match.end():end-1]


def attr(text, key, default=None):
    match = re.search(r'\b' + key + r'\s*:\s*("[^"]*"|[^;]+);', text)
    return match.group(1).strip().strip('"') if match else default


def numbers(text, key):
    match = re.search(r'\b' + key + r'\s*\((.*?)\)\s*;', text, re.S)
    if not match: raise ValueError('Missing Liberty array: ' + key)
    return [float(v) for v in re.findall(r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?', match.group(1))]


def function_truth(expression, pins):
    """Small Boolean parser; never eval arbitrary Liberty expressions."""
    tokens = re.findall(r'[A-Za-z_][A-Za-z_0-9]*|[01!~&*+|^()\']', expression)
    if ''.join(tokens) != re.sub(r'\s+', '', expression): raise ValueError(expression)
    mask = (1 << (1 << len(pins))) - 1
    values = {p: sum(((x >> i) & 1) << x for x in range(1 << len(pins))) for i,p in enumerate(pins)}
    pos = 0
    def primary():
        nonlocal pos
        t = tokens[pos]; pos += 1
        if t in ('!', '~'): v = primary() ^ mask
        elif t == '(':
            v = expr(0)
            if tokens[pos] != ')': raise ValueError(expression)
            pos += 1
        else: v = int(t) * mask if t in ('0','1') else values[t]
        while pos < len(tokens) and tokens[pos] == "'": v ^= mask; pos += 1
        return v
    precedence = {'|':0, '+':0, '^':1, '&':2, '*':2}
    def expr(level):
        nonlocal pos
        if level == 3: return primary()
        left = expr(level+1)
        while pos < len(tokens) and precedence.get(tokens[pos]) == level:
            op = tokens[pos]; pos += 1; right = expr(level+1)
            left = left & right if level == 2 else left ^ right if level == 1 else left | right
        return left
    result = expr(0)
    if pos != len(tokens): raise ValueError(expression)
    return result


@dataclass
class Cell:
    name: str
    pins: tuple
    truth: int
    caps: tuple
    senses: tuple
    tables: tuple


class CutLibrary:
    def __init__(self, path, max_matches=4, slew_ps=20.):
        text = re.sub(r'/\*.*?\*/', '', path.read_text(), flags=re.S)
        time_unit = attr(text, 'time_unit')
        m = re.fullmatch(r'([\d.]+)(ps|ns|fs)', time_unit or '')
        if not m: raise ValueError('Unsupported Liberty time unit')
        self.time_ps = float(m[1]) * {'ps':1., 'ns':1000., 'fs':.001}[m[2]]
        self.slew_ps, self.max_matches = slew_ps, max_matches
        self.cells, self.excluded = [], []
        templates = dict(blocks(text, 'lu_table_template'))
        for name, body in blocks(text, 'cell'):
            try:
                pins = dict(blocks(body, 'pin'))
                inputs = tuple(p for p,b in pins.items() if attr(b,'direction') == 'input')
                outputs = [b for b in pins.values() if attr(b,'direction') == 'output']
                if len(outputs) != 1 or not 1 <= len(inputs) <= 4 or list(blocks(body,'ff')) or list(blocks(body,'latch')):
                    raise ValueError('not a single-output <=4-input combinational cell')
                truth = function_truth(attr(outputs[0], 'function'), inputs)
                caps = tuple(float(attr(pins[p],'capacitance')) for p in inputs)
                arcs = {}
                for _, arc in blocks(outputs[0], 'timing'):
                    pin = attr(arc, 'related_pin')
                    if pin not in inputs: continue
                    if attr(arc,'when') is not None or attr(arc,'timing_type','combinational') != 'combinational':
                        raise ValueError('conditional/sequential arc')
                    if pin in arcs: raise ValueError('multiple arcs per pin')
                    sense = attr(arc, 'timing_sense')
                    if sense not in ('positive_unate','negative_unate','non_unate'): raise ValueError('sense')
                    tables = []
                    for kind in ('cell_rise','cell_fall'):
                        template, table = next(blocks(arc,kind))
                        spec = templates[template]
                        if (attr(spec,'variable_1'), attr(spec,'variable_2')) != ('input_net_transition','total_output_net_capacitance'):
                            raise ValueError('table axes')
                        xs, ys = numbers(table,'index_1'), numbers(table,'index_2')
                        z = np.array(numbers(table,'values')).reshape(len(xs),len(ys))
                        tables.append((xs,ys,z))
                    arcs[pin] = sense, tuple(tables)
                self.cells.append(Cell(name,inputs,truth,caps,tuple(arcs[p][0] for p in inputs),tuple(arcs[p][1] for p in inputs)))
            except (ValueError, KeyError, TypeError, StopIteration) as exc:
                self.excluded.append({'cell':name,'reason':str(exc)})
        if not self.cells: raise ValueError('No usable combinational library cells')
        self.cap = statistics.median(c for cell in self.cells for c in cell.caps)
        raw = {}
        for ci, cell in enumerate(self.cells):
            n = len(cell.pins)
            for perm in permutations(range(n)):
                for phase in range(1 << n):
                    truth = 0
                    for x in range(1 << n):
                        idx = sum((((x >> perm[p]) & 1) ^ ((phase >> p) & 1)) << p for p in range(n))
                        truth |= ((cell.truth >> idx) & 1) << x
                    raw.setdefault((n,truth),[]).append((ci,perm,phase))
        self.matches = {}
        for key, options in raw.items():
            # Fixed nominal-delay pruning. Pin identity/permutation remains explicit.
            self.matches[key] = tuple(sorted(options,key=lambda o:(float(np.max(self.arcs(o[0],1))),self.cells[o[0]].name,o[1],o[2]))[:max_matches])
        inv = [i for i,c in enumerate(self.cells) if len(c.pins)==1 and c.truth==1]
        if not inv: raise ValueError('Library needs a physical inverter')
        self.inverter = min(inv,key=lambda i:float(np.max(self.arcs(i,1))))

    @lru_cache(maxsize=8192)
    def arcs(self, ci, fanout):
        cell = self.cells[ci]
        result = np.full((len(cell.pins),2,2), -np.inf)
        for pin, (sense,tables) in enumerate(zip(cell.senses,cell.tables)):
            for output, (xs,ys,z) in enumerate(tables):
                # Structural fanout is a fixed estimate; no mapped capacitance is read.
                cols = [np.interp(max(fanout,1)*self.cap,ys,row) for row in z]
                delay = float(np.interp(self.slew_ps/self.time_ps,xs,cols))*self.time_ps
                for inp in range(2):
                    if sense=='non_unate' or (inp==output)==(sense=='positive_unate'):
                        result[pin,inp,output] = max(delay,0.)
        return result

    def metadata(self):
        return {'usable_cells':len(self.cells),'excluded_cells':self.excluded,'nominal_slew_ps':self.slew_ps,
                'median_input_capacitance_library_units':self.cap,'matches_per_truth':self.max_matches,
                'load_model':'max(structural reference count,1) * median library input capacitance; table-clamped',
                'time_scale_ps':self.time_ps}
