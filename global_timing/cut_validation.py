"""Certify exported cut functions, retaining rejected/unproved entries explicitly.

Support minimization can make cut leaves logically related. A mismatch under
free boundary variables is not itself an actual functional counterexample.
Fallback CEC compares the original root with its exported truth LUT, with all
leaf functions connected to the same original PI-driven AIG.
"""
from pathlib import Path

from .candidates import Catalogue
from .common import ToolError, run_tool, save_json, tool_path
from .evaluation import cec_passed
from .litecut import verify_cut_truth
from .truth import ContextProofRequired


def write_context_pair(aig, decisions, work):
    work.mkdir(parents=True, exist_ok=False)
    common = ['.model cut_context', '.inputs ' + ' '.join(f'n{i}' for i in range(1, aig.pi_count + 1)),
              '.outputs ' + ' '.join(f'check_{i}' for i in range(len(decisions))), '.names n0']
    # Repeat a signal as both fanins through distinct aliases. BLIF readers may
    # reject duplicate names in a truth table even when the AIG permits them.
    for node, (left, right) in sorted(aig.fanin_literals.items()):
        common += [f'.names n{left >> 1} edge_{node}_0', f'{0 if left & 1 else 1} 1',
                   f'.names n{right >> 1} edge_{node}_1', f'{0 if right & 1 else 1} 1',
                   f'.names edge_{node}_0 edge_{node}_1 n{node}', '11 1']
    actual, expected = list(common), list(common)
    for index, decision in enumerate(decisions):
        actual += [f'.names n{decision["root"]} check_{index}', '1 1']
        leaves = decision['cut_leaves']
        expected.append('.names ' + ' '.join(f'n{leaf}' for leaf in leaves) + f' check_{index}')
        truth = int(decision['truth']) & 65535
        expected.extend(''.join(str((assignment >> j) & 1) for j in range(len(leaves))) + ' 1'
                        for assignment in range(1 << len(leaves)) if (truth >> assignment) & 1)
    paths = [work / 'original_roots.blif', work / 'cut_luts.blif']
    for path, lines in zip(paths, [actual, expected]):
        with path.open('x') as stream:
            stream.write('\n'.join(lines + ['.end']) + '\n')
    return paths


def validate_catalogue(abc, partition, catalogue, output, timeout=300):
    output.mkdir(parents=True, exist_ok=False)
    validations, pending = {}, []
    for index, decision in enumerate(catalogue.decisions):
        try:
            validations[index] = dict(verify_cut_truth(partition.aig, decision), status='PASS')
        except (ContextProofRequired, RecursionError) as exc:
            pending.append(index)
            validations[index] = {'status': 'PENDING_CONTEXT_CEC', 'reason': str(exc)}
        except ValueError as exc:
            validations[index] = {'status': 'FAIL', 'reason': str(exc)}
    proof_counter = 0

    def prove(indices):
        nonlocal proof_counter
        proof_counter += 1
        work = output / f'context_proof_{proof_counter:05d}'
        reference, candidate = write_context_pair(partition.aig, [catalogue.decisions[i] for i in indices], work)
        command = f'cec -n -T {min(int(timeout), 120)} {tool_path(reference)} {tool_path(candidate)}'
        try:
            text = run_tool([abc, '-c', command], work, 'cec', timeout)
            proven = cec_passed(text)
        except ToolError as exc:
            text, proven = str(exc), False
        if proven:
            for index in indices:
                validations[index] = {'status': 'PASS', 'boolean_validator': 'ABC_ORIGINAL_CONTEXT_CEC',
                    'cut_semantics': 'FUNCTIONAL_SUPPORT_WITH_ORIGINAL_LEAF_RELATIONSHIPS', 'proof': str(work / 'cec.log')}
        elif len(indices) > 1:
            middle = len(indices) // 2
            prove(indices[:middle]); prove(indices[middle:])
        else:
            validations[indices[0]] = {'status': 'FAIL' if 'not equivalent' in text.lower() else 'UNPROVEN',
                'reason': 'Original-context CEC did not prove the exported cut LUT equivalent.', 'proof': str(work / 'cec.log')}

    # Batch shared original logic; failed groups are split to identify entries.
    for start in range(0, len(pending), 256):
        prove(pending[start:start + 256])
    accepted = [i for i in range(len(catalogue.decisions)) if validations[i]['status'] == 'PASS']
    remap = {old: new for new, old in enumerate(accepted)}
    decisions = [dict(catalogue.decisions[i], cut_validation=validations[i]) for i in accepted]
    result = Catalogue(decisions, [(remap[a], remap[b]) for a, b in catalogue.conflicts if a in remap and b in remap],
                       {rid: [remap[i] for i in indices if i in remap] for rid, indices in catalogue.region_decisions.items()})
    audit = {'raw_executable_decisions': len(catalogue.decisions), 'verified_decisions': len(decisions),
             'excluded_decisions': len(catalogue.decisions) - len(decisions),
             'policy': 'Only Boolean-proven cuts enter optimization; full raw catalogue and every failed/unproved entry are retained.',
             'entries': [{'raw_decision_index': i, 'root': d['root'], 'cut_slot': d['cut_slot'], **validations[i]}
                         for i, d in enumerate(catalogue.decisions)]}
    save_json(output / 'audit.json', audit)
    save_json(output / 'verified_catalogue.json', {'decisions': decisions, 'conflicts': result.conflicts,
                                                'region_decisions': result.region_decisions})
    return result, audit
