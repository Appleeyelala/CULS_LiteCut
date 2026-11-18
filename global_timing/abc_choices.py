"""Compile locally proved replacements for the research ABC choice reader."""
import math
from .boolean_candidates import verify_cut_truth, verify_library_truth


def choice_program(aig, decisions, vectors, scores, library, alpha):
    if not 1 <= len(vectors) <= 64 or any(len(v) != len(decisions) for v in vectors):
        raise ValueError("Expected 1..64 frozen vectors of catalogue width")
    if len({d["root"] for d in decisions}) != len(decisions):
        raise ValueError("Shared-root decisions require a joint choice class")
    if any(type(c) is not int or not 0 <= c <= len(d["candidate_subgraph_indices"])
           for v in vectors for c, d in zip(v, decisions)):
        raise ValueError("Invalid frozen choice")
    lines, options = [], []
    next_node = aig.maximum_variable + 1
    local = {(s["decision_index"], s["choice_index"]): s["predicted_local_abc_delay"]
             for s in scores}
    for i, d in enumerate(decisions):
        indices = d["candidate_subgraph_indices"]
        present = sorted(set(v[i] for v in vectors) - {0})
        if not present:
            continue
        verify_cut_truth(aig, d)
        masks = {c: sum(1 << k for k, v in enumerate(vectors) if v[i] == c)
                 for c in [0] + present}
        truth = d["truth"] & 0xffff
        perm = library["perms4"][library["p_perms"][truth]]
        phase = library["p_phases"][truth]
        leaves = list(d["cut_leaves"]) + [0]
        data = library["classes"][d["truth_class"]]
        for choice in present:
            sub = indices[choice - 1]
            complement = int(verify_library_truth(d, sub, library)["output_phase_complement"])
            root = data["roots"][sub]
            values = {k: 2 * leaves[perm[k]] ^ (phase >> k & 1) for k in range(4)}

            def literal(node):
                nonlocal next_node
                if node not in values:
                    x, y, p, q = data["fanins"][node - 4]
                    a, b = literal(x) ^ p, literal(y) ^ q
                    lines.append(f"g {a} {b}")
                    values[node] = 2 * next_node
                    next_node += 1
                return values[node]

            replacement = literal(root) ^ complement
            before, after = local.get((i, 0)), local.get((i, choice))
            valid = all(x is not None and math.isfinite(x) and x > 0 for x in (before, after))
            scale = (after / before) ** alpha if valid else 1.
            lines.append(f"r {i} {choice} {2*d['root']} {replacement} "
                         f"{masks[0]:x} {masks[choice]:x} {scale:.12g}")
            options.append({"decision_index": i, "decision_id": d["decision_id"], "choice": choice,
                            "original_root": d["root"], "cut_slot": d["cut_slot"],
                            "original_leaves": d["cut_leaves"], "subgraph": sub,
                            "replacement_literal": replacement, "output_complement": complement,
                            "original_local": before, "replacement_local": after,
                            "ratio_valid": valid, "pin_scale": scale})
    return lines, options


def cover_witness(log, vectors, options):
    admitted = (1 << len(vectors)) - 1
    statuses, trace, native = {}, [], []
    for line in log.splitlines():
        fields = line.split()
        if fields[:1] == ["CULS_MODE0"]:
            native.append(float(fields[1]))
        elif fields[:1] == ["CULS_OPTION"]:
            i, choice, rep, alt, phase = map(int, fields[1:6])
            if (i, choice) in statuses:
                raise ValueError("Duplicate ABC option")
            statuses[i, choice] = [rep, alt, phase, fields[6]]
        elif fields[:1] == ["CULS_CUT"]:
            root, polarity, supergate, pins = map(int, fields[1:5])
            permitted = int(fields[5], 16)
            admitted &= permitted
            trace.append([root, polarity, supergate, pins, permitted, *map(int, fields[6:])])
    if len(native) != 1 or not math.isfinite(native[0]) or set(statuses) != {
            (o["decision_index"], o["choice"]) for o in options}:
        raise ValueError("Incomplete ABC choices or mode0 trace")
    matches = [k for k in range(len(vectors)) if admitted & (1 << k)]
    return {"admitted_vectors": matches, "selected": min(matches, default=None,
            key=lambda k: (sum(c != 0 for c in vectors[k]), k)), "mode0": native[0],
            "cuts": trace, "options": [dict(o, abc=statuses[o["decision_index"], o["choice"]])
                                       for o in options]}

