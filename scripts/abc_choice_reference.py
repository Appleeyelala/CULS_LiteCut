"""Offline native choice covering, exact vector attribution and frozen replay remap."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from global_timing.abc_choices import choice_program, cover_witness
from global_timing.boolean_candidates import parse_library
from global_timing.candidates import Catalogue
from global_timing.common import digest, run_tool, save_json, tool_path
from global_timing.evaluation import Evaluator, cec_passed, parse_timing
from global_timing.execution import replay_many
from global_timing.structure import aig_structure_digest, parse_aig


def read(path):
    return json.loads(path.read_text())


def netlist_body(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for line in stream:
            if not line.startswith(b"//"):
                checksum.update(line)
    return checksum.hexdigest()


def research_map(binary, library, original, work, spec=None):
    work.mkdir(exist_ok=False)
    env = dict(os.environ, CULS_COVER_TRACE="1", CULS_GLOBAL_TIMING_REPORT="1")
    env.pop("CULS_CHOICES", None)
    if spec is not None:
        env["CULS_CHOICES"] = str(spec)
    cmd = (f"read_lib {tool_path(library)}; read_aiger {tool_path(original)}; map; topo; stime -p; "
           f"write_verilog {tool_path(work/'mapped.v')}; strash; write_aiger {tool_path(work/'cover.aig')}")
    log = run_tool([binary, "-c", cmd], work, "mapping", 900, env)
    return log, parse_timing(log)


def case_reference(folder, out, binary, lib, bdelay):
    out.mkdir()
    frozen = read(folder/"frozen_predictions.json")
    source = Path(frozen["source"])
    original = source / "input.aig"
    if digest(original) != frozen["input_sha256"]:
        raise ValueError("Original input hash changed")
    cat = Catalogue(**read(folder/"catalogue.json"))
    vectors = [v["choices"] for v in frozen["combinations"]]
    for v in vectors:
        cat.validate(v)
    scores = read(folder/"scores.json")["scores"]
    records = read(folder/"measurements.json")
    tools = records["tools"]
    abc, liberty = (Path(tools[k]["path"]) for k in ("abc", "library"))
    command = read(folder/"batch_replay/rewrite.command.json")
    gpuls = Path(command["argv"][0])
    for k in ("abc", "library", "checkpoint"):
        if digest(Path(tools[k]["path"])) != tools[k]["sha256"]:
            raise ValueError("Frozen asset drift: " + k)
    save_json(out/"inputs.json", {"panel": str(folder), "input_sha256": digest(original),
              "panel_files": {n: digest(folder/n) for n in ("catalogue.json","scores.json",
                        "measurements.json","frozen_predictions.json","batch_replay/rewrite.command.json")},
              "research_abc": [str(binary), digest(binary)], "replay": [str(gpuls), digest(gpuls)],
              "tools": tools, "owners_sha256": digest(source/"work/partition/owners.csv")})
    _, default = research_map(binary, liberty, original, out/"default")
    historical = read(folder/"baseline.json")
    checks = {k: default[k] == historical[k] for k in ("po_rise_fall", "delay_ps", "area", "mapped_gates")}
    checks["netlist"] = netlist_body(out/"default/mapped.v") == netlist_body(source/"work/original/mapped.v")
    save_json(out/"default_equivalence.json", checks)
    if not all(checks.values()):
        raise ValueError("Research ABC default differs from frozen mapping: " + str(checks))
    lines, options = choice_program(parse_aig(original), cat.decisions, vectors, scores, lib, 0.)
    spec = out/"choices.txt"
    spec.write_text("\n".join(lines)+"\n")
    log, intrinsic = research_map(binary, liberty, original, out/"choices", spec)
    cec = run_tool([abc,"-c",f"cec -n {original} {out/'choices/cover.aig'}"], out/"choices","cec",900)
    if not cec_passed(cec):
        raise ValueError("Choice cover failed CEC")
    witness = cover_witness(log, vectors, options)
    selected = witness["selected"]
    save_json(out/"frozen_choice.json", dict(witness, vectors=vectors, intrinsic=intrinsic,
                                              cover_cec="PASS", selection_uses_labels=False))
    # Both controls are evaluated afresh; the legal empty vector deploys the original.
    indices = [0] + ([selected] if selected else [])
    replay = replay_many(gpuls, original, source/"work/partition/owners.csv", cat,
                         [vectors[k] for k in indices], out/"batch",
                         [out/f"replay_{k}" for k in indices], timeout=900, allow_nonpositive_gain=True)
    evaluator = Evaluator(abc, liberty, original, 900)
    measurements = {r["mask"]: r for r in records["results"] if r["status"] == "PASS"}
    checked = {}
    for index, item in zip(indices, replay):
        if item["status"] != "PASS":
            raise ValueError(item)
        installed = Path(item["aig"])
        if index in measurements and index:
            old = Path(measurements[index]["execution"]["aig"])
            if aig_structure_digest(old) != aig_structure_digest(installed):
                raise ValueError("Fresh replay and archived numbered AIG differ")
        checked[index] = evaluator.evaluate(installed, out/f"remap_{index}")
    observed = {i: r["metrics"] for i,r in measurements.items() if r["metrics"]["cec"] == "PASS"
                and r["metrics"]["area"] <= historical["area"]*1.05}
    best = min(observed, key=lambda k: (observed[k]["delay_ps"], sum(bool(v) for v in vectors[k]), k))
    final = None if selected is None else (historical if selected == 0 else checked[selected])
    row = {"circuit": frozen["circuit"], "measured": len(measurements), "baseline_B_ps": bdelay,
           "original_ps": historical["delay_ps"], "serialized_noop_ps": checked[0]["delay_ps"],
           "oracle_mask": best, "oracle_ps": observed[best]["delay_ps"], "mode0": witness["mode0"],
           "intrinsic_ps": intrinsic["delay_ps"], "intrinsic_area_ratio": intrinsic["area"]/historical["area"],
           "cover_cec": "PASS", "admitted_masks": ",".join(map(str,witness["admitted_vectors"])),
           "selected_mask": selected, "replay_ps": final["delay_ps"] if final else None,
           "area_ratio": final["area"]/historical["area"] if final else None,
           "replay_cec": final["cec"] if final else None,
           "gap_ps": final["delay_ps"]-observed[best]["delay_ps"] if final else None,
           "cut_count_with_provenance":len(witness["cuts"]),
           "linked":sum(o["abc"][-1]=="linked" for o in witness["options"]),
           "alias":sum(o["abc"][-1]=="alias" for o in witness["options"]),
           "unavailable":sum(o["abc"][-1] not in ("linked","alias") for o in witness["options"])}
    save_json(out/"row.json",row)
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel", type=Path, required=True)
    p.add_argument("--abc", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--ids", type=int, nargs="+")
    a = p.parse_args()
    if not os.getenv("SLURM_JOB_ID") or not a.output.resolve().is_relative_to("/data/wushangge"):
        p.error("Use Slurm and /data")
    lib = parse_library(ROOT/"projects/culs/src/algorithms/rewrite_library.inc")
    table = ROOT/"analysis/joint_gradient_v2/stage1_2/selected_results.csv"
    with table.open() as f:
        b = {int(r["circuit_id"]):float(r["B_delay_ps"]) for r in csv.DictReader(f)}
    rows, failures = [], {}
    for i in a.ids or sorted(b):
        name = f"case_{i:04d}"
        try:
            row = case_reference(a.panel/name, a.output/name, a.abc, lib, b[i])
            rows.append(row)
            print(json.dumps(row),flush=True)
        except Exception as exc:
            import traceback
            failures[i] = traceback.format_exc()
            print(failures[i],flush=True)
    save_json(a.output/"status.json", {"attempted":len(rows)+len(failures),"failures":failures})
    if rows:
        with (a.output/"reference.csv").open("x",newline="") as f:
            w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
