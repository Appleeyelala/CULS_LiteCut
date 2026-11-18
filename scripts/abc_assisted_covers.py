"""Factorial full-choice ablation with the frozen build, same-input CEC and legal replay QoR."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from abc_choice_reference import read, research_map, netlist_body
from abc_mode0_panel import write_csv
from global_timing.abc_choices import choice_program, cover_witness
from global_timing.boolean_candidates import parse_library
from global_timing.common import digest, legacy, run_tool, save_json
from global_timing.evaluation import cec_passed, Evaluator
from global_timing.structure import aig_structure_digest, parse_aig


def run_case(folder, out, original_ref, binary, lib, alphas):
    out.mkdir()
    frozen=read(folder/"frozen_predictions.json")
    original=Path(frozen["source"])/"input.aig"
    if digest(original)!=frozen["input_sha256"]:
        raise ValueError("Input drift")
    original_metrics=read(folder/"baseline.json")
    frozen_tools=original_metrics["provenance"]
    abc,liberty=(Path(frozen_tools[k]["path"]) for k in ("abc","library"))
    for k in ("abc","library"):
        if digest(Path(frozen_tools[k]["path"]))!=frozen_tools[k]["sha256"]:
            raise ValueError("Frozen evaluator drift")
    vectors=[r["choices"] for r in frozen["combinations"]]
    cat=read(folder/"catalogue.json")["decisions"]; scores=read(folder/"scores.json")["scores"]
    aig=parse_aig(original)
    # Reuse a proof only for an identical numbered AIG with the identical positional original.
    if not cec_passed((original_ref/"choices/cec.log").read_text()):
        raise ValueError("Unverified reference netlist")
    proofs={aig_structure_digest(original_ref/"choices/cover.aig"):str(original_ref/"choices/cec.log")}
    rows=[]; measured={}
    for alpha in alphas:
        work=out/f"a{alpha}";work.mkdir()
        lines,options=choice_program(aig,cat,vectors,scores,lib,alpha)
        spec=work/"choices.txt";spec.write_text("\n".join(lines)+"\n")
        log,metrics=research_map(binary,liberty,original,work/"cover",spec)
        witness=cover_witness(log,vectors,options)
        graph=work/"cover/cover.aig"; fingerprint=aig_structure_digest(graph)
        if fingerprint not in proofs:
            cec=run_tool([abc,"-c",f"cec -n {original} {graph}"],work/"cover","cec",900)
            if not cec_passed(cec):
                raise ValueError("Unproved or inequivalent assisted cover")
            proofs[fingerprint]=str(work/"cover/cec.log")
        if alpha==0 and (netlist_body(work/"cover/mapped.v")!=netlist_body(original_ref/"choices/mapped.v") or
                witness["selected"]!=read(original_ref/"frozen_choice.json")["selected"]):
            raise ValueError("Native cover drift")
        save_json(work/"frozen_witness.json",dict(witness,cec=proofs[fingerprint],abc=digest(binary),
                  cover_structure_sha256=fingerprint,alpha=alpha))
        idx=witness["selected"]
        if idx is not None and idx not in measured:
            if idx==0:
                measured[idx]=original_metrics
            else:
                replay=read(folder/f"measurements/mask_{idx:02d}/rewrite/replay.json")
                if (replay["choices"]!=vectors[idx] or replay["status"]!="PASS" or
                        replay["applied_rewrites"]!=sum(bool(c) for c in vectors[idx])):
                    raise ValueError("Archived replay did not install the selected vector")
                measured[idx]=Evaluator(abc,liberty,original,900).evaluate(Path(replay["aig"]),out/f"frozen_remap_{idx}")
        real=measured.get(idx)
        rows.append(dict(circuit=frozen["circuit"],alpha=alpha,cover_cec="PASS",cover_ps=metrics["delay_ps"],
                         cover_area_ratio=metrics["area"]/original_metrics["area"],mode0=witness["mode0"],
                         witness=idx,real_ps=real["delay_ps"] if real else None,
                         area_ratio=real["area"]/original_metrics["area"] if real else None,
                         replay_cec=real["cec"] if real else None))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("panel","output","abc","weights","reference","recovery"):
        p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not a.output.resolve().is_relative_to("/data/wushangge"):
        p.error("Use a /data Slurm snapshot")
    alpha=read(a.weights)["alpha"]
    alphas=sorted({0.,alpha,1.})
    lib=parse_library(ROOT/"projects/culs/src/algorithms/rewrite_library.inc");legacy("gate3b_analyze")
    def one(f):
        ref=a.recovery/f.name
        if not (ref/"row.json").exists():
            ref=a.reference/f.name
        return run_case(f,a.output/f.name,ref,a.abc,lib,alphas)
    with ThreadPoolExecutor(max_workers=int(os.environ["SLURM_CPUS_PER_TASK"])) as pool:
        groups=list(pool.map(one,sorted(a.panel.glob("case_*"))))
    write_csv(a.output/"covers.csv",[r for group in groups for r in group])
    save_json(a.output/"protocol.json",{"frozen_alpha":alpha,"alpha_sha256":digest(a.weights),
                                       "binary_sha256":digest(a.abc),"alphas":alphas})


if __name__=="__main__":
    main()
