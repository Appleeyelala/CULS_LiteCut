"""Check hard conditional ABC timing against fixed per-replay native mode0 controls."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from abc_choice_reference import read
from abc_mode0_panel import write_csv
from global_timing.abc_choices import choice_program
from global_timing.boolean_candidates import parse_library
from global_timing.common import digest, legacy, run_tool, save_json, tool_path
from global_timing.structure import parse_aig


def check(folder, out, binary, lib, factorial):
    out.mkdir()
    frozen=read(folder/("frozen_predictions.json" if factorial else "frozen.json"))
    original=Path(frozen["source"])/"input.aig"
    if digest(original) != frozen["input_sha256"]:
        raise ValueError("Input hash mismatch")
    liberty=Path(read(Path(frozen["source"])/"context.json")["tools"]["library"]["path"])
    cat=read(folder/"catalogue.json")["decisions"]
    vectors=[v["choices"] for v in frozen["combinations"]]
    scores=read(folder/"scores.json")["scores"]
    lines, options=choice_program(parse_aig(original),cat,vectors,scores,lib,0.)
    spec=out/"choices.txt";spec.write_text("\n".join(lines)+"\n")
    save_json(out/"inputs.json",dict(original=digest(original),research_abc=digest(binary),
              liberty=digest(liberty),options=options,vectors=vectors,
              catalogue_sha256=digest(folder/"catalogue.json"),frozen_sha256=digest(
              folder/("frozen_predictions.json" if factorial else "frozen.json"))))
    cmd=[binary,"-c",f"read_lib {tool_path(liberty)}; read_aiger {tool_path(original)}; map"]
    env=dict(os.environ,CULS_COVER_TRACE="1")
    raw=run_tool(cmd,out,"native-original",900,env)
    native=[float(s.split()[1]) for s in raw.splitlines() if s.startswith("CULS_MODE0 ")]
    plain=run_tool(cmd,out,"hard-original",900,dict(env,CULS_HARD_VECTORS="2"))
    repeats=[float(s.split()[2]) for s in plain.splitlines() if s.startswith("CULS_HARD ")]
    if len(native)!=1 or repeats!=native*2:
        raise ValueError(f"Unconditioned hard forward disagrees with ABC or retained mutable state: {native} {repeats}")
    log=run_tool(cmd,out,"conditional",900,dict(env,CULS_CHOICES=str(spec),CULS_HARD_VECTORS=str(len(vectors))))
    actual=[s.split() for s in log.splitlines() if s.startswith("CULS_HARD ")]
    statuses=[s.split()[-1] for s in log.splitlines() if s.startswith("CULS_OPTION ")]
    if [int(s[1]) for s in actual]!=list(range(len(vectors))) or len(statuses)!=len(options):
        raise ValueError("Conditional trace incomplete")
    return [dict(circuit=int(folder.name[5:]),index=int(s[1]),hard=float(s[2]),unmatched_nodes=int(s[3]),
                 unmatched_outputs=int(s[4]),plain_mode0=native[0],plain_repeat=True,
                 linked=statuses.count("linked"),alias=statuses.count("alias"),unavailable=sum(
                     v not in ("linked","alias") for v in statuses)) for s in actual]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ("abc","factorial","panel","output"):
        p.add_argument("--"+arg,type=Path,required=True)
    a=p.parse_args()
    if not os.getenv("SLURM_JOB_ID") or not a.output.resolve().is_relative_to("/data/wushangge"):
        p.error("Use a /data Slurm snapshot")
    lib=parse_library(ROOT/"projects/culs/src/algorithms/rewrite_library.inc")
    legacy("gate3b_analyze")
    for stage in ("factorial","validation"):
        data=a.factorial if stage=="factorial" else a.panel
        folders=[f for f in sorted(data.glob("case_*")) if stage=="factorial" or read(f/"frozen.json")["role"]==stage]
        out=a.output/stage;out.mkdir()
        with ThreadPoolExecutor(max_workers=int(os.environ["SLURM_CPUS_PER_TASK"])) as pool:
            groups=list(pool.map(lambda f:check(f,out/f.name,a.abc,lib,stage=="factorial"),folders))
        rows=[r for group in groups for r in group]
        write_csv(out/"hard_scores.csv",rows)
        print(stage,len(rows),sum(r["unmatched_outputs"]>0 for r in rows),flush=True)


if __name__=="__main__":
    main()
