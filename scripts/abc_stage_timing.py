"""Separate the mode0 metric, Liberty on its actual cover, and final recovered Liberty delay."""
import argparse
import csv
import os
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from abc_choice_reference import read
from abc_mode0_panel import write_csv
from global_timing.common import digest, run_tool, save_json
from global_timing.evaluation import parse_timing


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ("abc","panel","output"):
        p.add_argument("--"+n,type=Path,required=True)
    a=p.parse_args()
    if not os.getenv("SLURM_JOB_ID") or not a.output.resolve().is_relative_to("/data/wushangge"):
        p.error("Use a /data Slurm snapshot")
    rows=[]
    for i, indices in [(44,(1,7)),(75,(6,8)),(179,(1,7,8))]:
        folder=a.panel/f"case_{i:04d}"
        metrics=read(folder/"measurements.json")["results"]
        for k in indices:
            record=next(r for r in metrics if r["index"]==k)
            full=record["metrics"]
            installed=Path(record["execution"]["aig"])
            if digest(installed)!=full["input_sha256"] or full["cec"]!="PASS":
                raise ValueError("Replay identity or CEC drift")
            liberty=Path(full["provenance"]["library"]["path"])
            if digest(liberty)!=full["provenance"]["library"]["sha256"]:
                raise ValueError("Library drift")
            work=a.output/f"case_{i}_{k}"
            work.mkdir()
            env=dict(os.environ,CULS_COVER_TRACE="1",CULS_GLOBAL_TIMING_REPORT="1")
            cmd=f"read_lib {liberty}; read_aiger {installed}; map -r -v; topo; stime -p"
            text=run_tool([a.abc,"-c",cmd],work,"initial_cover",900,env)
            native=[float(s.split()[1]) for s in text.splitlines() if s.startswith("CULS_MODE0 ")]
            if len(native)!=1:
                raise ValueError("Missing native mode0")
            early=parse_timing(text)
            save_json(work/"timing.json",dict(initial_cover_scl=early,recovered_source=str(folder/"measurements.json"),
                      installed_sha256=digest(installed),abc_sha256=digest(a.abc),liberty_sha256=digest(liberty)))
            rows.append(dict(circuit=i,index=k,mode0=native[0],
                mode0_printed_latest=" ".join(re.findall(r"^Output\s+(\S+)\s+: Delay",text,re.M)),
                delay_only_cover_scl_ps=early["delay_ps"], initial_cover_scl_po=early["critical_po_indices"],
                final_cover_scl_ps=full["delay_ps"], final_cover_scl_po=full["critical_po_indices"],
                delay_only_area=early["area"],final_area=full["area"]))
    write_csv(a.output/"stages.csv",rows)


if __name__=="__main__":
    main()
