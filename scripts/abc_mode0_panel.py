"""Measure the best-case hard ABC timing score on saved legal replays, offline only."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import math
import os
from pathlib import Path
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from abc_choice_reference import read
from global_timing.abc_choices import choice_program
from global_timing.abc_hard_choices import hard_annotations
from global_timing.boolean_candidates import parse_library
from global_timing.common import legacy, digest, run_tool, save_json, tool_path
from global_timing.structure import parse_aig


def score(binary, liberty, installed, work, tags, alpha):
    work.mkdir()
    env = dict(os.environ, CULS_COVER_TRACE="1")
    env.pop("CULS_CHOICES", None)
    if alpha and tags:
        spec = work/"pins.txt"
        lines = []
        for t in tags:
            anchor, *aliases = t["alias_literals"]
            lines.extend(f"e {anchor} {other}" for other in aliases)
            lines.append(f"t {anchor} {t['pin_scale']**alpha:.12g}")
        spec.write_text("\n".join(lines) + "\n")
        env["CULS_CHOICES"] = str(spec)
    log = run_tool([binary, "-c", f"read_lib {tool_path(liberty)}; read_aiger {tool_path(installed)}; map -r"],
                   work,"mode0",900,env)
    values = [float(line.split()[1]) for line in log.splitlines() if line.startswith("CULS_MODE0 ")]
    if len(values) != 1 or not math.isfinite(values[0]):
        raise ValueError("Missing native mode0 arrival")
    return values[0]


def predict_case(folder, output, binary, library, alphas, factorial):
    name = folder.name
    out = output/name
    out.mkdir()
    frozen = read(folder/("frozen_predictions.json" if factorial else "frozen.json"))
    original = Path(frozen["source"])/"input.aig"
    if digest(original) != frozen["input_sha256"]:
        raise ValueError("Original input drift")
    aig = parse_aig(original)
    cat = read(folder/"catalogue.json")["decisions"]
    scores = read(folder/"scores.json")["scores"]
    c = read(Path(frozen["source"])/"context.json")["tools"]
    liberty = Path(c["library"]["path"])
    if digest(liberty) != c["library"]["sha256"]:
        raise ValueError("Library drift")
    rows = []
    for v in frozen["combinations"]:
        idx = v["mask"] if factorial else v["index"]
        if not factorial and idx == 0:
            continue
        work = out/f"v{idx:02d}"
        work.mkdir()
        installed = original if idx == 0 else folder/"measurements"/(
            f"mask_{idx:02d}" if factorial else f"combo_{idx:02d}")/"rewrite/raw/aigs/choice.aig"
        program, options = choice_program(aig, cat, [v["choices"]], scores, library, 1.)
        try:
            checked = hard_annotations(aig, parse_aig(installed), program, options)
            status = "verified"
        except ValueError as exc:
            if str(exc) != "Candidate-composed and installed signed PO structures differ":
                raise
            checked, status = [], "signed_structure_mismatch"
        save_json(work/"correspondence.json", {"status": status, "annotations": checked})
        tags = [t for t in checked if t["installed_node"] is not None and t["pin_scale"] != 1.]
        native = score(binary,liberty,installed,work/"native",[],0)
        for alpha in alphas:
            value = score(binary,liberty,installed,work/f"a{alpha}",tags,alpha) if alpha and tags else native
            rows.append(dict(circuit=int(name[5:]), index=idx, edits=sum(bool(x) for x in v["choices"]),
                             correspondence=status, alpha=alpha, native=native, prediction=value, tags=len(tags),
                             installed=str(installed), sha256=digest(installed)))
    save_json(out/"predictions.json", rows)
    return rows


def evaluate(panel, rows, factorial, alphas):
    circuit_ids = sorted({r["circuit"] for r in rows})
    selected, points = [], []
    for i in circuit_ids:
        folder = panel/f"case_{i:04d}"
        key = "mask" if factorial else "index"
        metrics = {r[key]: r["metrics"] for r in read(folder/"measurements.json")["results"]
                   if r["status"] == "PASS" and r["metrics"]["cec"] == "PASS"}
        area = read(folder/"baseline.json")["area"]
        for alpha in alphas:
            group = [r for r in rows if r["circuit"] == i and r["alpha"] == alpha]
            chosen = min(group,key=lambda r:(r["prediction"],r["edits"],r["index"]))
            eligible = [metrics[r["index"]]["delay_ps"] for r in group if r["index"] in metrics]
            oracle = min(eligible)
            for r in group:
                m = metrics.get(r["index"])
                points.append(dict(r,real_ps=m["delay_ps"] if m else None,
                                   area_ratio=m["area"]/area if m else None))
            actual = metrics.get(chosen["index"])
            selected.append(dict(circuit=i,alpha=alpha,index=chosen["index"],pred=chosen["prediction"],
                            real_ps=actual["delay_ps"] if actual else None,oracle_ps=oracle,
                            regret=100*(actual["delay_ps"]/oracle-1) if actual else None,
                            top1=math.isclose(actual["delay_ps"],oracle,rel_tol=2e-6,abs_tol=2e-5) if actual else None,
                            area_ratio=actual["area"]/area if actual else None,labels=len(eligible)))
    return selected, points


def write_csv(path, rows):
    with path.open("x",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("panel","factorial","abc","output"):
        p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args()
    if not os.getenv("SLURM_JOB_ID") or not a.output.resolve().is_relative_to("/data/wushangge"):
        p.error("Use a /data Slurm snapshot")
    lib=parse_library(ROOT/"projects/culs/src/algorithms/rewrite_library.inc")
    legacy("gate3b_analyze")  # finish the lazy import before worker threads start
    frozen_alpha=None
    records=[]
    for stage in ("train","validation","factorial"):
        panel=a.factorial if stage=="factorial" else a.panel
        folders=[f for f in sorted(panel.glob("case_*"))
                 if stage=="factorial" or read(f/"frozen.json")["role"]==stage]
        if len(folders)!=12:
            raise ValueError(f"Unexpected {stage} cohort size")
        alphas=[0.,.25,.5,1.] if stage=="train" else sorted({0.,frozen_alpha,1.})
        out=a.output/stage;out.mkdir()
        with ThreadPoolExecutor(max_workers=int(os.environ["SLURM_CPUS_PER_TASK"])) as pool:
            chunks=list(pool.map(lambda f: predict_case(f,out,a.abc,lib,alphas,stage=="factorial"),folders))
        rows=[r for chunk in chunks for r in chunk]
        save_json(out/"predictions_frozen.json",{"research_abc_sha256":digest(a.abc),
            "files":{f.name:digest(out/f.name/"predictions.json") for f in folders},"alphas":alphas})
        selected,points=evaluate(panel,rows,stage=="factorial",alphas)
        write_csv(out/"selected.csv",selected);write_csv(out/"combinations.csv",points)
        for alpha in alphas:
            group=[r for r in selected if r["alpha"]==alpha]
            known=[r for r in group if r["real_ps"] is not None]
            records.append(dict(stage=stage,alpha=alpha,scored=len(known),
                                mean_regret=mean(r["regret"] for r in known),top1=sum(r["top1"] for r in known),
                                area_violations=sum(r["area_ratio"]>1.05 for r in known)))
        if stage=="train":
            frozen_alpha=min(records,key=lambda r:(r["mean_regret"],r["alpha"]))["alpha"]
            save_json(a.output/"alpha_frozen.json",{"alpha":frozen_alpha,"train":records.copy(),
                      "target":"percentage regret vs fastest measured CEC-passing vector; area audited separately"})
        print(json.dumps(records[-len(alphas):]),flush=True)
    save_json(a.output/"summary.json",records)


if __name__=="__main__":
    main()
