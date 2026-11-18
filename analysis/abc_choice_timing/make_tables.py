"""Rebuild the small review tables from the indexed, immutable Slurm records."""
import csv
import json
from pathlib import Path
from shutil import copyfile

HERE = Path(__file__).resolve().parent
RUNS = json.loads((HERE / "RUNS.json").read_text())["runs"]


def output(run):
    return Path(RUNS[run]["directory"]) / "output"


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def write_csv(name, rows):
    with (HERE / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def pin_ranking(stage, name):
    selected = read_csv(output("pin_ranking") / stage / "selected.csv")
    frozen = json.loads((output("pin_ranking") / "alpha_frozen.json").read_text())["alpha"]
    rows = {}
    for r in selected:
        alpha = float(r["alpha"])
        if alpha not in (0., frozen):
            continue
        i = int(r["circuit"])
        row = rows.setdefault(i, dict(circuit=i, measured=r["labels"], fastest_measured_ps=r["oracle_ps"]))
        assert r["oracle_ps"] == row["fastest_measured_ps"] and r["labels"] == row["measured"]
        prefix = "native" if alpha == 0 else "litecut"
        for key in ("index", "real_ps", "regret", "top1", "area_ratio"):
            row[f"{prefix}_{key}"] = r[key]
    assert len(rows) == 12 and all("litecut_area_ratio" in r for r in rows.values())
    write_csv(name, [rows[i] for i in sorted(rows)])


def reference():
    references = read_csv(output("reference") / "reference.csv")
    references.extend(read_csv(output("reference_676") / "reference.csv"))
    assert len(references) == 12 and sum(int(r["measured"]) for r in references) == 191
    covers = read_csv(output("covers") / "covers.csv")
    assert len(covers) == 36 and all(r["cover_cec"] == "PASS" for r in covers)
    choices = {(int(r["circuit"]), float(r["alpha"])): r for r in covers}
    rows = []
    for r in sorted(references, key=lambda row: int(row["circuit"])):
        i = int(r["circuit"])
        row = {k: r[k] for k in ("circuit", "measured", "baseline_B_ps", "original_ps", "serialized_noop_ps",
                                  "oracle_mask", "oracle_ps")}
        for prefix, alpha in (("native", 0.), ("litecut", .5)):
            c = choices[i, alpha]
            if not alpha:
                assert r["selected_mask"] == c["witness"] and r["intrinsic_ps"] == c["cover_ps"]
            for k in ("cover_ps", "cover_area_ratio", "cover_cec", "witness", "real_ps", "area_ratio", "replay_cec"):
                row[f"{prefix}_{k}"] = c[k]
            reason = "accepted"
            if not c["witness"]:
                reason = "no_legal_witness"
            elif c["witness"] == "0":
                reason = "empty_choice"
            else:
                assert c["replay_cec"] == "PASS"
                if float(c["area_ratio"]) > 1.05:
                    reason = "area_limit"
                elif float(c["real_ps"]) >= float(r["original_ps"]):
                    reason = "no_delay_improvement"
            row[f"{prefix}_outcome"] = reason
            row[f"{prefix}_after_check_ps"] = c["real_ps"] if reason == "accepted" else r["original_ps"]
        rows.append(row)
    write_csv("choice_covers.csv", rows)


def parity():
    result = []
    for stage in ("factorial", "validation"):
        scores = read_csv(output("conditional") / stage / "hard_scores.csv")
        replays = read_csv(output("pin_ranking") / stage / "combinations.csv")
        native = {(r["circuit"], r["index"]): float(r["native"]) for r in replays if float(r["alpha"]) == 0}
        assert len(scores) == (192 if stage == "factorial" else 108)
        for i in sorted({int(r["circuit"]) for r in scores}):
            group = [r for r in scores if int(r["circuit"]) == i]
            assert all(r["plain_repeat"] == "True" and int(r["unmatched_outputs"]) == 0 for r in group)
            errors = [abs(float(r["hard"]) - native[r["circuit"], r["index"]]) for r in group
                      if (r["circuit"], r["index"]) in native]
            result.append(dict(stage=stage, circuit=i, hard_vectors=len(group), independent_replays=len(errors),
                               within_1e_minus4=sum(x <= 1.e-4 for x in errors), max_abs_native_error=max(errors),
                               unconditioned_original_repeated=True, unmatched_outputs=0))
    write_csv("hard_parity.csv", result)


if __name__ == "__main__":
    pin_ranking("validation", "development.csv")
    pin_ranking("factorial", "factorial_pin_ranking.csv")
    reference()
    parity()
    for run, name in (("stages", "stages.csv"), ("sparse", "solver.csv")):
        write_csv(name, read_csv(output(run) / name))
    copyfile(output("pin_ranking") / "summary.json", HERE / "pin_summary.json")
