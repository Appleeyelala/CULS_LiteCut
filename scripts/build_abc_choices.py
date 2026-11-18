"""Build the offline choices observer using the frozen endpoint instrumentation."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from global_timing.common import digest, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    if not os.getenv("SLURM_JOB_ID") or not args.destination.resolve().is_relative_to("/data/wushangge"):
        parser.error("Use a /data run in Slurm")
    shutil.copytree(ROOT / "third_party/abc", args.destination)
    timing = args.destination / "src/map/scl/sclSize.c"
    content = timing.read_text()
    start = "void Abc_SclTimePerformInt("
    call = "    Abc_SclTimeNtkPrint( p, fShowAll, fPrintPath );"
    assert content.count(start) == content.count(call) == 1
    observer = "\n".join((ROOT / "framework/abc" / name).read_text()
                         for name in ("endpoint_report.c.inc", "path_report.c.inc"))
    timing.write_text(content.replace(start, observer + "\n" + start).replace(
        call, call + "\n    CulsGlobalTimingReport( p );\n    CulsGlobalPathReport( p );"))
    command = ["make", "-j", os.environ["SLURM_CPUS_PER_TASK"], "ABC_USE_NO_READLINE=1",
               "CC=/usr/local/bin/gcc", "CXX=/usr/local/bin/g++", "LD=/usr/local/bin/g++"]
    subprocess.run(command, cwd=args.destination, check=True)
    save_json(args.destination.parent / "abc_choices.json",
              {"binary": str(args.destination / "abc"), "sha256": digest(args.destination / "abc"),
               "command": command, "job": os.environ["SLURM_JOB_ID"]})


if __name__ == "__main__":
    main()
