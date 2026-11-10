#!/usr/bin/env python3
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
with path.open("rb") as f:
    header = f.readline().decode("ascii").strip().split()
if header[0] not in ("aig", "aag"):
    raise SystemExit("not an AIGER file")
M, I, L, O, A = map(int, header[1:6])
data = path.read_bytes()
names = {"pi_names": [], "po_names": []}
for line in data.splitlines():
    try:
        s = line.decode("utf-8")
    except UnicodeDecodeError:
        continue
    if s.startswith("i") and " " in s and s[1:s.index(" ")].isdigit():
        names["pi_names"].append(s.split(" ", 1)[1])
    elif s.startswith("o") and " " in s and s[1:s.index(" ")].isdigit():
        names["po_names"].append(s.split(" ", 1)[1])
result = {
    "path": str(path.resolve()), "format": header[0], "max_var": M,
    "pi_count": I, "latch_count": L, "po_count": O, "and_count": A,
    **names,
}
json.dump(result, sys.stdout, indent=2)
print()
