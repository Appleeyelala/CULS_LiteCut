#!/usr/bin/env bash
#SBATCH --job-name=gate1cr-e
#SBATCH --output=/data/wushangge/gate1a_AIGRewrite/gate1c_r_out/determinism-%j.out
#SBATCH --error=/data/wushangge/gate1a_AIGRewrite/gate1c_r_out/determinism-%j.err
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --chdir=/data/wushangge/gate1a_AIGRewrite
set -Eeuo pipefail
ROOT=/data/wushangge/gate1a_AIGRewrite; OUT="$ROOT/gate1c_r_out"; BUILD="$ROOT/build_gate1c"; INPUT=/data/public/AIGRewrite/case/c1355.aig
cmake --build "$BUILD" --target abcg -j2 >/dev/null
for run in 1 2; do
  export GATE1C_SPEC=146:0:1,226:0:0 GATE1A_ROOT=146 GATE1A_CUT=0 GATE1A_SUB=1 GATE1A_PROVENANCE="$OUT/BC_run${run}.json"
  LD_LIBRARY_PATH="$BUILD" "$BUILD/abcg" -c "gread $INPUT; grw; gwrite $OUT/BC_run${run}.aig" > "$OUT/BC_run${run}.rewrite.log" 2>&1
done
python3 -c 'import json,hashlib; from pathlib import Path; a=json.load(open("gate1c_r_out/BC_run1.json")); b=json.load(open("gate1c_r_out/BC_run2.json")); norm=lambda d:[(x["target_root"],x["subgraph_index"],x["library_subgraph_root"],x["installation_kind"],x["physical_graph_modified"],x["cut_leaves"]) for x in d["decisions"]]; same=norm(a)==norm(b) and a["dependencies"]==b["dependencies"] and a["rewritten_root_count"]==b["rewritten_root_count"]; stats=[{"sha256":hashlib.sha256(Path(p).read_bytes()).hexdigest(),"bytes":Path(p).stat().st_size} for p in ("gate1c_r_out/BC_run1.aig","gate1c_r_out/BC_run2.aig")]; json.dump({"deterministic_batch_semantics":same,"provenance_equal":same,"runs":stats},open("gate1c_r_out/determinism.json","w"),indent=2); raise SystemExit(0 if same else 1)'
