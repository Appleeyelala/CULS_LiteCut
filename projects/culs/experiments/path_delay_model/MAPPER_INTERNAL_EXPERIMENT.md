# R9 mapper-internal observation

Status: **PASS** for stable function-point → subject node → cuts → available matches → best timing → final cover observation. Fixed circuit: c1355. **Region root=213; decision/cut root=197**. Candidate: `root197_cut2_subgraph2` with leaves `[17,33,151,188]`. The existing replay `S002.aig` has exactly one ACCEPTED decision with this identity; its trace and SHA were checked and CEC rerun. No new rewrite choices or sequential accumulation were introduced.

## Proven identities

The runner reuses the Gate3B AIG parser. With fixed seed 20261003 and 4096 bit-parallel assignments it identifies unique reachable counterpart literals, then proves each function via CEC using separate one-output proof AIGs. These proof files never enter the mapper experiment. The unchanged input AIGs are passed to `map`.

| Original function | Candidate serialized AIG var | Literal polarity | CEC |
|---:|---:|---:|---|
| 17 | 17 | 0 | PASS |
| 33 | 33 | 0 | PASS |
| 151 | 232 | 0 | PASS |
| 188 | 234 | 0 | PASS |
| 197 | 287 | 0 | PASS |
| 213 (region root) | 322 | 0 | PASS |
| 211 | 313 | 0 | PASS |
| 212 | 314 | 0 | PASS |

At AIGER read, private debug data records variable→ABC object IDs, including structural hashing/polarity. `Abc_NtkToMap` provides actual object→mapper node links. Baseline 197→ABC object229→mapper196; candidate function197→serialized287→ABC object319→mapper194. PIs preserve their names. AIGER internal AND nodes have no supported original symbols; their name field is explicitly `UNNAMED_INTERNAL_AIGER_NODE` instead of manufacturing `n197` as an ABC name.

## Read-only instrumentation and verification

All changes reside in an independent ABC copy under `/data/wushangge/runs/CULS-mapper-internal-20261003-34U6OZ/abc`. `AIGREWRITE_MAP_DEBUG_NODE` specifies the actual serialized target variable (baseline197/candidate287); `AIGREWRITE_MAP_DEBUG_IDS` adds only the small correspondence list. Normal output is unchanged when the flag is absent. Debug callbacks read current fields and print JSON lines prefixed `MAPDBG`; they do not add cuts, rerun timing, compare matches again, change marks/refs/costs, or alter final cover. Cut/supergate matching is not extended. Internal-cover membership uses private visited memory.

Each variant was run using the original experimental ABC binary, debug binary with flags off, and debug binary with flags on. The mapped BLIF body is identical across those three executions; only comment headers are ignored. Gate count, area, `stime` delay, and full-precision source-restricted MIO pair delay match exactly. Both the original and instrumented mapped networks pass CEC against their input AIG.

| Variant | Gates | Area | Liberty `stime` global delay | MIO PI4→PO11 pair delay | Equality / CEC |
|---|---:|---:|---:|---:|---|
| baseline | 382 | 513.22 | 360.11 | 194.169998 | PASS |
| candidate | 381 | 509.72 | 360.11 | 193.519989 | PASS |

The PO delays above serve only to verify instrumentation invariance. They are not used to derive any local timing quantity.

## Reproduction and provenance

Source commit: `c738ed6e86e6ca10bcef0d4c649cc07d925b4be8` (clean original source rechecked). Instrumentation is [abc_mapper_debug.patch](abc_mapper_debug.patch); it applies to that source independently of the prior source-restricted timing patch. To reproduce the measured command, apply both patches in an isolated `/data` copy. Modified files: `src/base/io/ioReadAiger.c`, `src/base/abci/abcMap.c`, `src/map/mapper/{module.make,mapper.h,mapperInt.h,mapperCore.c,mapperCut.c,mapperTruth.c,mapperMatch.c}`, plus new `mapperDebug.c`.

Build: `make -j8` under Slurm (initial job48504; final incremental build/experiment48505; final export and additional function proofs48508). Job48508 COMPLETED, exit0; resource request8 CPUs,16G,0 GPUs. Binary `/data/wushangge/runs/CULS-mapper-internal-20261003-34U6OZ/bin/abc_pair_timing`, SHA256 `f255acd802f928fb011737843419d1b39454f060bbd74968a355d3e476822784`. Reference binary `/data/wushangge/runs/CULS-source-pair-20260930-qU1wxi/bin/abc_pair_timing`, SHA256 `aa2d17b6de367acd16a622aa894f96da0e92431927dbb68e1ca26462b696c1ae`. Frozen `/home/wushangge/AIGRewrite/abc/abc` is untouched.

Runner: [run_mapper_internal.py](run_mapper_internal.py), staged with the existing `tools/gate3b_analyze.py`. Submit [mapper_internal.slurm](mapper_internal.slurm) with `sbatch --chdir=/data/wushangge/runs/CULS-mapper-internal-20261003-34U6OZ ...`. Mapping script is identical for both variants and binaries: `read ASAP7; read AIG; map; write_blif <run artifact>; topo; stime; print_stats; print_pair_delay po11 pi03`. `write_blif` is export-only. Raw logs, matches, proof AIGs, and invariance manifests remain under the run's `output/` directory. Requested compact CSV/JSON/MD artifacts are copied into this experiment's `results/`.

## Scope of the conclusion

This supports **CASE 1 with an explicit limitation**: the same function point and its mapper timing state are stable and accessible, but the exact four-leaf rewrite cut has **zero direct supergate matches** in the current derived library. Its own direct match arrival is therefore `NOT AVAILABLE`; the function's best timing is computed through other cuts. CASE2 (cut absent) and CASE3 (function point untraceable) do not occur. Next route: **A, mapper-internal timing route**. Injecting this already-present cut would not address the missing implementation. No candidate timing propagation or optimization is implemented.
