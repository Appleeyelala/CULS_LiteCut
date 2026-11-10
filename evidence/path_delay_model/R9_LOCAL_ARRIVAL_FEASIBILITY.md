# R9 region-root local mapper arrival feasibility — PASS with conservative PI closure

## Question and scope

Can baseline context plus the fixed candidate AIG predict initial delay-DP, all-PI, positive-phase R9 root arrival without candidate full mapping? **Yes for this one case using the PI-closed R9 fanin cone.** This does not establish an exact compact13/11-node update or a multi-path predictor.

Object: c1355, same `root197_cut2_subgraph2`; original cut root197→candidate287; original region root213→candidate322. The selected metric is `Map_Node_t.tArrival[1]` after mode0 phase reconciliation. It is not final best arrival at197, source-restricted PI4→PO11, or Liberty stime.

## Native implementation and baseline context

Independent ABC source: `/data/wushangge/runs/CULS-R9-local-arrival-20261014-ZM1dee/abc` (RUN/abc), upstream commit `c738ed6e86e6ca10bcef0d4c649cc07d925b4be8`, existing pair/mapper patches plus `../r9_local_arrival.patch`. Binary RUN/abc/abc SHA256 `91835b1729d24f4826e7d11b95fb8b25d6991ea1dd75a314fb087971268572c4`. Frozen binaries/source in AIGRewrite were not changed.

Only one normal baseline `map` is executed. `Map_R9CaptureFrozen` is a read-only hook immediately after initial `Map_MappingMatches` completes, **before** reference covering/area recovery. It captures PI phase arrivals, structural references, flags/epsilon, inverter metadata, and the five compact-frontier nodes' cuts/best-phase context. It records actual baseline root arrival for the sanity gate, not as a candidate prediction input.

The independent `local_r9_arrival <baseline AIG> <candidate AIG>` command:

1. Reads the baseline AIG into a fresh manager through native `Abc_NtkToMap`. Constructs the full subject graph but replaces the processing vector with the PI-closed root TFI.
2. Seeds fixed baseline PI arrivals/references; leaves cut order, Num identities, library, phase and comparator rules native.
3. Calls native `Map_MappingSetChoiceLevels`, `Map_MappingCuts`, `Map_MappingTruths`, and `Map_MappingMatches` in mode0. These invoke the original `Map_MatchNodeCut`, `Map_TimeCutComputeArrival`, `Map_MatchCompare`, phase-dropping and arrival-transfer routines.
4. Checks baseline local arrival against the freshly captured normal-baseline mode0 value with0.00002 ps tolerance. If it fails, returns before even reading the candidate AIG.
5. Only after this gate succeeds, performs the same TFI-restricted calculation for candidate root322. No candidate `Map_Mapping`, covering/area recovery, final mapped netlist, or downstream timing call occurs.

Code changes are experimental orchestration/observation, not modifications of cut enumeration, truth matching, cost, comparator, pin permutation or propagation rules. The normal baseline mapped BLIF body is byte-identical to the previously verified original baseline BLIF (timestamp comment excluded), confirming noninterference with baseline mapping. No new candidate full-map check was run.

Library: same loaded asap7 SCL→derived fixed MIO/supergate state initialized by baseline `map`, kept live for both local managers. Rise/fall, four transition arcs, match cap30,5-feasible cuts,250 retained-cut cap, phase selection and0.001 comparator epsilon are unchanged. No Liberty stime analysis is used. Technology libraries/binaries are not included in GitHub.

`R9_FROZEN_BASELINE_CONTEXT.json` is an auditable baseline-only metadata/state export. The prototype uses the complete native library table already initialized in memory; the JSON deliberately does **not** redistribute a full technology/supergate library or constitute a standalone portable execution input. It preserves relevant baseline-root supergate arc metadata and compact-frontier best implementations in addition to PI context.

## State locality and global structural requirements

| State | Treatment |
|---|---|
| PI positive seeds and baseline PI reference counts | frozen external context |
| Library arc/match tables, inverter, mode0 flags/epsilon | frozen baseline initialized state |
| Internal cuts/truths/matches/arrival and area flow | recomputed only within root TFI |
| Internal subject structural fanout counts and original Num/order | reconstructed from the AIG, including full-network structural references; no candidate timing involved |
| Required limits | original fresh-manager initialization and native mode0 tightening; no recovery-required times imported |
| Candidate full-map root/cover, NAND4/downstream state, PO delay | forbidden during prediction; only root reference label read afterward for validation |

Area flow is needed even in mode0 for tie-breaking. `Map_MappingEstimateRefsInit` uses structural fanout counts, not final-cover references. Local internal costs are recomputed; fixed PI costs/reference policy does not inject candidate mapping results. The whole candidate AIG is structurally scanned to recover these references, so this is not strictly O(compact-window) computation. There is no demonstrated equivalent algorithm using just the13/11 changed-window nodes and scalar boundary timings.

## Save-before-validation / leakage check

Prediction native output was completed without opening candidate full-map logs/context. `finalize_r9_local_arrival.py` writes `local_predictions_prevalidation.json` **before** opening the prior pin-audit evidence for labels. Final prevalidation SHA256: `eabdb447e640a3900249bdace03233c139382cebc188cbf50d3aade16ad65db6`. After this file exists, the validator reads only the existing first mode0 NAND4 matching event's R9 leaf arrival as reference:109.239990 baseline,106.219986 candidate. No fitted constant, feedback adjustment or second candidate calculation is used.

The structural audit reads only input AIGs/accepted replay metadata and performs function-only CEC for seven interface/root points. It never runs candidate mapping. Old published values were known to the investigator, but the prediction implementation contains no106.219986 literal or candidate full-map file access.

## Results

| Variant | Function | Phase/model | Local arrival ps | Full-map reference ps | Absolute/relative error |
|---|---:|---|---:|---:|---:|
| Baseline | 213 | positive / mode0 all-PI | 109.239990 | 109.239990 | 0 /0 |
| Candidate | 322 | positive / mode0 all-PI | **106.219986** | **106.219986** | **0 /0** |

Equality is at float32 round-trip9g output precision, not an assertion of infinite precision. Baseline matches with zero printed-value mismatch before candidate execution. Predicted delta **−3.020004 ps** matches the reference delta and correctly predicts improvement. No Pearson/MAE/RMSE is meaningful or computed for this one sample.

The predicted value is below the supplied approximate108.59 ps two-permutation comparison threshold by2.370014 ps. This sanity check is compatible with the earlier pin-switch observation, but does not recompute NAND4 mapping, guarantee final best selection, or predict endpoint delay. The−2.7300033 ps final-best change at197/287 is not used.

## Cost, counted work and measurement limits

| Work | Baseline local sanity | Candidate local calculation |
|---|---:|---:|
| Recomputed AND nodes | 49 | 47 |
| Used PI frontier nodes | 14 | 14 |
| Whole subject objects structurally constructed | 553 | 551 |
| Retained nontrivial cuts generated | 1220 | 944 |
| Supergate match attempts (`nMatches`) | 3356 | 2750 |
| Phase evaluations (`nPhases`) | 3450 | 2844 |
| Local invocation elapsed | 0.009016 s | **0.008150 s** |
| Mode0 matching portion | 0.000963 s | 0.002564 s |

CSV `evaluated_cuts` counts retained nontrivial cuts generated in the local cone, including unmatchable ones; it is not the number of all pre-pruning cuts or completed timing evaluations. Match counters include attempted/bounded-pruned alternatives, not only winners.47 is recomputed-cone size, not a claim that47 functions changed; the compact structural window contains11 candidate ANDs.

One-time normal baseline mode0 preprocessing measured cuts0.014161 + truths0.168294 + matching0.006391 = **0.188846 s**. This excludes library loading/derivation, later area recovery/netlist construction, build time and isolated context-export overhead. Context-export overhead was not separately measured (reported unavailable, not fabricated); no second baseline run was made to measure it. Local elapsed time includes AIG reading/full subject construction and local cuts/truth/matching, with debugging active, excluding one-time library initialization.

On this Linux build `Abc_Clock` uses **CLOCK_MONOTONIC wall time**, not CPU time. The first debug field names incorrectly said `_cpu_s`; the exporter normalizes them to `_wall_s` without rerunning mapping. Values above are wall seconds. These single short observations support no speedup claim.

## Execution and reproducibility

RUN: `/data/wushangge/runs/CULS-R9-local-arrival-20261014-ZM1dee`.

- 49551: structural frontier + function-only CEC, COMPLETED.
- 49556: initial independent build FAILED on an undefined stopwatch helper; no baseline/candidate map executed. Fixed by using ABC's ticks/CLOCKS_PER_SEC conversion.
- 49557: successful incremental build and sole normal-baseline map plus two TFI-local calculations, COMPLETED/exit0,5 seconds,4 CPUs/8G requested.
- 49558/49559: validation/export only, COMPLETED/exit0; no mapping runs. Final exporter corrects clock labeling.

Staged helpers: `run_r9_local_frontier.py`, `finalize_r9_local_arrival.py`, and the existing first-party `gate3b_analyze.py`. Build: `make -C RUN/abc -j4`. Reproduction wrapper: `../r9_local_arrival.slurm`; caller must stage the independent source and patches under a new /data run. Actual command:

```text
AIGREWRITE_R9_LOCAL=1 AIGREWRITE_MAP_DEBUG_NODE=213 RUN/abc/abc -c
  'read PRIOR/input/asap7.lib; read PRIOR/input/c1355_orig.aig;
   map; write_blif RUN/baseline_observed.blif;
   local_r9_arrival PRIOR/input/c1355_orig.aig PRIOR/input/S002.aig'
```

PRIOR is `/data/wushangge/runs/CULS-mapper-internal-20261003-34U6OZ`. AIG hashes remain baseline `2602b10f131a6ba5996521036ba130aab128a4aa15e3ab49ce44f2f02047f0b0`, candidate `59dbe54c03d28c336775785e42d3e4781713e66e96769ac688fd38977672b5e1`. The candidate is the previously independently replayed accepted artifact, not a new or cumulative rewrite. Normal baseline mapping count1; normal candidate mapping count0. Raw graphs, library and binary stay on /data; only small reviewed research artifacts are synchronized.

## Research judgment and remaining limitations

Worth continuing **local mapper timing prediction**, now that a native PI-closed one-root calculation reproduces the held-out mode0 reference without full candidate mapping. It establishes feasibility of restricted root-cone evaluation, not yet the hoped-for minimal-window incremental method.

Main limitations: PI-expanded cone recomputes unchanged logic; full subject graph is still built for identity/order/global structural fanout context; exact cut/area-flow closure at an internal frontier is not implemented; immutable native library state must remain available; only one candidate, root and initial mode0 positive phase were validated. Area recovery, downstream mapping, endpoint impact, dynamic updates and generalization are not claimed. No additional experiment/model/optimizer starts automatically.
