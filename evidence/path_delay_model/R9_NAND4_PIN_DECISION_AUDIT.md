# R9 NAND4 pin assignment decision audit — PASS

2026-10-14. One c1355 candidate, one NAND4 function, PI4→PO11 only. No mapper decisions, costs, phases or permutations were forced or modified.

## 1. Object identity and logical interface

Fixed replay artifact: `root197_cut2_subgraph2` / S002, ACCEPTED, fresh CEC PASS. R9 region root is original213; rewrite cut root is197, candidate equivalent287. The NAND4 under investigation is **downstream function385**, not either R9 root.

Previous CEC-backed trace correspondence, checked again against the same artifact hashes:

| Quantity | Baseline | Candidate |
|---|---|---|
| Implemented output function | complement(AIG385), literal771 | complement(AIG365), literal731 |
| Mapper subject node | 384 | 382 |
| Mapped gate before topo | 334 | 330 |
| Mapped gate after topo | 333 | 329 |
| Final output phase | 0 | 0 |
| Selected cut index (variant-local) | 12 | 10 |
| Ordered cut leaves | [139,176,213,238] | [338,339,322,340] |
| Ordered mapper leaf IDs | [138,175,212,237] | [138,175,210,235] |
| Cut truth low/high | 536879104 /536879104 | 536879104 /536879104 |
| Selected input phase mask | 2 | 2 |
| Selected supergate | 2046 | 2047 |
| Library cell | NAND4xp25_ASAP7_75t_L | same |
| Physical [A,B,C,D]→ordered leaf indices | [0,2,3,1] | [0,3,1,2] |
| Implemented phase refs | [1,0,1] | [1,0,1] |

The logical output is `NAND(L0, !L1, L2, L3)`: mask2 selects the negative phase of leaf1. The same-number gate IDs are not used as correspondence evidence. The four matched functions are:

| Logical slot | Baseline function | Candidate equivalent | Role | Final baseline pin | Final candidate pin |
|---:|---|---|---|---|---|
| 0 | AIG139 | AIG338 | source-unreachable for this pair | A | A |
| 1 | complement(AIG176) | complement(AIG339) | source-restricted critical predecessor | D | C |
| 2 | AIG213 | AIG322 | R9 region output, competing branch | B | D |
| 3 | AIG238 | AIG340 | source-unreachable for this pair | C | B |

All output/frontier equivalences were CEC-proved in the previous case's `R9_mechanism_evidence.json` and proof logs; current mapping reproduces those same netlist bodies. The candidate IDs are serialized identifiers, not assumed physical gate identities. Cuts are sorted by **mapper Num**, so candidate serialized leaves need not be numerically ascending.

## 2. Actual responsible source path

Source commit: `c738ed6e86e6ca10bcef0d4c649cc07d925b4be8` (rechecked). The executed classic mapper uses the existing independent read-only instrumentation build under `/data/wushangge/runs/CULS-mapper-internal-20261003-34U6OZ` (PRIOR).

| Stage | Source/function | Relevant data and actual action |
|---|---|---|
| Library implementations/permutations | `src/map/mapper/mapperLib.c:Map_SuperLibDeriveFromGenlib` → `src/map/super/superGate.c:Super_PrecomputeStr` / `Super_Compute` | Four-input generation loops over distinct input supergates, computes truth/pin delays and creates implementation trees. Different permutations survive as different `Map_Super_t` entries. |
| Leaf ordering | `src/map/mapper/mapperCut.c:Map_CutMergeTwo` | Sorts subject leaves by `Map_Node_t.Num`, not arrival. This is not a dynamic arrival sort. |
| Cut matching | `src/map/mapper/mapperTruth.c:Map_TruthsCut` / `Map_TruthsCutOne` | Computes ordered truth; N-canonical phase lookup supplies `Map_Match_t.pSupers` and `uPhase` for both output phases. |
| Assignment selection within a cut | `src/map/mapper/mapperMatch.c:Map_MatchNodeCut` (line163) | Loops matching `Map_Super_t` entries and their phase alternatives; `uPhaseBest=uPhase XOR super.uPhases[i]`. Calls timing, cost evaluation and `Map_MatchCompare`; stores `pSuperBest`. |
| Timing and early pruning | `src/map/mapper/mapperTime.c:Map_TimeCutComputeArrival` (line75) | Uses each ordered leaf's selected-phase rise/fall arrival and the supergate's four transition-delay tables. Maxes contributions; returns a large sentinel immediately if a contribution exceeds the current worst limit. |
| Direct comparator | `src/map/mapper/mapperMatch.c:Map_MatchCompare` (line77) | Delay mode0: smaller worst arrival first, then area/area flow, fanout limit, fanin count; exact ties keep incumbent. Epsilon is0.001. Recovery modes prioritize area before arrival. |
| Best cut for node phase | `src/map/mapper/mapperMatch.c:Map_MatchNodePhase` (line266) | Compares completed cut matches, saves `pCutBest[phase]`, performs reference/area-recovery bookkeeping. |
| Final wiring | `src/base/abci/abcMap.c:Abc_NodeFromMapPhase_rec` (line410) → `Abc_NodeFromMapSuper_rec` (line366) | Builds `pNodePIs` in cut-leaf order with selected polarity; follows stored `pSuper->pFanins`. A variable supergate's Num indexes the leaf. No new timing-based physical pin selection occurs here. |

There is no standalone permutation field that dynamically sorts arrivals in `Map_Match_t`. The permutation is encoded in the selected `Map_Super_t` implementation tree; phase is a separate bitmask. **ABC actively evaluates alternative precomputed permutations for timing/cost**, rather than using one immutable NAND4 ordering. For this cut/output phase both runs expose24 NAND4 permutations, all within the matching loop's30-supergate cap. Both C and D alternatives were actually attempted; they were not artificially injected.

## 3. Recorded input context at first delay-DP matching

The relevant values are mapper **all-PI**, phase-specific, mode0 values. Rise/fall/worst are equal here, but instrumentation records them separately. They are not the post-map PI4-restricted arrivals.

| Ordered slot | Baseline arrival ps | Candidate arrival ps | Difference |
|---:|---:|---:|---:|
| 0:139↔338 | 109.239990 | 109.239990 | 0 |
| 1:!176↔!339 | 108.069992 | 108.069992 | 0 |
| 2:213↔322, R9 output | **109.239990** | **106.219986** | **−3.020004** |
| 3:238↔340 | 109.239990 | 109.239990 | 0 |

Cut area-flow cost is58.1298523 baseline vs53.1589432 candidate; within **each** context all these NAND4 permutations have the same cost, library area1.39999998 and fanout limit1. Thus the decisive within-context comparison is arrival, not an area or cell-type tradeoff. The cost difference between networks is also observed, but is not needed to explain this particular permutation switch.

Physical pin MIO block delays are A15.5900002, B16.7299995, C17.5300007, D18.1800003 ps, equal across rise/fall in this fixed-delay library. Supergate2046 and2047 attach these delays to different ordered leaves:

| Template | Critical slot1 pin | R9 slot2 pin | [A,B,C,D]→leaf index |
|---|---|---|---|
| 2046 | D | B | [0,2,3,1] |
| 2047 | C | D | [0,3,1,2] |

## 4. Baseline decision and candidate decision

The following are mode0 timing evaluations. A rejected bounded attempt may contain only partial rise/fall values and a large Worst sentinel. Full maxima for such attempts below are explicitly **audit recomputations from recorded inputs/arcs**, not an alternative mapping run. All non-rejected recomputations agree with recorded timing within0.00002 ps.

| Context | Template / critical pin | Complete max ps | Area-flow cost | Executed outcome |
|---|---|---:|---:|---|
| Baseline | 2046 / D | 126.769989 | 58.1298523 | eligible, selected_in_cut; final best2046 |
| Baseline | 2047 / C | 127.419991 (audit recomputation) | 58.1298523 | rejected by delay_limit126.769989 |
| Candidate | 2046 / D | 126.769989 | 53.1589432 | eligible, not selected; tied current incumbent |
| Candidate | 2047 / C | 125.969986 | 53.1589432 | eligible, selected_in_cut; final best2047 |

Why baseline D: assigning R9's109.239990 ps input to slow D under2047 produces127.419991 ps. In `Map_TimeCutComputeArrival`, that exceeds the tightened126.769989 ps limit, so2047 aborts and `Map_MatchNodeCut` retains a better match. Template2046 places the earlier critical-slot arrival108.069992 on D and R9 on B. Its maximum126.769989 comes from slot3 on C. Another later template2048 ties2046 in delay/area/fanout/fanin count; the comparator's incumbent preference preserves2046.

Why candidate C: the same R9 logical input now arrives106.219986 ps and can take D without dominating timing. Template2047 gives this R9 input124.399986 ps after its D arc, critical slot1 about125.599991 ps after C, and slot3 about125.969986 ps after B. The complete maximum125.969986 improves the incumbent126.769989 by0.800003 ps, much more than epsilon0.001. The delay-first branch of `Map_MatchCompare` accepts2047. The result is chosen for all-PI timing; **ABC is not directly optimizing PI4's source-restricted traceback**.

Recorded mode0 selected-in-cut histories:

- Baseline:2043 (127.419991) →2046 (126.769989).
- Candidate:2043 (127.419991) →2044 (126.769989) →2047 (125.969986).

`delay_dp` snapshots already contain2046 vs2047 on the same logical cut. Final snapshots after mode3 retain those templates. Final physical connections exactly implement their stored trees, so covering/netlist derivation is not where the reassignment originates.

## 5. Earliest relevant difference and limits of attribution

Within this NAND4's corresponding selected cut, the first semantically relevant recorded input difference is **slot2's all-PI arrival before mode0 comparison** (and its associated upstream area-flow cost). Truth, output phase, logical slot order, input mask and available24 NAND4 templates are unchanged.

The first observed **selection-outcome** divergence is template2044, not2047: its physical ordering is [0,1,3,2]. Baseline2044 ties the incumbent2043 at127.419991 and is not selected; candidate2044 improves to126.769989 and is selected because the R9 input is now early enough for D. Subsequent comparisons lead to the final2046/D vs2047/C difference.

Classification: **A, changed input arrival, directly changes B's selected supergate/permutation**. The available matching set and phase do not change. Different cut indices are local numbering, not evidence of a changed logical frontier. C (phase change) and D (netlist-only reassignment) are ruled out for this connection. Library templates were fixed before either context was evaluated.

This is the earliest decisive difference **at the audited NAND4**, not a claim to have traced the first differing event anywhere in the whole network. The upstream reasons for the R9 region-output mode0 arrival change are outside this experiment. In particular,−3.020004 ps at this mode0 region output must not be replaced by the previous−2.7300033 ps final best arrival delta at the different function197/287.

Area-recovery attempts use different input arrivals/limits from mode0. The CSV preserves each context, bounded raw results and explicit audit maxima separately. Final stored best/node arrival126.769989 vs126.409988 is mapper pass bookkeeping, not a freshly evaluated PI4 pair delay; baseline mode3 attempt maxima can differ from the retained stored match. We do not infer a later optimization from partial/sentinel records or conflate those states with the final network timing. The mode0 choice and its persistence to final wiring are directly recorded.

## 6. Relation to the0.65 ps pair-delay improvement

The final source-restricted critical-slot input has108.329994 ps arrival in both networks. Its selected arc is D18.1800003 vs C17.5300007. The complete traced chain remains logically equivalent with the same critical predecessor functions. Its downstream NAND4 arrival becomes126.509995→125.859993 ps, and float32 propagation yields PO11 194.169998→193.519989 ps. Delta−0.650009 ps is therefore the **consequence** of the all-PI-driven pin selection, not the objective value used by the mapper to select it.

## 7. Verification, artifacts and reproduction

No new ABC patch/build was necessary: reused existing guarded `mapperDebug.c` hooks in the independent experimental binary. `AIGREWRITE_MAP_DEBUG_NODE=385` for baseline and365 for candidate limit cut/match observation to this subject function. Existing node, truth, inventory, attempt, stage-best and final-cover records were collected; no other candidate/path was tested. MFFC/LiteCut/core mapper remain untouched.

Run directory: `/data/wushangge/runs/CULS-R9-pin-audit-20261014-iLFKEK` (RUN).

```text
AIGREWRITE_MAP_DEBUG_NODE=<385 or365> PRIOR/bin/abc_pair_timing -c
  'read PRIOR/input/asap7.lib; read PRIOR/input/<c1355_orig.aig or S002.aig>;
   map; write_blif RUN/<variant>.blif; topo; stime; print_stats;
   print_pair_delay -tv po11 pi03'
```

Reference runs use the original source-pair experimental binary without mapper debug, identical library/inputs/options/order. Debug and reference mapped BLIF bodies are byte-identical (excluding timestamp comment), and match the historical original BLIFs. Per-variant cell counts/Liberty area match: baseline382/513.22; candidate381/509.72. These are invariance checks, not the pair-delay timing model. MIO/statistics area is a different metric; whole-network BLIF identity also protects that result.

Pair delays reproduce exactly to printed9g values; threshold0.00002 ps. Fresh whole-network CEC passes. MIO selected pins have real non-unit delays. `map` performs MIO delay derivation/transfer; `stime` is retained only to reproduce historical command order, not claimed to initialize MIO timing.

Artifact identities:

- Baseline AIG SHA256 `2602b10f131a6ba5996521036ba130aab128a4aa15e3ab49ce44f2f02047f0b0`.
- Candidate AIG SHA256 `59dbe54c03d28c336775785e42d3e4781713e66e96769ac688fd38977672b5e1`.
- Liberty/SCL source and derived-MIO input library `asap7.lib` SHA256 `819ecfc2609a1f53b9d298bf0cba75af0f1a8c737f33d4bbc5c9fb1156e8fa55`.
- Observer binary SHA256 `f255acd802f928fb011737843419d1b39454f060bbd74968a355d3e476822784`; original reference binary `aa2d17b6de367acd16a622aa894f96da0e92431927dbb68e1ca26462b696c1ae`.

Slurm jobs49510 (observe),49512 (inspect),49515/49518 (validate/export): COMPLETED/exit0; each1 CPU, at most2G requested, observation/validation about3 seconds. Original frozen binaries were never overwritten. RUN holds raw logs, same-variant mapped BLIFs, manifest checks, CEC and JSON evidence. `../run_r9_pin_audit.py` is the staged analysis/reproduction runner. `R9_NAND4_PIN_DECISION_COMPARISON.csv` contains20 rows: mode0 templates2043/2044/2046/2047 and recovery modes1/2/3 templates2046/2047 for both variants. JSON proof/state extract: `R9_NAND4_PIN_DECISION_EVIDENCE.json`.

CSV semantics: leaf arrays use logical cut order; `physical_pin_to_leaf` maps physical [A,B,C,D] to those indices; transition-delay vectors are leaf-indexed. `recorded_output_*` for `delay_limit` are incomplete/sentinel and must not be treated as a measured complete arrival. `audit_unbounded_output` is an explicitly labeled arithmetic reconstruction from observed state, not a forced test or new predictor. Load is UNAVAILABLE: the mapper uses fixed supergate arcs, not a per-attempt Liberty load query.

## 8. Availability for a future predictor

| Availability | Potential information | Limitation |
|---|---|---|
| Before rewrite, from baseline mapping/library | Baseline frontier and phase, leaf arrival vectors, pin asymmetry, allowed permutations/order, selected template, epsilon/cost rules, baseline source-restricted critical branch/margin | Depends on retaining intermediate mapper state; full-remap outcomes are not known. |
| Candidate-local structural/analysis work | Rewrite logic/cut truth, local function correspondence, possible changes to local/region transfer behavior | Exact updated region-output all-PI arrival at this downstream matching event is **not demonstrated obtainable** from the candidate cut alone; cut root197 differs from region root213. |
| Current evidence requires candidate mapping | Actual stage-specific leaf arrivals, changed upstream area flow, required limits, recovery/reference context, new downstream best supergate, final trace/PO delay | These were observed via full remap here; cannot be advertised as free online inputs or used as leakage-prone labels/features. |

Research implication: pin-delay-aware alternative-match evaluation needs the **full leaf arrival context**, not only a candidate-local scalar or the source-critical input arrival. A candidate speeds a competing branch, letting that branch occupy a slower pin and freeing a faster pin for the unchanged source-critical signal. Arrival uncertainty, phase/frontier correspondence, permutation availability and incumbent/tie behavior may be useful predictor context. This one case establishes a decision mechanism, not a fast method to obtain exact candidate-context arrivals or predictive accuracy. No model, one-round/full-dynamic update or optimization was implemented. Stop after this audit.
