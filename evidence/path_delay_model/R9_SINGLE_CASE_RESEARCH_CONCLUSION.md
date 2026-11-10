# R9 Mapping-Aware Path Impact: one verified case

## Experiment and result

c1355; one fixed ACCEPTED/CEC-PASS `root197_cut2_subgraph2`; region R9 root213; rewrite cut `[17,33,151,188] → 197`; candidate corresponding function287. Only PI4 (`pi03`)→PO11 was investigated. Cached independent replay artifact S002 was identity-checked and freshly CEC-checked; no additional candidate was applied.

Mapped MIO fixed-delay source-restricted timing was restored and reproduced:194.169998→193.519989 ps, delta−0.650009 ps. Full trace arc accumulation, predecessor connectivity, upper bound and single-source checks pass. Read-only instrumentation does not change mapped netlist bodies or measured timing. See `R9_TIMING_ENV_REPRODUCTION.md` and the two trace CSVs.

## Mapper and selected-cover evidence

The prior mapper audit remains valid: baseline function197→mapper196 best positive arrival93.4799957 ps; candidate function287→mapper194 arrival90.7499924 ps. Those are **all-PI** estimates, delta−2.7300033 ps; they are not source-restricted endpoint timing. Function197/287 has no independent implemented mapped output and zero final-cover references.

Existing selected covers cross the point at baseline211/212 and candidate305/313/314/322, including a candidate phase0 A2O1A1I supergate at region root322. Fresh mapped-state connectivity establishes that implemented cells at the proved211↔313,212↔314 and213↔322 functions lie in the PO11 cone. They are **not** on the selected PI4→PO11 critical chain. Both positive and negative region-root implementations occur in that cone; do not assign one unique absorbing gate.

Concrete positive-phase region branch evidence (pre-topo, separate variant ID scopes):

| Original function | Baseline mapped cell / gate / source arrival | Candidate mapped cell / gate / source arrival |
|---|---|---|
| 211, equivalent313 | AOI21xp33 /315 /93.7999954 ps | AOI21xp33 /309 /91.0699921 ps |
| 212, equivalent314 | NOR3xp33 /316 /95.8399963 ps | NOR3xp33 /312 /93.2199936 ps |
| 213, equivalent322 | NOR2xp33 /317 /109.679993 ps | NOR2xp33 /313 /107.059998 ps |

The corresponding cells can retain cell type while changing their own mapped input implementation/pin order. The prior baseline212 phase1 cover is super1303 with leaves `[195,196,209]`; candidate314 phase1 is super1301 with leaves `[277,278,235]`. These numeric leaf namespaces are not equated without function proof. The positive R9 output remains a NOR2 but its arrival falls about2.62 ps. A candidate negative region-root implementation is the A2O1A1Ixp33 cell observed as gate188; it is also in the PO11 cone but not on the critical trace. These are actual remapping observations, not inferred from correlation.

## Verified endpoint mechanism

All11 critical-trace output functions and input frontiers correspond by CEC; the cell-type sequence is unchanged. At downstream function385↔365 (NAND4xp25), input assignment changes from `[AIG139,AIG213,AIG238,!AIG176]` on pins `[A,B,C,D]` to equivalent functions `[AIG338,AIG340,!AIG339,AIG322]`. The critical input176/339 moves from D to C. Its incoming arrival is unchanged108.329994 ps, while its chosen arc becomes18.1800003→17.5300007 ps. The new gate arrival is125.859993 rather than126.509995 ps; the improvement propagates along the same remaining selected logical chain to PO11.

The R9 output213/322 is a competing input at this NAND4, not the critical predecessor. Its own source-restricted arrival improves109.679993→107.059998 ps, while it moves from B to D. Its full input-arc contribution remains below the selected branch on both sides. Therefore the local candidate can affect the endpoint through **downstream pin assignment on another branch**, without the candidate function point itself being an implemented critical gate.

**Mechanism judgment: B is the direct endpoint mechanism (downstream mapping/pin reassignment).** Local cover/timing changes are also observed, so A+B coexist in the wider timing cone; C (logical critical path switching) is not observed. Exactly one selected arc changes in the measured critical chain. Float32 accumulation accounts for the entire−0.650009 ps endpoint difference. Cell size/type changes on the critical chain are not required in this example.

## Observed evidence, verified conclusion, uncertainty

- Observed: local selected-cover changes, faster source-restricted region output, downstream NAND4 frontier permutation, asymmetric MIO pin delays, unchanged critical predecessor functions and downstream cell sequence.
- Verified: the measured endpoint improvement is explained by that downstream selected pin arc plus float rounding; not by summing MFFC delays or using the unused function197 mapper arrival as root timing. The−2.73 ps mapper value has no proved direct additive contribution to the endpoint.
- Uncertain: which detailed DP/area-recovery/tie-breaking comparisons caused this particular permutation to win; whether local all-PI arrival improvement alone would reproduce that choice with all other mapping context held fixed. No counterfactual mapper intervention was done. Final cover alternatives, load/slew changes and other endpoints are not separately evaluated here. In this fixed MIO model per-network load/slew is not an active timing variable, so a Liberty load explanation is not warranted.

## Research Implication

Possible features include baseline mapper cuts/matches, phase-specific local arrivals, candidate structure, fanout/reconvergence location, path-specific boundary arrival and critical margins, and library pin-delay asymmetry/permutation flexibility. Candidate full-remap selected covers, final pin assignment, new critical trace and PO delay are **post-outcome explanatory evidence**, not available online inputs; using them directly in a fast predictor would leak ground truth. Candidate-local mapper state plus frozen baseline path/reconvergence context might expose risks of downstream pin reassignment, but this case does not establish sufficient features or predictive accuracy. A scalar local-arrival delta alone misses the observed off-critical-branch→critical-pin mechanism. No model was trained and no one-round/full-dynamic/optimization experiment was performed.

## Artifacts and stop

Results: `R9_TIMING_ENV_REPRODUCTION.md`, `R9_BASELINE_PAIR_TRACE.csv`, `R9_CANDIDATE_PAIR_TRACE.csv`, `R9_PAIR_TRACE_COMPARISON.md`, `R9_mechanism_evidence.json`, `R9_TRACE_FANIN_ALTERNATIVES.csv`. Script: `../run_r9_mechanism.py`; read-only patch: `../r9_case_trace.patch`. Raw logs, full mapped networks/state, CEC proofs and Slurm scripts remain in `/data/wushangge/runs/CULS-R9-mechanism-20261012-9CY36o`. The earlier2026-10-10 stopped report is preserved as history and superseded by this successful reproduction. Work stops after this single case.
