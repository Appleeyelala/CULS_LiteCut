# R9 single-candidate investigation — STOPPED

## Object and established historical results

One fixed candidate: `root197_cut2_subgraph2`, cut `[17,33,151,188]`, region R9 root213. CEC-backed corresponding candidate function point is serialized287, not serialized197. Inputs are the existing mapper-internal experiment's `c1355_orig.aig` and `S002.aig`.

Verified input SHA256:

- Baseline: `2602b10f131a6ba5996521036ba130aab128a4aa15e3ab49ce44f2f02047f0b0`
- Candidate: `59dbe54c03d28c336775785e42d3e4781713e66e96769ac688fd38977672b5e1`

Historical source-restricted MIO results: baseline194.169998 ps, candidate193.519989 ps, delta−0.650009 ps. Historical all-PI mapper positive arrivals:93.4799957 and90.7499924 ps. These two kinds of timing are not interchangeable.

## Reuse attempt and failed invariant

Run directory: `/data/wushangge/runs/CULS-R9-case-20261010-MspaQd`; Slurm49256 COMPLETED, exit0, elapsed3 seconds, requested1 CPU/2G. No ABC source or instrumentation was changed.

The existing mapped BLIFs and experimental binary were reused from `/data/wushangge/runs/CULS-mapper-internal-20261003-34U6OZ`.

For each variant the attempted command was:

```text
read <prior>/input/asap7.lib;
read <prior>/output/<variant>_original.blif;
topo;
print_pair_delay -tv po11 pi03
```

Both variants returned11 ps, with every selected arc delay equal to1 ps. Raw evidence: `baseline.log`, `candidate.log` in the run directory. Upper-bound and single-source consistency checks pass **within this incorrectly restored timing context**; they do not certify equivalence to the historical timing model. Thus no new critical traceback is accepted as scientific evidence.

The historical command instead read the original AIG, ran `map; write_blif ...; topo; stime; print_stats; print_pair_delay po11 pi03`. The BLIF reuse attempt did not reproduce the same initialized MIO timing context. Exact cause of the unit-delay state has not been audited; omission of `stime` is a concrete flow difference, not yet a demonstrated cause. BLIF cell names alone do not certify restored pin delays.

Stop condition4 is triggered: failure to reproduce known pair delay far beyond float error. Mapping/timing environment equivalence is also not established. No remapping, additional instrumentation, or further analysis was performed after this failure.

## Requested deliverables not available

- `R9_CASE_baseline_trace.csv`: NOT AVAILABLE — attempted trace uses incorrect unit-delay state.
- `R9_CASE_candidate_trace.csv`: NOT AVAILABLE — same reason.
- `R9_CASE_timing_trace_comparison.md`: NOT AVAILABLE — valid comparable traces missing.
- `R9_CASE_mapper_cover_changes.md`: NOT AVAILABLE — cannot connect historical covers to validated new timing traces.

## What remains established / unknown

Existing mapper audit proves target function point has zero final-cover references in both variants; selected larger covers absorb it, and the surrounding cover relations change. This does not prove which changes are on the source-restricted critical traceback. No direct attribution of−2.7300033 ps mapper arrival change to−0.650009 ps pair-delay change is justified.

New evidence is insufficient to determine cell/arc changes, critical predecessor switching, or the mechanism responsible for the historical improvement.

## Minimal next repair (not executed)

Restore the exact historical sequence on the same frozen baseline and accepted candidate AIGs, including `map; topo; stime`, then request `print_pair_delay -tv po11 pi03`. Alternatively test whether the identical timing initialization after BLIF loading restores all pin delays. Accept either route only after reproducing194.169998 and193.519989 ps and verifying non-unit MIO arcs. This uses `stime` solely as historical context initialization; measured quantities must remain source-restricted MIO `print_pair_delay`, not Liberty stime delay.

## Research Implication

Reliable candidate→path analysis needs reproducible library/pin timing initialization in addition to function-point and selected-cover correspondence. This stopped run supports no new timing-impact model or causal mechanism.
