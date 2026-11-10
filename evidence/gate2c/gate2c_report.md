# Gate 2C — Timing-Active Rewrite Decision Screening

Status: **PASS**

The sample and candidate-pair rule were frozen before replay timing was run.
This run uses the Gate 1 engine for candidate provenance, the existing Gate 2A mapping-free LiteCut bridge, and the exact Gate 2B `map; topo; stime` flow.

## Frozen sampling

- Captured sampling frame: **512** Gate 1B records; engine `records_seen` was **1723**, but the uncaptured tail was not sampled.
- Sample rule: `deterministic_sorted_prefix`, sort by `(root, cut_slot)` ascending, filter `candidate_count >= 2`, take first **50**.
- Selected candidates replayed: **109** across **50** decisions.
- No STA or LiteCut result was used to select a decision or candidate pair.

## Results

- TIMING_ACTIVE: **24**
- TIMING_INACTIVE: **26**
- BORDERLINE: **0**
- TIMING_ACTIVE fraction among valid decisions: **0.480000**
- Fixed tolerance: **0.01 ps**; reported precision: **0.01 ps**.

| pair type | decisions | active | inactive | borderline |
|---|---:|---:|---:|---:|
| physical-vs-physical | 3 | 0 | 3 | 0 |
| no-op-vs-physical | 47 | 24 | 23 | 0 |

## LiteCut comparison

- LiteCut local score difference but timing-inactive (strictly below tolerance): **26**.
- LiteCut local score difference but whole-design delta within tolerance including borderline: **26**.
- Among TIMING_ACTIVE decisions, same direction: **9**, opposite direction: **15**, tie/uncertain: **0**.
These are single-decision perturbation direction comparisons, not LiteCut global accuracy.

## Next Gate pair

- Pair search status: **FOUND**.
- Selected compatible timing-active decisions: `c1355_r000148_c000_n004, c1355_r000149_c000_n006`.
- Compatibility was established by all four no-STA joint replays and CEC checks; no joint timing or interaction experiment was run in Gate 2C.

## Required answers

1. Q1: **24** of **50** valid sampled decisions are TIMING_ACTIVE.
2. Q2: TIMING_ACTIVE fraction is **0.480000**.
3. Q3: **26** have a nonzero selected-pair LiteCut local delta while whole-design timing is TIMING_INACTIVE; **26** are within tolerance including borderline.
4. Q4: Among TIMING_ACTIVE decisions, LiteCut and whole-design direction agree **9**, disagree **15**, and are tie/uncertain **0** times.
5. Q5: A compatible pair for the next Gate was **found**.

## Artifacts

- Summary: `/data/wushangge/gate2c_runs/gate2c_sm86_20260918/output/gate2c_summary.json`
- Sampling manifest: `/data/wushangge/gate2c_runs/gate2c_sm86_20260918/output/sampling_manifest.json`
- Candidate pool: `/data/wushangge/gate2c_runs/gate2c_sm86_20260918/output/candidate_pool.json`
- Screening CSV: `/data/wushangge/gate2c_runs/gate2c_sm86_20260918/output/decision_screening.csv`
- Active decisions: `/data/wushangge/gate2c_runs/gate2c_sm86_20260918/output/timing_active_decisions.json`
- Next-Gate pair candidates: `/data/wushangge/gate2c_runs/gate2c_sm86_20260918/output/next_gate_pair_candidates.json`
