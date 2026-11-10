# Gate 3D report — c1355_orig

## Result

- legal Gate 3C decisions: **1815**
- executable Gate 3C decisions: **91**
- preselected root-level decisions: **74**
- final ACCEPTED: **62**
- COMPROMISED: **8**
- REVERTED: **4**
- active regions: **54 / 62**

The preselected count is a root-level count: the current GPU evaluator keeps
one best `(cut_slot, subgraph_index)` per root. It is therefore not required
to equal the 91 executable `(root, cut_slot)` decisions.

## Accepted rewrites per active region

- 0 accepted: **4**
- 1 accepted: **40**
- 2 or more accepted: **10**
- maximum accepted rewrites in one region: **3**
- mean accepted rewrites per active region: **1.148148**
- max-region IDs: `[16, 24]`

## Compatibility and conflicts

`intra_region_accepted_pairs.csv` classifies every pair of decisions that were
both ACCEPTED in the same original rewrite order. Relation counts:
`{"DISJOINT": 8, "OVERLAPPING": 6}`.

There are **8** damaged cut-leaf
events in COMPROMISED decisions; **8**
are traced to an earlier accepted deletion set. See
`intra_region_conflicts.csv` for the causal edges.

## Order sensitivity

The controlled replay summary is:
`{"any_order_sensitive": true, "regions_tested": 3, "results": [{"order_sensitive": true, "original_accepted": [[74, 4], [86, 5]], "region_id": 2, "reverse_accepted": [[86, 5], [90, 6]]}, {"order_sensitive": true, "original_accepted": [[123, 2], [135, 5]], "region_id": 5, "reverse_accepted": [[135, 5], [139, 7]]}, {"order_sensitive": true, "original_accepted": [[160, 4], [172, 5]], "region_id": 7, "reverse_accepted": [[172, 5], [176, 7]]}]}`.

## PASS

Gate 3D provenance covers every original root in the CPU sequential install
loop, maps preselected roots to Gate 3A regions, records the actual source
status branches, exports the pre-execution removable/cut-leaf state for the
relevant outcomes, and emits deterministic artifacts. The main result keeps
the original GPU choice strategy and original ascending reordered-root order.
