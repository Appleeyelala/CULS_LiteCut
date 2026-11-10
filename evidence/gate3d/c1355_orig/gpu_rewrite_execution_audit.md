# Gate 3D — Original GPU rewrite execution semantics audit

Input path is the same `c1355_orig.aig`; this document describes the current
`src/algorithms/rewrite.cu`, not the earlier Gate 1 implementation.

## Actual flow

1. `CutEnumerate` (`rewrite.cu:273`) enumerates and filters
   cuts level by level.
2. `EvaluateNode` (`rewrite.cu:383`) evaluates every root in
   parallel. It tests all eligible cuts/subgraphs, exports candidate masks, and
   retains only one best `(cut_slot, subgraph_index)` per root. With
   `use_zeros=false`, only positive `saved - nodesAdded` candidates can be
   selected. Thus PRESELECTED means one root-level best decision, not every
   executable Gate 3C decision.
3. `ReplaceSubgr` (`rewrite.cu:522`) runs in parallel and
   builds the replacement DAG. Its `replace[root]` value is copied back and
   generated nodes are re-IDed before installation. A `replace[root] == -1`
   result is the current full-correspondence/no-new-replacement case and is
   reported as `FULL_CORRESPONDENCE` by the audit while preserving the source
   counter's NEW_IDX rejection branch.
4. The CPU install loop (`rewrite.cu:1119`)
   normally visits internal roots in ascending reordered ID. Gate 3D records
   this order without changing it. The controlled replay changes only the
   selected region's relative order.

## Status mapping

- `PRE_EVAL_REJECT`: `selectedCuts[root].used == 0`.
- `SMALL_CUT_REJECT`: selected cut has fewer than three leaves.
- `FULL_CORRESPONDENCE`: source `newIdx == -1` branch with `replace == -1`.
- `NEW_IDX_REJECT`: the remaining invalid replacement-node case.
- `NO_OP`: source's `newIdx == root` or replacement fanin contains root branch.
- `COMPROMISED`: the source cut-leaf deletion test
  (`rewrite.cu:1212`) found an already deleted leaf.
- `ACCEPTED`: source accepted `saved - added > 0` (or zero gain when
  `use_zeros=true`; Gate 3D uses false).
- `REVERTED`: source installed and deleted speculatively, then restored the
  old root because the gain condition failed.

For ACCEPTED/COMPROMISED/REVERTED, the trace includes the pre-execution
cut-bounded removable set, cut-leaf deletion state, and saved/added values.
For COMPROMISED, `damaged_by_prior_root` is the first earlier accepted root
whose recorded deletion set contains the damaged leaf, when available.

The audit does not invoke LiteCut, gradient, joint optimization, replacement
outside the original rewrite path, STA, or technology mapping.
