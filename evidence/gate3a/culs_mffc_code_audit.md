# CULS MFFC code audit

This Gate 3A audit uses the repository's original PO-to-PI discovery
front half and does not call the refactoring back half.

## Actual call chain

```text
PO drivers
  -> initial roots (AND PO drivers, node-id sort/unique)
  -> recordMFFC()
       -> Aig::findReconvMFFCCut()
            -> findReconvMFFCCut_iter()
            -> ref-count-guided cut expansion
  -> setStatus()
       -> AND nodes in the current cut whose vCutSizes == -1
  -> next roots
  -> repeat toward PI/constant boundaries
```

Relevant implementation points:

- `src/algorithms/refactor_mffc.cu`: `recordMFFC`, `setStatus`, and the
  discovery loop in `refactorMFFCPerform`.
- `src/aig/mffc.cuh`: `findReconvMFFCCut` initializes the root and its
  fanins, then `findReconvMFFCCut_iter` expands only a zero-reference
  cut node.
- `src/aig/truth.cuh`: the shared `collectCutConeNodes` traversal is used
  by the existing truth path and by the new audit-only dump kernel.

## Cut limit semantics

`findReconvMFFCCut_iter` returns without expanding when
`vCutSizes[root] - 1 + bestCost > nMaxCutSize`. Thus `nMaxCutSize` is
a real early-stop condition. The audit retains every discovered
discovery cone, including singleton cones; it does not apply the later
refactor candidate filter `isSmallMFFC` (`cone_size < 2`).

Accordingly, a dumped region is not called a complete MFFC until it is
compared with the independent unlimited CPU deref/ref oracle. An exact
match is `STANDARD_MFFC`; a strict subset is classified as
`CUT_LIMITED_FFC`; any non-subset is a `MISMATCH` and blocks the gate.

## Audit boundary

The Gate 3A command stops after cut discovery and the shared
cut-bounded internal-node traversal. It does not execute truth-table
factorization, candidate evaluation, resynthesis, replacement, LiteCut,
STA, rewrite, or gradient code.
