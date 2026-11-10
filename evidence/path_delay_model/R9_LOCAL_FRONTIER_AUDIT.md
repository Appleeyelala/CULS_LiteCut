# R9 local frontier audit

Only c1355, the existing ACCEPTED/CEC-PASS `root197_cut2_subgraph2`, and positive-phase mode0 arrival at the R9 region root are studied. No MFFC/candidate regeneration or candidate full mapping is performed.

## Structural window (not automatically a mapper boundary)

Actual replay manifest identifies cut `[17,33,151,188]→197`; region root is213. Manifest reports saved8/added6 and actually deleted original nodes `[189,190,191,192,194,195,196]`. These describe structural edits, not a change in the equivalent Boolean function. Serialized IDs across variants are not node identities.

New function-only CEC checks, using the original accepted AIG snapshots, prove:

| Baseline function | Candidate serialized function | Polarity | CEC |
|---:|---:|---:|---|
| 17 | 17 | 0 | PASS |
| 33 | 33 | 0 | PASS |
| 151 | 232 | 0 | PASS |
| 188 | 234 | 0 | PASS |
| 197 (rewrite cut root) | 287 | 0 | PASS |
| 209 (external region input) | 235 | 0 | PASS |
| 213 (region root) | 322 | 0 | PASS |

Cut-to-region connectivity:

```text
baseline: 197 → 210 → {211,212} → 213, with external function209
candidate:287 → 305 → {313,314} → 322, with external function235
```

The structurally delimited root cone, stopped at the five corresponding inputs, contains:

- Baseline frontier `[17,33,151,188,209]`;13 AND nodes `[189,190,191,192,193,194,195,196,197,210,211,212,213]`.
- Candidate frontier `[17,33,232,234,235]`;11 AND nodes `[63,250,251,268,277,278,287,305,313,314,322]`.

The fanin literals are in the raw structural audit under the /data run. Candidate node63 represents reused logic; counting every different serialized number as a new function is invalid.

## Why five scalar arrivals are not a cut-closed context

ABC merges all retained child cuts (`Map_CutCompute`/`Map_CutMergeLists`), including cuts that expand through a nominated boundary node. The prior baseline197 dump includes supports such as `[143,147,188,193]`, which cross the five-input structural frontier. The captured baseline frontier nodes themselves retain nontrivial upstream cuts. Freezing each as an opaque pseudo-PI would remove mapping options and change cut/match semantics. The fixed four-leaf rewrite cut also has no direct supergate match; a single-cell transfer assumption is not valid.

No opaque five-input mapper, protected-boundary map, or forced cut was used. Exact compact-window reuse would require caching/cloning upstream cut DAGs, phase timing, area flow/reference context and ordering; that optimization is not implemented here.

## Adopted reliable timing frontier: PI closure

Conservatively expand the computation to **all transitive fanins of the one R9 root**, down to these14 PI functions:

`[3,4,5,6,9,12,13,16,17,21,30,33,34,38]`.

Both variants have the same PI interface, with standard AIG PI indices and CEC-proved root equivalence. Baseline positive PI seeds are all0, and negative PI arrivals follow the native inverter semantics; the baseline context preserves both phases. No internal hard boundary remains for cuts to cross. Choices are absent; the prototype refuses unsupported choice/node-delay context.

The native mapper subject graph is built from the whole AIG to preserve actual object correspondence, Num/topological order and structural fanout counts. Before cut/truth/matching calls, its processing vector is restricted to the root's TFI plus PI seed objects. **Only49 baseline or47 candidate ANDs** are evaluated, versus553/551 subject objects constructed. All41 PI seed slots are initialized by the native API, but only14 feed this root.

This is a reliable PI-closed local cone, not a successful13/11-node incremental frontier. Unchanged upstream cone logic is recomputed too. Full subject-graph construction is disclosed as global **structural preprocessing**, not hidden as a purely local operation; no candidate outside-cone cut/match, area recovery, cover, mapped netlist, NAND4 decision or PO timing is calculated.

Outcome: frontier is reliable after PI closure. A tighter boundary remains a future context-caching problem. See `R9_LOCAL_ARRIVAL_FEASIBILITY.md` for the native baseline sanity check and held-out label comparison.
