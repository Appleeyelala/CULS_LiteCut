# Gate 3C report - c1355_orig

## Scope

The same c1355_orig AIG was used. The raw enumeration calls the existing GPU rewrite cut enumeration and candidate matching path with `use_zeros=false`; candidate identities come from the GPU `EvaluateNode` mask export, and no replacement is performed.

Terminology is fixed as follows:

- MFFC Region: a Gate 3A strict maximal standard MFFC.
- Rewrite Decision: one legal `(root, cut_slot)` considered by the GPU rewrite evaluator; its candidate count may be zero.
- Executable Rewrite Decision: a Rewrite Decision with at least one legal replacement candidate.
- Rewrite Region: the decision's actual cut-bounded removable cone from GPU rewrite `CalcMFFC/Decrease` ref-deref semantics.
- Replacement Candidate: one `subgraph_index` under a Rewrite Decision.

## Alignment

- strict maximal MFFC regions: `62`
- eligible GPU rewrite cuts: `1815`
- total legal rewrite decisions: `1815`
- executable rewrite decisions: `91`
- INTRA_REGION decisions: `1815`
- CROSS_REGION decisions: `0`
- executable INTRA_REGION decisions: `91`
- executable CROSS_REGION decisions: `0`
- replacement candidates: `326`
- regions with an opportunity: `54`
- regions without an opportunity: `8`

Classification uses `{ owner[n] | n in removable_nodes }` only. Cut leaves are retained as decision metadata and are not used to reject boundary-leaf decisions.

## Outputs

`region_to_decisions.json` indexes every region by all decisions whose removable set touches it. A CROSS_REGION decision is therefore listed under each touched region and has no single `decision_owner_region`.

Gate 3C PASS: all legal decisions have a non-empty removable set, a complete owner-based classification, and a deterministic Region -> Decision -> Candidate index. Decisions with zero candidates are retained and do not count as rewrite opportunities. LiteCut, gradient, joint optimization, replacement, and STA were not executed.
