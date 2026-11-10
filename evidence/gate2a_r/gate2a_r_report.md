# Gate 2A-R — Gate1 candidate to canonical mapping alignment audit

## Result: NOT_ALIGNABLE

Scope was deliberately limited to d1=(root 146, cut slot 0), candidates A and B. No Student inference, STA, or Gate 1C-R modification was performed.

Canonical flow used for each replay: `read asap7.lib; read <candidate>.aig; rewrite -c; map; write_verilog`, with `/home/wushangge/abc_for_cutLib/abc`, followed by the unmodified `Aig2MapParser._build_graph_payload()` path.

| candidate | Gate1 kind | Gate1 cut | Gate1 subgraph | AIG→mapper root | direct `abc_map`/`superCreate` root supergate | parser selected index | canonical first cut |
|---|---|---|---|---:|---|---:|---|
| A | NO_OP_FULL_CORRESPONDENCE | `[2, 42, 82]` | 0 | 216 | no | -1 | `[8738, [9, 215], 2]` |
| B | PHYSICAL_REPLACEMENT | `[2, 42, 82]` | 1 | 218 | no | -1 | `[8738, [13, 216], 2]` |

### Answers

1. A/B replayed AIGs both map successfully and the canonical parser extracts the surrounding graph, but neither preserved root has a direct selected canonical supergate: `supergateIndex=-1`, no matching root record in `abc_map.log`, and no matching `Creating Super` event.
2. `aig2map.log` does provide an AIG→mapper correspondence (A: 146→216; B: 146→218). This is not a stable LiteCut supergate identity; matching the mapped netlist would require function/boundary matching.
3. A/B canonical supergates cannot be declared different or equal because neither root has a selected supergate/subg. The audit does not manufacture one from a nearby node.
4. The canonical root cut interface is not the Gate1 fixed cut `[2, 42, 82]`: after translating mapper IDs back to the replayed AIG namespace, A's first cut is `[10, 94]` and B's is `[14, 94]` (both two leaves). Mapping therefore reselects a different interface; this is boundary change, not an aligned Gate1 candidate.
5. Mapping fusion is directly visible: A's root 146 is a fanin of AIG 194 and B's root 146 is a fanin of AIG 192; those fanout supergates are selected instead of a root-146 supergate. This permits the canonical mapped implementation to cross the Gate1 removable-region boundary.
6. The two fused fanout subgraphs are structurally isomorphic (ID-independent hash `9449c0d1ca77a3162fabb4562ddbcb6f` for both), but this is not a root-candidate correspondence; root-level many-to-one collapse remains undetermined.

Per the gate rule, C/D are not promoted to an alignment result after A/B fail the direct correspondence test. Their raw canonical logs remain under `mapping_C/` and `mapping_D/` only as reproducible tool outputs.

## Reproducibility artifacts

- `candidate_A_mapping_alignment.json`, `candidate_B_mapping_alignment.json`: Gate1 provenance, mapper IDs, canonical cuts, and direct-supergate checks.
- `mapping_A/`, `mapping_B/`: complete ABC logs and parser payloads.
- `gate2a_r_alignment_audit.json`: machine-readable result and decision gate.
