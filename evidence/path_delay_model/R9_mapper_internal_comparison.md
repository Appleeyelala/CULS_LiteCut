# R9: baseline and one accepted candidate

The cut root is **197**, while maximal-MFFC region root is **213**. CEC proves original function197 equals candidate serialized node287, and the four original leaf functions equal candidate nodes `[17,33,232,234]`. Consequently both sides refer to the same logical point and interface, despite different serialized/ABC/mapper IDs.

| Quantity | baseline | candidate |
|---|---|---|
| AIGER variable of function197 | 197 | 287 |
| ABC pre-map object ID | 229 | 319 |
| Mapper node ID | 196 | 194 |
| Retained nontrivial cuts | 50 | 57 |
| Total including trivial self-cut | 51 | 58 |
| Fixed rewrite cut present | YES, index15 | YES, index11 |
| Fixed cut truth (ordered proved leaves) | `0x78877887` | `0x78877887` |
| Fixed cut direct supergate matches | 0 in both output phases | 0 in both output phases |
| Final best positive cut (variant AIGER namespace) | `[195,196]` | `[277,278]` |
| Final positive supergate | 1669, NOR2xp33_ASAP7_75t_L | 1670, NOR2xp33_ASAP7_75t_L |
| Final positive rise/fall/worst arrival | 93.4799957 each | 90.7499924 each |
| Final negative supergate | 2020, NAND2xp33_ASAP7_75t_L | 2020, NAND2xp33_ASAP7_75t_L |
| Final negative rise/fall/worst arrival | 91.6399994 each | 89.4199982 each |
| Final phase-specific exact area cost | 0.930000007 | 0.930000007 |
| Final target phase refs / total | `[0,0,0]` | `[0,0,0]` |
| Target used as a cover output | NO | NO |

The forward delay-oriented pass's positive best arrival is 93.0399933 vs 90.019989; area recovery subsequently selects final states shown above. Final positive best arrival changes by **−2.7300033 ps**. These are all-PI mapper estimates at the logical function point, not source-restricted PI4→PO11 delay. The fixed four-leaf cut itself supplies no direct match timing, because its truth is absent from the current supergate table. Its match availability does not change across the rewrite. Other cut supports and pin permutations differ; numerical cut/leaf IDs across variants are not treated as equivalent without proof.

## Actual absorption proof

Baseline final cover references selected outputs211 and212 in both phases, with cut leaves `[195,196,209]`. Their subject interior contains197 and210:

| Cover output | Phase | Supergate / cell | Witness |
|---:|---:|---|---|
| 211 | 0 | 1837 / OAI21xp33_ASAP7_75t_L | `211→210→197` |
| 211 | 1 | 1567 / AOI21xp33_ASAP7_75t_L | `211→210→197` |
| 212 | 0 | 2037 / NAND3xp33_ASAP7_75t_L | `212→210→197` |
| 212 | 1 | 1303 / NOR3xp33_ASAP7_75t_L | `212→210→197` |

The candidate's selected covers containing function197 at serialized287 are output305 (phase1/super1567),313 (phase1/super1567),314 (phase0/super2037 and phase1/super1301), and322 (phase0/super1619). CEC separately proves313 represents original211,314 represents original212, and322 represents region root213. Output305 is reported in its serialized namespace; no unproved baseline counterpart is assigned. Thus there are multiple absorption relations, not a manufactured unique absorbing gate. Exact leaves, arrivals and structural witnesses are in the two final-cover reports.

## Judgment

**Mapper-internal correspondence is feasible at the proved function point.** Final mapped-boundary absence does not prevent observing cuts, matching alternatives and best timing. This is CASE1 with a zero-direct-match limitation for the requested four-leaf cut. The target cut was retained in both variants, so cut injection (route B) does not address this observation. Recommend **route A: mapper-internal timing**, keeping direct cut matching and decomposed function-point timing distinct. Native choice support exists, but this experiment supplies no reason to construct a choice network now.

## Correction to the earlier boundary audit

The earlier candidate-side inspection looked up serialized variable197 in the rewritten AIG. The present CEC-backed correspondence proves that rewritten function197 is actually serialized287; those earlier same-number candidate comparisons were not function-point comparisons. The baseline finding that original197 has no separate implemented mapped gate remains valid, and the present selected-cover witnesses explain it directly. This experiment supersedes the earlier candidate-side numeric-ID interpretation.
