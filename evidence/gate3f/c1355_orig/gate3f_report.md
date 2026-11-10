# Gate 3F - LiteCut Choice vs Real ABC Timing

## Fixed case

- original snapshot: c1355_orig.aig
- target PO: PO11
- region path: R6 -> R9 -> R24 -> R36 -> R37 -> R38
- fixed decisions: R9/(root=197, cut=2), R24/(root=386, cut=2)

## Timing results

| Config | R9 | R24 | PO11 arrival (ps) | Global delay (ps) | Area | Gates | CEC |
|---|---:|---:|---:|---:|---:|---:|---|
| GG | G | G | 194.10 | 425.05 | 521.15 | 396 | PASS |
| GL | G | L | 194.30 | 389.02 | 504.58 | 391 | PASS |
| LG | L | G | 193.90 | 369.60 | 519.28 | 397 | PASS |
| LL | L | L | 193.60 | 388.94 | 502.02 | 396 | PASS |

## LiteCut local choices

- R9 root 197: G=24.503954, L=23.639927; local preference=L
- R24 root 386: G=38.550567, L=35.148452; local preference=L

## Effects

- PO11: R9 deltas = -0.20 ps / -0.70 ps; R24 deltas = 0.20 ps / -0.30 ps; interaction = -0.50 ps (MEASURABLE_INTERACTION).
- global: R9 deltas = -55.45 ps / -0.08 ps; R24 deltas = -36.03 ps / 19.34 ps; interaction = 55.37 ps (MEASURABLE_INTERACTION).

## Alignment and determinism

- Case classification: Case D - timing active + measurable interaction.
- Candidate-level post-mapping correspondence: NOT_ALIGNABLE for roots 197 and 386. No nearest-node matching was used.
- Both replay and ABC timing were repeated independently. See `determinism.json`.

Gate 3F status: PASS.
