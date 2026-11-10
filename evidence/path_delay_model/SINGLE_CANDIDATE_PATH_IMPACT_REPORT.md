# Single-Candidate Path Impact Calibration — c1355 PO11

## Experimental scope

Only isolated, single-candidate rewrites are included. The fixed Gate 3B path is PI node 4 (`pi03`) → PO11, with MFFC Region sequence `R6 → R9 → R24 → R36 → R37 → R38`. Timing means the mapped endpoint-pair delay reported by ABC for that same PI/PO pair; it is not an exact mapped correspondence to the original AIG node sequence. LiteCut delta is candidate local-cone score minus the same decision's original cut-cone proxy.

All 67 candidates in these path regions were selected (not only local-best candidates). 15 previously measured pilot samples were reused after checking CEC, ACCEPTED status, baseline, and replay-AIG hash; 52 candidates were replayed independently from the original snapshot. Current outcome counts: CEC PASS 67/67; accepted 67/67. Valid calibration fits use only ACCEPTED, CEC-passing candidates with endpoint timing.

Baseline endpoint delay was remeasured as **194.40 ps** (required 194.40 ps). ABC `print_delay` prints Arrival values to one decimal place, so the observed timing resolution is **0.1 ps**; an older Gate3F manifest says 0.01 ps, which is finer than the output actually retained here. Path-effect thresholds therefore use 0.1 ps. The sign-neutral epsilon remains the separately configured 0.01 ps; neutral/negative/positive counts are 16/37/14.

## Overall calibration (in-sample descriptive fits)

Valid samples: **67**. Raw LiteCut delta: Pearson 0.0512, Spearman 0.1024, MAE 15.1692 ps. Predicting no endpoint change (`ŷ=0`) gives MAE 1.0537 ps. Global through-origin fit `y=αx`: α=-0.0007, MAE 1.0560 ps. Global linear fit: `y=αx+β`, α=0.0057, β=0.6535, MAE 1.4645 ps. These in-sample fits are descriptive only; model comparison below uses grouped cross-validation.

## Region-level descriptive calibration

| Region | N | α origin | α linear | β linear | Pearson | Spearman | Raw MAE | Origin-scaled MAE | Sign accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R6 | 10 | -0.1152 | 0.1040 | 4.7002 | 0.2504 | -0.1741 | 13.0533 | 4.6195 | 0.1250 |
| R9 | 17 | 0.0085 | 0.0039 | -0.0459 | 0.1287 | 0.1519 | 7.7307 | 0.0978 | 1.0000 |
| R24 | 37 | -0.0020 | -0.0017 | 0.1978 | -0.0358 | 0.1464 | 17.8918 | 0.5495 | 0.5833 |
| R36 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| R37 | 1 | n/a | n/a | n/a | n/a | n/a | 34.7441 | n/a | 1.0000 |
| R38 | 2 | -0.0416 | -0.0415 | 0.0015 | n/a | n/a | 28.8218 | 1.1500 | 0.0000 |

## Five-fold grouped cross-validation

All candidates from one `(region, root, cut)` decision are assigned to the same fold. Region models without any training examples for a held-out region fall back to the fold's global through-origin alpha (fallback predictions are counted below).

| Model | N | MAE (ps) | RMSE (ps) | Pearson | Spearman | Sign accuracy | Sign N | Fallback predictions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| raw_delta | 67 | 15.1692 | 23.1407 | 0.0512 | 0.1024 | 0.5490 | 51 | 0 |
| zero_delta | 67 | 1.0537 | 2.5699 | n/a | n/a | 0.0000 | 51 | 0 |
| global_alpha | 67 | 1.1265 | 2.6234 | -0.0914 | -0.2133 | 0.2353 | 51 | 0 |
| region_alpha | 67 | 1.0832 | 2.6052 | -0.2675 | -0.2561 | 0.2745 | 51 | 13 |
| region_alpha_beta | 67 | 1.1773 | 2.6149 | -0.3278 | -0.4612 | 0.1961 | 51 | 13 |

Held-out region-specific vs global-alpha MAE: 1.0832 vs 1.1265 ps; raw-delta MAE: 15.1692 ps; zero-change reference MAE: 1.0537 ps. Region-specific scaling's nominal held-out MAE reduction over global scaling is 0.0432 ps, below the 0.10 ps observed timing resolution; there is no resolved region-specific benefit, and the zero-change reference is better. Compare both calibrated models to the zero-change reference: a lower MAE than raw delta alone does not establish that LiteCut adds predictive value.

## Cross-path shared-region audit

Shared-region audit measured 128 candidate/path endpoint pairs across 8 candidates and 16 paths (1 distinct PI node, 16 distinct POs). Among 8 candidates with at least two path measurements, 0 had a within-candidate true-delta range greater than the 0.10 ps observed report resolution; maximum observed range was 0.00 ps. No path-dependent impact was visible at the observed report resolution; sub-resolution effects are unresolved.

## Research questions

**Q1. Can LiteCut delta directly serve as path delay delta?** Raw pooled Pearson=0.0512, Spearman=0.1024, MAE=15.1692 ps and grouped-CV MAE=15.1692 ps. This is the direct empirical answer; the local score is not itself a measured path contribution.

**Q2. Does a global scaling factor help?** Grouped-CV raw MAE 15.1692 ps → global-alpha MAE 1.1265 ps; zero-change reference is 1.0537 ps. Global α does not outperform this no-change reference.

**Q3. Does region-specific scaling further help?** Grouped-CV global-alpha MAE 1.1265 ps → region-alpha MAE 1.0832 ps; zero-change reference is 1.0537 ps. Region-specific scaling's nominal held-out MAE reduction over global scaling is 0.0432 ps, below the 0.10 ps observed timing resolution; there is no resolved region-specific benefit, and the zero-change reference is better. Region α also does not outperform the no-change reference.

**Q4. Is there a path-dependent effect for the same region/candidate?** Among 8 candidates with at least two path measurements, 0 had a within-candidate true-delta range greater than the 0.10 ps observed report resolution; maximum observed range was 0.00 ps. No path-dependent impact was visible at the observed report resolution; sub-resolution effects are unresolved.

No multi-candidate interaction or optimization experiment was run.
