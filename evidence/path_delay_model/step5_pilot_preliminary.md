# Step 5 preliminary single-candidate result

Run: Slurm job `48211`, `/data/wushangge/runs/CULS-path-delay-model-20260929-v3`. The exact input hash matches the previously used c1355 snapshot: `2602b10f131a6ba5996521036ba130aab128a4aa15e3ab49ce44f2f02047f0b0`.

The run reused the 32 saved Gate 3B paths, 62-region Gate 3A partition, 91 Gate 3C executable decisions, and 326 Gate 3E candidate scores. It did not enumerate new paths or change the partition. The deterministic rule selected the saved `PI node 4 → PO11` path (`R6 → R9 → R24 → R36 → R37 → R38`) and 15 distinct `(root, cut)` decisions along it. For each decision it replayed that decision's minimum-LiteCut-delta candidate from the same original AIG, independently.

All 15 requested decisions were preselected and finally `ACCEPTED`; all 15 replayed AIGs passed CEC. ABC's explicit `print_delay <PO> <PI>` query returned a fixed endpoint-pair delay for all 15. The saved AIG PI node 4 maps to ABC `pi03`; baseline `pi03 → po11` endpoint-pair delay was 194.4 ps. Exact mapped correspondence to the original AIG node sequence is not claimed. PO-wide maximum timing is recorded separately in the CSV.

Using the configured neutral epsilon of 0.01 ps:

- LiteCut predicted delta was negative for all 15 selected choices; the most-negative-per-decision selection makes this a biased sanity sample, not a random candidate sample.
- True endpoint-pair delta was neutral for 5/15, negative for 8/15, and positive for 2/15.
- On the 10 cases where both measured deltas are non-neutral, sign agreement was 8/10 (80%).
- Across all 15, prediction-vs-true MAE was 17.14 timing units and RMSE 19.92; Pearson correlation was 0.644 and Spearman correlation 0.363. These small-sample summaries do not establish predictive validity.
- The true endpoint-pair changes ranged from −6.3 ps to +6.9 ps, while predicted local deltas ranged from approximately −0.038 to −41.04. Several large predicted improvements produced no endpoint change; one candidate predicted a slight improvement but worsened the endpoint by 6.9 ps.

The largest methodological caveat is the current-side score: it is the Student prediction on the original AIG cone bounded by the same decision's cut leaves, not a measured mapped current supergate label. Also, candidates belong to Rewrite Decisions, not directly to MFFC Regions; several distinct decisions can occur in one region. The tested quantity is therefore a decision-level proxy delta, not a validated scalar contribution `Δd(Ri)`.

## Step 5 conclusion

The replay → CEC → exact PI/PO endpoint timing pipeline works end to end on 15 isolated choices. This pilot is not evidence that local LiteCut deltas numerically approximate path-delay changes: the magnitude match is poor, the sample is selection-conditioned, and the current score is a proxy. The additive two-/three-candidate hypothesis remains untested. As requested, no combination experiment has been started; pause here for review before proceeding.

Detailed rows are in `single_candidate_validation.csv`; all 32 path records and endpoint definitions are in `path_catalog.csv`; the 326 per-candidate model deltas are in `litecut_candidate_delta.csv`.
