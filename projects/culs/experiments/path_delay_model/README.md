# Path Delay Model pilot

This directory implements only Steps 1–5 of the requested study. It consumes the saved c1355 Gate 3A/3B/3C/3E data, never re-partitions the AIG, and stops after at most 15 isolated single-candidate replays. No two/three-candidate experiment, gradient, or LiteCut retraining is started.

## Existing findings

Read [EXISTING_PIPELINE.md](EXISTING_PIPELINE.md) before interpreting the pilot. In particular, the LiteCut target is a selected mapped supergate's local cone delay label, not a region-level timing contribution. Current-side score is defined as a model score of the original AIG cone bounded by the same decision cut leaves. It is a proxy, not measured mapped ground truth.

## Inputs and run

`config.json` freezes the input AIG, existing Gate artifacts, LiteCut checkpoint/library, Gate 2B ABC binary/ASAP7 library, 15-case cap, and deterministic path/candidate selection. The checked-in script `run_pilot.py` expects a staged run root under `/data/wushangge/runs/` and writes all intermediate files there. It performs:

1. Path catalog extraction from the 32 already saved Gate 3B paths and the existing Gate 3A roots.
2. LiteCut current cut-cone proxy inference and candidate delta table generation from the 326 saved candidate scores.
3. A baseline ABC map/timing sweep over the 32 saved PI/PO pairs using `print_delay <CO> <CI>`, plus a separate PO-wide maximum query. **Source-audit correction:** this command constrains the printed path's PI reachability but keeps all-CI forward arrivals; it does not supply a PI-restricted endpoint delay. Values extracted from its last `Arrival` are legacy trace-arrival values, not endpoint-pair ground truth. See `ABC_TIMING_AUDIT.md`.
4. Deterministic selection of up to 15 candidates on saved paths, one candidate per rewrite decision, ranked by executable-decision count, region count, then PO ID.
5. Independent replay from the original AIG for each candidate, exact status extraction, CEC, then frozen ABC mapping and the same PI-constrained trace extraction. The last part is **not** an exact endpoint-pair timing measurement per the source audit; do not use those labels for a new model-validation claim.

The job is submitted through Slurm from `/data`; do not run replay/mapping directly from `/home`. `run_path_delay_pilot.slurm` records the successful run configuration and includes a guard against reusing its existing output directory. For a rerun, stage a new unique run root and update the Slurm input/output paths first; inspect `squeue` and validate with `bash -n` before submission.

## Outputs

The run writes `results/path_catalog.csv`, `results/litecut_candidate_delta.csv`, `results/single_candidate_validation.csv`, replay AIGs/traces, per-case CEC and ABC logs, exact commands, `run_manifest.json`, and `step5_pilot_report.md`. The small CSV/Markdown/JSON summary is copied back to this repository's `results/` after the job completes. Mapping logs and generated AIGs remain under `/data`.

For the preliminary sign summary, `config.json` sets `neutral_epsilon_ps` to 0.01 ps. The trace's displayed arrival resolution is 0.1 ps, so that epsilon is below its textual resolution. In addition, source audit found the queried arrival is not PI-restricted. Accordingly the old pilot correlations/sign accuracy are only correlations against legacy trace-arrival labels; they are not evidence about true pair-specific timing. Its candidate sample is also selection-conditioned (the minimum predicted delta candidate per decision), so it is not a population estimate.

Do not rerun `run_calibration.py` as an endpoint-pair calibration until timing extraction is replaced with a validated source-restricted propagation method. Its existing 194.4-ps baseline guard checks the old trace-arrival extraction and does not validate PI4→PO11 pair timing.

No later phase should start until the pilot's feature construction, requested-vs-realized status, CEC, and timing extraction have been checked.

## Single-candidate calibration continuation

`run_calibration.py` reuses the fixed PO11 Gate 3B path and the existing LiteCut delta table. It selects **all** candidate rows whose regions occur on `R6 → R9 → R24 → R36 → R37 → R38` (67 candidates in the current artifacts), so the experiment covers negative, near-zero, and positive deltas. It reuses a prior pilot row only when its candidate identity, ACCEPTED status, CEC result, baseline, and replay-AIG SHA256 all match; otherwise it performs a fresh independent replay. Before any candidate replay it re-measures PI node 4 (`pi03`) → PO11 and stops unless the result is exactly 194.4 ps.

The first-stage Slurm recipe is `run_path_impact_calibration.slurm`. Its input tables and script are staged under a unique `/data/wushangge/runs/CULS-path-impact-calibration-20260929-v1/` run root. Resource request is 2 CPU, 8G RAM, 1 GPU. The outputs include the complete `results/single_candidate_calibration.csv`, global/per-region plots and metrics, and five-fold decision-grouped CV metrics. CV keeps all candidates for one `(region, root, cut)` decision in the same fold; unseen region/fold coefficients fall back to that fold's global alpha and are counted. CV also includes a no-change (`pred=0`) reference so shrinkage toward zero is not mistaken for useful calibration. Although an older Gate3F manifest records 0.01 ps precision, the frozen ABC `print_delay` Arrival text displays one decimal place; this experiment therefore uses 0.1 ps as observed timing resolution and keeps the requested 0.01 ps neutral epsilon separate.

Only after the single-candidate stage completes and its baseline/CEC/status outputs are checked should `run_path_impact_multipath.slurm` be submitted. It uses already-tested accepted candidate AIGs (no combined rewrites), samples four LiteCut-delta quantiles in each of R9 and R24, and measures each same candidate on existing Gate 3B PI/PO endpoint pairs that share its region. It first rechecks the catalog endpoint baselines. No two- or three-candidate interaction experiment is part of this continuation.
