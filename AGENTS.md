# CULS LiteCut workspace rules

This repository consolidates existing AIG rewriting, MFFC, LiteCut and ABC timing research. The long-term objective is a paper. The current completed task is historical organization; a new research direction or experiment has not been selected.

- Read `README.md`, `docs/WORK_HISTORY.md` and `docs/KNOWN_LIMITATIONS.md` before interpreting results. Use `docs/PROJECT_MAP.md` for source locations.
- Preserve the original repositories and historical run directories. Develop in this repository. Do not rewrite imported evidence to make old conclusions look current.
- Keep MFFC Region, Rewrite Decision, cut-bounded Rewrite Region, Replacement Candidate and local LiteCut Score distinct. Original AIG IDs, GPU IDs, serialized replay IDs, mapper IDs and mapped-netlist IDs are different namespaces.
- Gate3F PO arrival uses fixed MIO timing; its global delay uses Liberty `stime`. Corrected PI/PO pair timing uses `print_pair_delay`. Initial mapper mode0 arrival is a fourth metric. Do not combine them as interchangeable labels.
- Keep baseline evidence in `evidence/` and provenance in `provenance/`. These are small, reviewed historical copies. Full logs, AIGs, checkpoints, libraries, binaries, builds and new results belong under `/data/wushangge/CULS_LiteCut/`.
- Follow the installed `slurm-data-workflow` skill for substantial computation. Compile, train, benchmark and run experiments only in a `/data` workspace through Slurm. Do not execute imported experiment scripts directly from `/home`.
- Before a Slurm submission inspect current cluster configuration and the user's RUNNING and PENDING requests. Total concurrent allocation must stay within 64 CPUs, 256 GiB RAM and 2 GPUs. Do not assume a partition or cancel existing jobs automatically.
- Use `scripts/prepare_run.py` to prepare an immutable committed-source snapshot when an experiment is authorized. It creates directories and manifests only. Record task purpose, source commit, input/model/library/binary hashes, command, environment, resources, Slurm ID and outputs in each run.
- Keep `main` reviewable. Use focused branches and commits for later changes; tag confirmed milestones. Do not amend imported baseline commits, force-push, publish or configure an external remote without the user's direction.
- The initial import intentionally retains historical hard-coded paths and command semantics. Path migration is a separate explicit change; never silently redirect a historical run's outputs.
- Do not spawn sub-agents unless the user explicitly requests delegation.
