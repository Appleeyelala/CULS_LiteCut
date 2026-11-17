# CULS LiteCut workspace rules

## 核心研究目标

本项目研究基于 MFFC 结构化覆盖、LiteCut 局部时序预测和全局可微时序代理的 GPU 加速 AIG Rewrite 优化方法。论文的两个核心创新方向是：

- **全局梯度联合优化（最高优先级）**：构建准确、可微的全局时序代理，利用梯度下降在多个 MFFC 中联合选择 Rewrite Candidate，使 Technology Mapping 后的真实全局 Critical Delay 尽可能降低。
- **GPU 并行加速（同样是核心创新点）**：利用 GPU 加速候选处理、全局时序传播及梯度优化等主要计算阶段，显著降低完整优化 Runtime，使方法在实际电路上具有竞争力。

研究优先级：**梯度联合优化与真实 Delay 改善 > GPU 加速与端到端 Runtime > 其他辅助问题**。LiteCut 是现有的局部预测基础，研究重点是全局时序建模、跨 MFFC 联合优化和 GPU 高效实现。

最终评价以真实 ABC Technology Mapping 的 Critical Delay、完整 Runtime、功能等价性及面积约束为准，而不是仅看代理 Loss、局部 Delay 或候选数量。

具体算法、模型设计、梯度策略、GPU 实现方式和超参数均可根据实验结果探索和改进；当前 STE＋Adam 的 40 步、16 个编辑等配置不是固定原则。所有研究和工程改进都应服务于上述两个核心创新方向。

## 工程管理与实验规范

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
