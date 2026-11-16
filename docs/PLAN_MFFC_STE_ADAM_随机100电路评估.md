# Codex 执行 Plan：恢复 Softmax + STE + Adam，多步代理优化，随机 100 电路 A/B/Ours 评估

## 0. 任务边界（必须遵守）

这是**实施计划**，不是开放式 Goal。按本 Plan 执行代码修改、少量必要检查及正式 100 电路实验，交付真实结果。**不得自行改变优化路线、增加对照组、扩大到 905 正式重跑或增加多轮在线 ABC Mapping 搜索。**

项目：`Appleeyelala/CULS_LiteCut`。目前依据的核心入口：`global_timing/mffc_experiment.py`、`mffc_search.py`、`mffc_model.py`、`surrogate.py`、`feasibility.py`、`optimization.py`、`scripts/run_mffc905.py`。历史 905 正式算法冻结源码 `ecc85e6ac82571caf9f1a6004098dbe976536256`，其原始结果和数据目录一律不覆盖；在**新分支/新实验 run 目录**实现。

**目标**：针对同一张 AIG，GPU 上通过可微的 MFFC Region DAG 时序代理，执行 **Softmax + STE + Adam 多步候选联合优化**；最后只提交一组离散 Candidate，CEC 后真实 ABC Mapping。比较 905 原集合中随机、无偏抽取的 100 个电路，方法仅 **A（原图）、B（单次 ABC rewrite）、Ours**，指标仅最终**真实 Mapping Delay 与完整端到端 Runtime**。

## 1. Ours 算法流程（最重要）

1. 输入原始 AIG。由 Ours 自己执行 **1 次真实 ABC Mapping**，记录原图 global Delay/所有 PO/Area 作为代理零选择锚点；这次成本计入 Ours 冷启动 Runtime。**不再额外做 serialized replay no-op 的 Mapping**。
2. 沿用现有 strict maximal MFFC 分区/Region DAG、CULS 候选枚举、必要的 Boolean 检查、LiteCut 冻结 Student、现有可微全局代理和 CUDA Region DAG 前/反传播。保持各个 `(root,cut_slot)` Decision 的 Candidate/no-op 离散定义。
3. **no-op 统一为零编辑、原始 AIG 本身**。`choice=0` 对应原图，代理在全零选择时应等于上一步的真实原图时序；如果最后没有选中任何改写，则直接返回已经测量的原图，不运行 replay-no-op，不再次 Mapping。修正旧版 original 与 serialized no-op 锚点语义混用的问题。
4. 构建各 Decision 的可训练 `logits`，冻结 LiteCut Student 和全局代理的 12 个参数；不在线校准代理，也不对真实 ABC 求导。
5. **必须实现 Softmax + Straight-Through Estimator + Adam 的连续多步选择优化**：
   - 有效候选掩码，`p = softmax(masked_logits / temperature)`；
   - 每一步将 logits 投影成不冲突、满足已有 `CandidateFeasibility` 约束且最多 `max_edits` 的 hard Candidate 联合组合；
   - `v = hard - p.detach() + p`，前向用合法 hard 组合，反向经 Softmax 梯度传给 logits；
   - 目标沿用现有 `scalar_objective`：全 PO 平滑时序、超过 1.05 倍基准面积的惩罚；
   - `loss.backward()` 后 **Adam 更新 logits**。不允许退回一次方向导数排序或用 Adam 只更新代理参数。
6. 训练过程每一步只通过可微代理算目标/梯度；保存各步已出现的合法 hard 组合及其**代理评分**，从中选预测目标最优者（含原图/no-op），不能仅取最后一步。**整个优化内循环禁止调用 ABC/CEC/真实 Mapping，也禁止将已测候选结果反馈校准代理**。
7. 若最优组合非空：从原始 AIG 执行 **1 次**受控 Replay，完整 ACCEPTED 后执行 CEC、同版本 ABC/ASAP7 `map; topo; stime -p`，得到唯一最终真实验证。未完整安装或 CEC FAIL 不能标成功；不再修复、重试或额外 Mapping，直接返回自己已付费测量的原图，失败保留日志。
8. 最终选择：若已验证改写满足 `Area <= 1.05 × 原图 Area` 且真实 Delay 比原图小，输出改写；否则输出已测原图。**不能无成本偷用方法 A/B 的结果做 Ours 决策。** Ours 正常情形最多 2 次 Mapping（原图锚点 + 1 次最终候选）；无有效候选或不产生修改时仅 1 次。

### 预先冻结的优化参数

- 单次搜索、**40 个 Adam steps、1 个 restart**；无额外随机 warmup、greedy、repair、反馈轮次。
- `Adam(lr=0.12)`，梯度范数裁剪为 5；温度由 `1.0` 线性退火到 `0.3`。
- 每个电路最多 **16 个非零 Decision edits**；保留候选 legality 规则与真实顺序安装。
- 初始 logits 以 no-op 为主（例如给 no-op +2 的初始偏置，其余加轻微固定种子噪声）；使用固定优化随机种子 `20261117`，不按历史/测试结果调整参数。
- float64 代理/选择，冻结 LiteCut Student，保留 GPU 上 Candidate→Decision→MFFC 聚合、DAG 前反传播；小图沿用现有 CPU fallback；记录真实 device。
- 保证大图性能：**不得每步把完整 D×W logits/所有 Candidate 复制到 CPU**。在 GPU 过滤、选每个 Decision 的优选候选，传少量稀疏编辑给现有可行性投影；hard 组合在 GPU 构造 STE 张量。避免 40 步重复做全量无效 JSON 落盘。

## 2. 实施路径（限定修改范围）

- 优先重用 `global_timing/optimization.py` 旧版 STE＋Adam 数学逻辑、`mffc_model.py` 现有 GPU DAG 代理、`feasibility.py` 候选可执行性投影；必要时新建 `global_timing/mffc_ste_search.py`。不得直接沿用旧 STE 实现中的全量 CPU `logits.tolist()` 热路径。
- 单独实现新的 Ours 流程入口（例如 `global_timing/mffc_ste_experiment.py`），不要直接继承当前 `MffcExperiment.__init__` 从而意外执行原图+replay-noop 两次 Mapping。保持旧 `mffc_experiment.py` / `mffc_search.py` 的旧实验可复现。
- 基线 A、B 沿用 `scripts/run_mffc905.py` 已有**相同 Evaluator、ABC、ASAP7**：A 为原图直接 Mapping；B 为 `strash; rewrite` 仅一次，然后 Mapping。新建或最小扩展 `scripts/run_ste100.py`，使用独立冷进程、A/B/Ours 同机资源和固定输入。历史 905 CSV 和缓存不能为 Ours 提供免费真实 timing。
- 仅保存最终选择、每步代理 loss 简明轨迹、最终真实测量、最终来源、总 runtime 及失败原因；**不需要额外制作大量防御性审计、完整 tensor dump、复杂诊断图或新的消融实验。**

## 3. 最小正确性检查

仅进行以下必要检查，然后正式跑 100 个：

1. 一个很小的代理/选择检查：确认 `logits.grad` 非零、Adam 在多步中确实更新选择参数；STE 前向是合法 hard，梯度沿 Softmax 回传；完整代理 `x=0` 锚定原图时序。
2. 三个已知输入的简短 smoke：`OpenABC/c1355_orig.aig`、`ACE/DMA_comb.aig`、一个无有效 Decision 的 `forgeEDA/mig_7series_v2_1_ecc_merge_enc.aig`。检查结果能生成、合法改写精确 ACCEPTED、最终 CEC 与 timing 成功、Mapping 次数符合 **<=2** 的要求。无候选只能做 1 次原图 Mapping。
3. 保留**最终候选的 CEC**，这是功能正确性的基本要求，不算可删除的防御性测试。除此之外不运行全仓测试/41 项冻结门槛/大量重复种子/多种消融。

如检查失败，直接修复影响正确性或运行的错误；不要把 smoke 结果计入正式 100 电路统计。

## 4. 固定随机 100 个电路

- 数据母集：原 `benchmark905_manifest.json` 中 **全部 905 个唯一电路**（原始 index 0–904），绝不根据旧版 Delay、Runtime、规模、是否有候选或测试成功率过滤。
- 在**任何正式方法运行前**固定采样：Python `random.Random(20261116).sample(range(905), 100)`，升序保存为 `STE100_SAMPLE_MANIFEST.json`，每条包含 index、相对 AIG 路径及 SHA256。输入和文件内容通过完整原 905 manifest 交叉核对。
- 全 100 个电路各自跑 A、B、Ours，共 **300 个方法终态**。B 如失败，保留失败和缺失 Delay，不伪造成功；无候选电路仍保留在分母中。
- 只跑**这一次预定抽样、一个优化种子、一组参数**。不要为了提升结果更改随机种子、过滤负例或重复试验挑最佳。

## 5. 公平计时与指标

- 同一 ABC 二进制、ASAP7 Liberty、`map; topo; stime -p` 及输出 global Delay（单位 ps）；三方法在相同机器/GPU/CPU 配额与独立冷进程下执行。沿用现有固定工具路径及已记录 hash。
- Runtime（秒）计入完整方法 cold E2E：进程启动、结构与候选准备、LiteCut/模型初始化、全部 STE+Adam 步骤、GPU 同步/传输、自己付费原图 Mapping、最终 Replay/CEC/Mapping、失败处理和必要输出；不含 Slurm 排队和一次性的源码构建、汇总分析。
- 新版算法自己的原图锚点 Mapping **不属于 Baseline A 共享的免费结果**。
- 不改变比较两种基线的算法定义；不额外加入随机/greedy/旧方向导数方案作为主要对照。
- 面积约束和 CEC 保留为**内部合法性条件**，正式用户结果展示以 Delay/Runtime 为主。

## 6. 必须完成的交付（真实实验结束后）

存入新的独立结果目录，例如 `runs/ste-adam-100-<run-id>/output/`：

1. `STE100_SAMPLE_MANIFEST.json`：固定随机 100 电路清单。
2. `STE100_RESULTS.csv`：**必须有 100 行**，列为 `circuit_id,input,A_delay_ps,A_runtime_s,B_delay_ps,B_runtime_s,Ours_delay_ps,Ours_runtime_s`。失败的 Delay 留空，并在单独状态文件中记录原因；不用 0 冒充。所有数值为**真实 Mapping 后 Delay 与完整 wall time**，不是代理预测。
3. `STE100_SUMMARY.md`：只保留简洁结果：三方法的成功数、Delay 中位数（说明不能把不同电路原始 Delay 中位数当配对性能收益）、Runtime 中位数；Ours/A、Ours/B 的配对 Delay 几何平均比和胜/平/负，配对 Runtime 比中位数。包含明确实际电路数与配对分母。
4. `STE100_STATUS.json`：300 个方法终态是否全部完成、失败/超时和 Ours 实际 Mapping 调用计数；保存代码 commit、工具与随机种子。只作轻量复现依据。

**成功标准是跑完随机 100 电路的真实三方法对照，客观报告 Delay 与 Runtime；不要求对所有电路胜出。** 原始数据不覆盖，不按结果调整配置。如果资源导致未完成，如实写已完成数，绝不可宣称 100 完成。

## 7. Codex 执行方式

使用普通 Codex 聊天/Plan 模式，不使用 Goal 模式。执行时严格按照此文件的固定算法、固定采样和输出格式，不自行改研究方向。允许必要的工程细节调整，但不得恢复中途真实 Mapping 反馈环；不得临时追加 benchmark/消融/训练 LiteCut。完成代码、最小 smoke、随机 100 电路 300 方法点后，直接给出结果文件路径与一页简明 Delay/Runtime 结论。
