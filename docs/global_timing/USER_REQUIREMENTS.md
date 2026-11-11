# Codex Goal：基于 MFFC、LiteCut 和多目标梯度下降的 AIG 全局时序优化系统

## 1. 总体目标（Primary Goal）

在现有项目代码基础上，设计、实现、测试并验证一个完整的 **Global Timing-Aware AIG Rewriting Optimization Framework**。

核心研究思想：

**MFFC 全覆盖分区 + Region DAG + Rewrite Candidate Enumeration + LiteCut Delay Prediction + Multi-Objective Gradient-Based Optimization + ABC Technology Mapping Validation。**

目标不是简单地对每个 Cut 单独选择最小预测 Delay 的候选，而是通过全局优化方法，在多个相互关联的 MFFC Region 之间联合选择 Rewrite Candidates，使整个 AIG 在 Technology Mapping 后的全局 Delay 尽可能降低，同时兼顾 Area 和其他约束。

所有算法必须最终通过真实 AIG Rewriting、ABC Mapping、Static Timing Analysis 和功能等价性检查验证。

**要求实现真正可以执行、复现实验、生成结果的系统，不允许只提供伪代码、数学描述或模拟实验数据。**

## 2. 现有代码检查与复用

开始开发之前，必须完整检查当前代码仓库、已有数据库和实验程序。

首先：

1. 遍历项目目录，识别现有源代码、脚本、配置、依赖、测试、数据集及实验结果。
2. 理解目前 MFFC 分区、Region DAG、Cut 枚举、LiteCut 预测、ABC Rewrite 和 Mapping 的实际实现。
3. 判断每个模块已经完成的功能、接口、缺陷和缺失部分。
4. 运行现有程序，建立原始 Baseline，记录运行环境、命令、输出和测试结果。
5. 尽可能复用现有代码，禁止未经分析就从零重写或破坏原有功能。
6. 建立模块依赖关系与执行流程文档。

如果项目中有数据库、数据库脚本或历史实验程序，也必须纳入检查和测试范围。

**重要：项目目前已经实现了 MFFC 全覆盖、Region DAG、Rewrite Candidate 提取、LiteCut 接入，以及部分跨 MFFC 组合的真实 Rewrite 和 ABC Mapping 验证。应首先检查这些成果，并在此基础上继续开发。**

## 3. 核心算法实现

### Phase 1：检查并完善 MFFC Region DAG

对整个 AIG 的内部 AND 节点进行互不重叠的 MFFC-based Region 覆盖。

确保：

- 所有需要覆盖的内部 AND 节点均被覆盖。
- 每个内部 AND 节点只属于一个 Region。
- Region DAG 的依赖关系准确。
- Region DAG 不存在环。
- PI 到 PO 的时序传播关系能够正确表达。
- 能够建立 AIG 节点、Region、Rewrite Candidate 之间的映射。

通过自动化测试验证覆盖完整性和拓扑正确性。

### Phase 2：Rewrite Candidate 管理

在每个 MFFC Region 内，提取实际可执行的 Rewrite Decisions，包括：

- Rewrite Root
- Feasible Cut
- Replacement Candidate
- 原始实现（No-Rewrite）
- Candidate 对应的结构信息与 LiteCut 预测信息

对不同 Region 的候选建立统一的管理结构。

尤其需要处理：

- Region 内多个 Rewrite Decisions 之间的冲突。
- 不同 Region Rewrite 操作的依赖关系。
- Candidate 替换引起的结构变化。
- 无法实际执行的 Candidate。
- Rewrite 执行顺序。
- 原始结构回退与结果复现。

禁止将结构上不兼容的 Candidate 直接作为合法组合执行。

### Phase 3：LiteCut Delay Prediction

复用现有 LiteCut 模型，预测每个 Rewrite Candidate 的局部时序指标。

要求：

1. 检查 LiteCut 模型实际输入和输出的语义。
2. 保证 Candidate 特征提取与训练时的特征定义一致。
3. 批量执行 Candidate Delay Prediction。
4. 缓存预测结果以减少重复推理。
5. 分析候选之间预测 Delay 的差异。
6. 区分局部 Cut Delay、Region Delay 和完整电路的 Global Delay。

**LiteCut 的 Cut-level Delay Prediction 不能直接当作整个电路的 Delay。**

必须研究并实现从 Cut-level Prediction 到 Region-level Timing Cost，再到 Global Timing Objective 的合理建模方式。

如果现有 LiteCut 输出不足以直接支持 Region Timing，需要设计明确的聚合模型或校准方法，并通过真实 ABC Mapping 数据验证其有效性。

### Phase 4：Global Multi-Objective Optimization

这是整个研究系统最关键、需要重点实现的部分。

定义每个 Region 的离散 Candidate Selection Variable：

x_i ∈ {0, 1, ..., K_i}

其中 x_i 表示第 i 个 Region 选择的合法 Rewrite Candidate。

优化目标包括：

**Objective 1：Global Delay Minimization**

利用 Region DAG 的依赖关系，构建从 PI 到 PO 的时序传播模型，估计 Global Critical Path Delay。

必须考虑不同 Region 之间的拓扑关系，而不能简单地将所有 Region 的预测 Delay 相加。

**Objective 2：Multi-Path Timing Optimization**

同时考虑多条重要 PI-to-PO Path，而不仅仅优化当前最长的一条路径。

研究和实现：

- Critical Path
- Near-Critical Paths
- Path Delay Aggregation
- Path Weighting
- Smooth-Max / Log-Sum-Exp 等可微近似
- 关键路径随 Candidate Selection 变化而转移的问题

**Objective 3：Area / ADP Constraint**

在优化全局 Delay 的同时，控制面积增长。

支持至少两种模式：

- Delay Minimization under Area Constraint
- Joint Delay-Area Optimization

以真正的 Technology-Mapped Delay 和 Area 作为最终评价指标。

### Phase 5：Gradient-Based Candidate Optimization

由于 Rewrite Candidate Selection 是离散变量，不能直接假设普通梯度下降可以优化 Candidate ID。

必须设计合理的可微松弛或梯度近似方法，例如：

- Softmax Relaxation
- Gumbel-Softmax
- Differentiable Candidate Selection
- Projected / Discretized Optimization
- Gradient-guided Candidate Search

要求：

1. 构建可微的 Global Timing Surrogate Objective。
2. 通过梯度反映不同 Region 对全局时序的影响。
3. 联合优化多个 Region 的 Candidate Selection。
4. 将连续优化结果转换为合法的离散 Candidate Combination。
5. 对离散组合执行真实 Rewrite。
6. 利用 ABC Mapping 得到真实 Global Delay 和 Area。
7. 将真实评价结果反馈到下一轮搜索或模型校准。
8. 记录优化过程、梯度、候选变化和目标函数的收敛情况。

必须保证优化算法的数学模型与真实 Rewrite 操作一致，避免出现预测最优但候选无法执行的情况。

### Phase 6：真实 ABC 验证闭环

实现以下完整流程：

AIG Input → MFFC Partition → Region DAG → Candidate Extraction → LiteCut Prediction → Global Optimization → Multi-Region Rewrite → ABC Mapping → Timing/Area Evaluation → Feedback。

每一次实际候选组合评估都必须：

- 使用真实的 AIG Rewrite 操作。
- 验证替换前后的 Boolean Equivalence。
- 使用统一的 ABC Mapping 和 Timing Evaluation Flow。
- 使用一致的 Technology Library、约束和工具设置。
- 记录实际 Delay、Area、ADP。
- 保留候选组合、执行命令和完整日志。
- 确保优化结果可以复现。

如果 Rewrite、Mapping、Timing 或等价性检查失败，必须显式报告错误，不能使用虚构数值替代。

## 4. 强制测试要求（Mandatory Testing）

**代码仓库内的所有程序都必须纳入测试范围。**

测试不仅针对新开发的优化模块，也包括现有代码和数据库中的所有可执行程序。

### 4.1 全仓库程序测试

扫描并建立程序清单，对所有可识别的模块、程序、可执行脚本、测试脚本和实验入口安排测试。

执行以下测试：

- Syntax / Import Tests
- Unit Tests
- Integration Tests
- Smoke Tests
- End-to-End Tests
- Regression Tests
- Functional Equivalence Tests
- Timing / Area Evaluation Tests

对于数据库相关代码，测试数据库连接、读取、写入和数据一致性，优先使用独立测试环境，禁止破坏原始数据。

如果某个程序因缺少外部环境、License、数据或依赖而无法测试，必须明确记录其路径、失败原因及所需条件，禁止直接标记 PASS。

### 4.2 关键模块测试

对每个核心功能进行独立测试：

1. MFFC 分区覆盖完整性。
2. Region DAG 拓扑与依赖关系。
3. Cut 枚举正确性。
4. Rewrite Candidate 功能等价性。
5. LiteCut 模型推理及接口。
6. Candidate Selection 合法性。
7. Global Timing Objective 计算。
8. Gradient Calculation 和数值稳定性。
9. Multi-Objective Optimization。
10. Multi-Region Rewrite 执行。
11. ABC Mapping 和 Timing Evaluation。
12. 结果保存、读取和实验复现。

同时增加异常输入、空候选、候选冲突、梯度异常、ABC 执行失败等测试。

### 4.3 Benchmark 实验

优先在现有已经使用的 c1355 Benchmark 上验证完整流程。

之后对仓库中所有可运行的 Benchmark 逐一测试。

至少比较：

- Original AIG / No-Rewrite
- Native ABC Rewrite
- Greedy LiteCut Candidate Selection
- Independent Region Optimization
- Random 或其他简单搜索 Baseline
- Proposed Global Gradient-Based Optimization

记录：

- Original Delay / Optimized Delay
- Delay Improvement (%)
- Area / Area Change (%)
- ADP
- Runtime
- Number of Evaluated Candidates
- Number of Applied Rewrites
- Functional Equivalence Result
- Predicted vs Actual Timing
- Optimization Convergence

所有结果必须来自真实程序执行，不允许伪造。

## 5. 工程质量和测试覆盖率

新增模块必须具有清晰 API、自动化测试和基本说明。

要求：

- 保持代码结构清晰，避免重复逻辑。
- 固定随机种子，记录实验配置。
- 统一日志和错误处理。
- 避免破坏现有工作流程。
- 自动生成测试报告。
- 尽可能建立持续集成测试流程。
- 对关键算法执行回归测试。
- 对已经发现的问题增加相应测试，防止重复出现。

每次修改关键算法后，必须重新执行所有相关测试及全局回归测试。

如果全量测试成本较高，可先运行快速测试，但最终验收必须执行完整测试矩阵，并明确报告未能完成的测试。

## 6. 必须交付的结果

最终提交以下成果：

1. **可执行的完整优化框架**：能够从原始 AIG 运行到最终 ABC Timing/Area 验证。
2. **Multi-Objective Gradient Optimization 实现**：具有真正的算法逻辑和可复现的搜索结果。
3. **自动化测试系统**：覆盖仓库内所有程序和核心模块。
4. **测试报告**：逐项展示 PASS、FAIL、SKIP、错误原因及测试覆盖情况。
5. **Benchmark 对比结果**：包含各 Baseline 与新方法的真实性能比较。
6. **优化过程分析**：包括梯度变化、Candidate 选择、收敛曲线及关键路径变化。
7. **使用说明**：描述依赖安装、数据准备、运行命令、参数配置和结果复现。
8. **问题与限制报告**：如实说明未解决的问题、预测模型误差、计算复杂度和优化失效案例。

## 7. 验收标准（Definition of Done）

只有满足以下条件，才可以宣告任务完成：

- MFFC 分区和 Region DAG 在测试电路上正确。
- Rewrite Candidates 均经过合法性检查。
- LiteCut 可以正常输出预测并接入优化算法。
- 实现真正可运行的 Global Gradient-Based Candidate Optimization。
- 支持联合优化多个 MFFC Region。
- 支持多路径时序目标以及 Area 约束。
- 生成的 Rewrite 组合通过功能等价性验证。
- 能够在 ABC 中获得真实的 Post-Mapping Delay 和 Area。
- 能够与多种 Baseline 进行公平对比。
- 仓库内所有程序均有测试记录。
- 所有未通过或未执行的测试均有明确说明。
- 实验结果可重复运行，并有完整日志和文档。

**研究方法不保证一定优于所有 Baseline；即使没有实现 Delay Improvement，也必须如实输出结果并分析原因，不得为了通过验收修改或伪造实验数据。**

## 8. Codex 执行原则

请以长期研究工程的标准完成任务。

优先复用现有模块，不要无必要地大规模重构。

先理解和测试现有代码，再逐步实现算法。

对于数学模型、候选可行性和时序传播机制，应优先保证正确性，而不是只让程序表面运行成功。

每完成一个开发阶段，就执行对应测试，确认通过后再继续。

遇到 Bug，定位根因、修复并增加回归测试。

不要仅停留在方案讨论、TODO、伪代码或没有验证的实现。

**最终目标是获得一套算法正确、工程可运行、功能可验证、性能可测量、实验可复现的 AIG 全局多目标时序优化研究系统。**
