# Codex Goal：MFFC 级 GPU 并行改造与 905 电路评估

## 0. 本轮范围与优先级

本文件是本轮任务的最高优先级说明；与旧 Goal 中的实验范围、对照方法冲突时，以本文件为准。

**先修整并实现算法，再完成完整 905 个电路的实验。主实验只包含：本方法、基线 A、基线 B。**

不再以超过随机搜索为本轮验收门槛，不运行随机搜索对照、旧版全套对照、LiteCut greedy 对照或 delay–area joint 扩展。允许优化器使用必要的随机初始化，但它不是另一个随机搜索实验。旧代码只用于理解、回归和必要的数值核对。

研究目标：**以全局时序代理的梯度联合指导多个 MFFC 中的 rewrite candidate 选择，并使单个电路内部的计算更适合 GPU 并行，在较低端到端 Runtime 下获得有竞争力的真实 mapped critical delay。**

本轮是新版全量 905 电路正式评估。不得仅完成十几个或几十个电路就宣布本任务完成。

## 1. 开始前读取的材料

阅读 `GRADIENT_OPTIMIZATION_STATUS.md`、`FINAL_RESULTS.md`、现有源码、对应配置与原始日志。文件不在根目录时，在仓库中定位，不猜测不存在的路径。

已有报告只作为诊断起点：新版大图主要并行多条搜索轨迹，仍存在 CPU DAG adjoint；候选组合的完整安装失败和修复成本较高。先核对实际实现，不把报告中的假设当作已证实的根因。保留现有代码快照和实验结果，不覆盖历史证据。

## 2. 第一阶段：修整方案并完成必要实现

### 2.1 主线保持不变

原始 AIG → MFFC 分区与 Region DAG → 提取可执行 rewrite decisions/candidates → LiteCut 局部候选信息 → 全局可微代理目标 → 跨 MFFC 联合梯度更新 → 合法组合安装 → CEC → 统一 ABC mapping/timing。

一个 MFFC 可以包含多个 decision；优化变量是具体 decision 的候选选择，不强行简化成“每个 MFFC 只能选择一次 rewrite”。LiteCut 局部分数不是可直接相加的真实 MFFC delay。

### 2.2 本轮重点修整的内容

**A. 让 MFFC/候选结构成为并行计算的主要组织方式。**

- 以单个电路内部的 MFFC、decision、candidate 为任务组织依据，批量执行可并行的局部评价、特征聚合和选择参数更新。
- 在全局目标的前向与反向传播中保留 Region DAG 依赖；探索同层/就绪任务并行及合适的分组策略，不把所有 region 误认为独立。
- 一块 GPU 可通过不同线程组处理多个 MFFC；本轮不要求“一块物理 GPU 对应一个 MFFC”，也不强制多 GPU。
- 多轨迹并行可以作为辅助，但不能是唯一新增的并行机制；同时运行多个 benchmark 也不等于单电路 MFFC 并行。
- 关注大小不均、深且窄的 DAG、CPU–GPU 往返和同步开销。允许小任务回退 CPU，但必须记录实际执行设备及原因。

**B. 保留全局联合选择，而不是批量独立贪心。**

说明什么目标对什么选择参数求梯度，以及全局反馈如何同时影响多个 MFFC。优化器、离散表示、内核和数据结构由你决定，不强制更换 Adam/STE，也不要求引入新论文的梯度估计器。

**C. 把可执行性纳入算法，而不是只靠失败后修复。**

减少明显不可完整安装的提案，控制冲突、依赖和共享结构的影响。节点集合不重叠不代表任意 rewrite 可以并发提交。允许“并行评价、受控提交”的实现，不强制危险的并发改图。最终请求必须完整安装并通过 CEC；修复作为另一实际请求单独计时、计数，不能给部分安装的组合贴完整成功标签。

**D. 降低真正的总成本。**

减少不必要的重复准备、重复传输、无效迭代、失败修复和在线 ABC 调用。不要仅为制造更好 delay 而无限增加反馈测量，也不要只为占满 GPU 而增加轨迹和计算量。允许必要的轻量代理调整，但不将精确 local-to-global 预测或大型新模型训练变成主要工作。

### 2.3 第一阶段的完成要求

先提交简短的 `MFFC_GPU_REVISION_PLAN.md`，说明拟修整内容、CPU/GPU 职责、关键依赖及拟使用的公开代码；随后自主实现和验证，不停留在方案文档。

通过必要的功能、数值和等价性检查后，冻结源码、工具、配置、输入清单和实验协议，再进入 905 电路评估。小规模测试只能作为开发检查，不替代第二阶段。若存在真正阻断执行的环境/正确性问题，保留证据并明确说明，不编造测试完成。

## 3. 可参考的公开代码

以下链接已核实可访问。用于借鉴设计、数据组织和接口，不要求全部移植，也不把第三方既有功能计为本项目新增贡献。采用前核对许可证、依赖与实际可用源码，记录采用版本/commit；不要为查看参考代码而擅自升级已固定的 ABC 测量版本。

### 3.1 CULS：最直接的 GPU AIG 参考

- 仓库：https://github.com/cuhk-eda/CULS
- 算法目录：https://github.com/cuhk-eda/CULS/tree/main/src/algorithms
- MFFC 相关实现：https://github.com/cuhk-eda/CULS/blob/main/src/algorithms/refactor_mffc.cu
- Rewrite：https://github.com/cuhk-eda/CULS/blob/main/src/algorithms/rewrite.cu
- 并行哈希数据结构：https://github.com/cuhk-eda/CULS/tree/main/src/hash_table

参考 MFFC/cone 任务组织、CUDA 局部处理、共享结构与修改管理。不能直接把 cut-limited discovery 当成当前 strict maximal MFFC 分区。README 明确部分 GPU LUT mapping 等扩展只提供二进制，不要把未公开源码的模块当作可直接修改的实现。

### 3.2 INSTA：可微时序图传播参考

- 仓库：https://github.com/NVlabs/INSTA
- 正反向层次组织：https://github.com/NVlabs/INSTA/blob/main/src/graph/levelization.py
- 传播接口：https://github.com/NVlabs/INSTA/blob/main/src/timing/propagation.py
- 主接口：https://github.com/NVlabs/INSTA/blob/main/src/core/insta.py

参考依赖图组织、张量化传播和梯度提取接口。它面向已有时序图，不会直接预测 AIG rewrite 后的 mapping。项目采用 NVIDIA Source Code License，README 还列出预编译内核环境要求；不要假设所有 CUDA 内核源码都能直接获取或任意移植。

### 3.3 GCS-Timer：可选 GPU 图计算实现参考

- 仓库：https://github.com/cuhk-eda/GCS-Timer
- GPU 实现入口：https://github.com/cuhk-eda/GCS-Timer/blob/main/src/gpu_timer.hpp

README 提供 CPU/GPU 运行入口并声明 BSD 3-Clause License。只参考 GPU 任务组织和计算方式；不把 CCS 物理时序模型引入本轮主线，不用其 timing 数值替代 ABC 标签。

### 3.4 ABC：固定的真实测量与对照工具

- 仓库：https://github.com/berkeley-abc/abc

沿用项目中核对过的构建和版本，统一 rewrite/mapping/timing 命令。参考仓库不意味着需要重新移植或改变 mapper。

## 4. 第二阶段：完整 905 个电路，只比较三个方法

### 4.1 电路范围与冻结清单

使用已有 **905 个电路的完整冻结输入清单**，逐一运行所有电路，不抽样、不以小规模面板替代正式实验。使用现有清单记录各来源、AND/PI/PO、MFFC 数、Region DAG 深度/宽度和候选数量，供后续分组分析；不能按历史收益、运行速度或候选数量排除电路。

正式运行前冻结输入清单、输入 hash、优化 seed、工具及配置。无候选、失败、超时的输入也必须保留并报告，不能筛掉后补样本。主实验使用一个预先固定的优化 seed；小规模测试仅作开发/烟雾测试，不替代全量评估。需要重复计时时单列，不按结果选择最佳重复。

三个方法分别处理全部 905 个电路，**共 2,715 个方法运行点**（不含本方法优化过程内部的额外 ABC/CEC 调用）。每个方法点必须有可审计的终态；若因资源或环境阻断，保留未完成状态与原因，不能声称全量完成。

### 4.2 仅允许的三个主实验方法

| 方法 | 固定含义 |
|---|---|
| 基线 A | 原始 AIG → ABC `map` → `topo; stime -p` |
| 基线 B | 原始 AIG → ABC 原生 `rewrite` → 与 A 完全相同的 mapping/timing |
| 本方法 | 原始 AIG → 改进后的 MFFC 级 GPU 梯度联合优化 → 与 A 完全相同的 mapping/timing |

基线 B 默认单次原生 `rewrite`，不擅自替换成 `resyn2`、其他优化序列或按设计挑 recipe。正式运行前记录完整加载库/读取文件/优化/测量命令、ABC commit、编译配置与工具 hash；必要的输入格式准备一致并明确记录。

三者使用相同原始输入、ABC 二进制、ASAP7 Liberty、PI/PO 与 timing 条件。基线 A 直接使用原始 AIG，不能悄悄用序列化后的 replay no-op 代替。方法内部确需 no-op 校准时保留该测量并计入本方法成本，但它不是第四条竞争基线。

**不运行随机搜索、LiteCut greedy、独立 region、旧版方法、native CULS 或 joint 双目标作为本轮正式对照。**不额外扩展为九方法实验。CPU 数值参考只用于必要的实现正确性检查，不另增 905 电路 CPU 方法组。

### 4.3 优化与面积口径

本轮采用 delay 优先的单一主模式，不同时跑两种梯度目标。默认沿用现有面积约束 `Area <= 1.05 × replay no-op Area`，并在运行协议中明确；如调整，必须在评估前统一冻结。

ABC A/B 的原始测量照实报告，即使不满足该面积阈值也不删除或替换；将“原始 QoR”和“统一面积条件下的可行性”分开。报告本方法相对 A、B 的 area ratio，避免通过面积变化掩盖 delay 的取舍。

优化迭代、在线真实评价预算、停止条件与 timeout 在实验前固定。可以减少不必要的 mapping，但保留最终真实评价和功能等价检查。不允许把基线 B 的结果无成本注入本方法；任何实际调用的额外流程都必须计入时间并披露。

### 4.4 Runtime：必须清楚且完整

主表报告每个方法的**单电路冷启动端到端 wall time**：从该方法读取同一输入开始，到最终电路、必要正确性验证和真实 mapped timing 结果可用为止。

本方法时间包含其实际需要的初始化、MFFC/候选准备、LiteCut、传输、前反向、选择、replay、失败/修复、校准，以及所有在线 CEC/mapping/timing 调用。基线不承担不需要的候选、模型和梯度准备。各方法最终验证的计时口径保持明确、一致。

另记录优化部分、最终 mapping/timing、CEC 及主要 CPU/GPU 阶段时间，但这些分项不能替代端到端时间；有重叠的阶段不能相加充当总时间。GPU 计时需要正确同步，冷/热缓存分开标识，不把已有历史缓存结果当作零成本的新运行。

使用一致的主机、CPU线程限制和设备配置，核实优化编译；性能计时避免同 GPU/CPU 的竞争负载。整批 905 实验可以按资源调度，但批量吞吐/套件 elapsed 不等于单电路 latency。说明实际并发和硬件，不能跨不同 GPU 直接拼接成纯加速比。

本轮相对 A/B 的时间比称为**整体方法相对速度**，不是同算法 CPU/GPU 的纯硬件加速。若只做 A/B 对照，就不要声称已经独立证明 GPU 或梯度的因果贡献；可以报告实际设备、并行实现和阶段成本。

## 5. 必须交付的结果

### 5.1 逐电路主表

至少含以下字段，实际详细记录可用长表，不要求一张表放下全部字段：

- 电路 ID、输入 hash、来源、AND/PI/PO 数、MFFC 数、有效 decision/candidate 数。
- 三个方法的状态、真实 mapped critical delay（ps）、mapped area（标明库单位）、完整 runtime（s）。
- 本方法分别相对 A、B 的 delay 改善百分比、area ratio 和时间比。
- 请求数、实际 mapping/CEC 次数、完整安装成功/失败数、接受 rewrite 数、涉及 MFFC 数。
- 最终结果来源（直接梯度、修复、初始化、no-op 等）、CEC 结果、实际 CPU/GPU 路径。

计算口径：

`delay_improvement_A = 100 × (D_A - D_ours) / D_A`，B 同理；正数代表改善。

`runtime_ratio_A = T_A / T_ours`，B 同理；大于 1 代表本方法更快。

零分母/失败/超时标记不可用，不能填零 delay 或删除设计。胜平负容差在运行前确定。

### 5.2 总体与分组结论

分别对 A、B 给出：

1. 905 个输入中尝试、成功、失败、超时、无有效候选的数量及有效配对分母。
2. Delay 胜/平/负、逐电路改善中位数、配对 delay ratio 的几何均值。
3. 各方法端到端 runtime 中位数、P90、总记录时间，以及逐电路配对时间比；不要用两个时间中位数相除冒充配对加速。
4. 面积变化与约束满足情况。
5. 按来源和电路规模分组的 delay/runtime；补充 MFFC/DAG 形状对实际并行表现的描述。
6. 主要时间花在哪里，在线 mapping/CEC 及失败修复各占多少；额外给整批套件 elapsed，不与算法 runtime 混淆。
7. 在 A/B 共同有效且满足约定面积条件的配对上，统计“delay 更好且不更慢”“更快且 delay 不差”“更快但质量下降”“更慢但质量改善”等取舍，不事后改变门槛。

至少输出 Delay 相对 A/B 的图、Runtime 相对 A/B 的图和按规模的质量—时间对比图。不能只报少数成功实例。

### 5.3 文件清单

- `MFFC_GPU_REVISION_PLAN.md`：修整方案与公开代码参考。
- `BENCHMARK905_PROTOCOL.md` 与 `benchmark905_manifest.json`：冻结规则、方法、计时、面积/预算/容差、seed、硬件、工具身份和 905 输入。
- `BENCHMARK905_RESULTS.csv`：完整逐电路、逐方法结果；缺项保留状态。
- `BENCHMARK905_SUMMARY.json`：总体和分组统计，注明分母。
- `BENCHMARK905_RESULTS.md`：先用不超过一页讲清核心结论，再附完整结果与证据索引。
- 对应的配置、运行命令、原始日志、CEC、mapping 结果、计时和图表；沿用已有证据格式，避免为审计反复重跑整套实验。

## 6. 完成标准与执行边界

完成标准是：方案已修整并实现、必要正确性测试通过、冻结清单的 905 个电路中三个方法均被尝试并有终态、规定指标和结果表全部交付。效果不好也可以如实交付负结果，不能为了宣布成功改变规则。

出现资源问题时支持断点续跑，明确已完成/未完成范围，不把小样本当作本轮完成。无须等待“保证一定优于 ABC”才开始评估，也不预设一定能超过 ABC。

本轮只证明相对 A/B 的真实效果与整体成本；梯度及 GPU 的严格独立消融留待单独授权，不在本轮偷偷扩大对照范围。

**总主线：先让单电路内部跨 MFFC 的联合梯度算法更适合 GPU，再用 905 个电路对照原始 ABC mapping 和原生 ABC rewrite＋mapping，交付真实 Delay、Runtime、Area、可靠性及成本明细。**
