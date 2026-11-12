# 已有科研工作整理

本页保存初始导入时的历史阶段结论，下文“当前证据”指该阶段，不替代后来新增框架的结果。完整实现、905输入验证和保留的限制见[全量结果](global_timing/FULL905_SEARCH_RESULTS.md)与[逐项验收](global_timing/REQUIREMENTS_AUDIT.md)；历史报告和原始日志未回写。

当前工作已经形成从候选提取、MFFC 分组、局部推理到顺序 replay、CEC、真实 mapping/timing 的证据链。后续工作还修正了 PI/PO timing 标签，并深入验证了一个 R9 案例。下表按工作依赖顺序整理已有结论，不将单案例、旧标签或局部指标扩大为通用预测及优化结果。

| 阶段 | 已完成结果 | 证据入口 |
|---|---|---|
| Gate1 | 多候选提取、指定 root/cut/candidate replay、兼容多 decision 安装；AC/AD/BC/BD 均有 CEC PASS | [Gate1C-R](../evidence/gate1c_r/gate1c_r_report.md) |
| Gate2A | 恢复预训练 Student；19维输入；mapping-free candidate bridge；局部 abc_delay 预测 | [推理审查](../evidence/litecut_recovery/gate2a_mapping_free_report.md) |
| Gate2B | Gate1 四组联合配置 global delay 都为263.60 ps，baseline263.32 ps；无可分辨 interaction/ranking reversal | [Gate2B](../evidence/gate2b/gate2b_report.md) |
| Gate2C | 固定规则选择50个 decision，109个候选 replay；24 timing-active、26 inactive；active 中方向9相同、15相反 | [Gate2C](../evidence/gate2c/gate2c_report.md) |
| Gate3A | c1355_orig：41 PI、32 PO、512 AND；62 strict maximal regions；全覆盖、无重叠、无遗漏 | [Gate3A](../evidence/gate3a/gate3a_report.md) |
| Gate3B | 62 regions、128条去重 region edges；32条保存路径；192个 region segments 均命中 root | [Gate3B](../evidence/gate3b/c1355_orig/gate3b_report.md) |
| Gate3C | 1815合法 decision、91 executable、326 candidates；全部位于单个 maximal MFFC；54/62 region 有机会 | [Gate3C](../evidence/gate3c/c1355_orig/gate3c_report.md) |
| Gate3D | 74 preselected；62 accepted、8 compromised、4 reverted；10个 region 接受多个 rewrite，最多3个；R2/R5/R7 对顺序敏感 | [Gate3D](../evidence/gate3d/c1355_orig/gate3d_report.md) |
| Gate3E | 326候选打分；74个有 GPU 选择的 decision 中52相同、22不同；R9/R24 的 GG/GL/LG/LL 全部接受并 CEC PASS | [Gate3E](../evidence/gate3e/c1355_orig/gate3e_report.md) |
| Gate3F | 四组真实 mapping/timing；跨 region 候选选择存在 interaction 和 R24 ranking reversal | [Gate3F](../evidence/gate3f/c1355_orig/gate3f_report.md) |
| Timing 标签更正 | 旧 `print_delay PO PI` 仅限制回溯路径，arrival仍来自所有PI；旧15/67候选 pair标签与相关指标不再作为正式pair结果 | [更正记录](../evidence/path_delay_model/LABEL_CORRECTION_NOTE.md) |
| Source-restricted timing | `print_pair_delay` 实现并验证；32个原路径端点对基线、16候选pilot、不可达fixture | [实现记录](../projects/culs/experiments/path_delay_model/SOURCE_RESTRICTED_TIMING_IMPLEMENTATION.md) |
| One-round oracle 边界审查 | baseline mapped网表缺少所需 cut/root端口，按定义停止，没有完成预测/传播实验 | [实现状态](../projects/culs/experiments/path_delay_model/ORACLE_ONE_ROUND_IMPLEMENTATION.md) |
| Mapper 内部观察 | 对同一函数点建立 CEC 支持的身份关系；观察 cuts、matches、phase和final cover | [mapper记录](../projects/culs/experiments/path_delay_model/MAPPER_INTERNAL_EXPERIMENT.md) |
| R9 机制分析 | 固定 root197/cut2/subgraph2，PI4→PO11 改善约0.650009 ps；直接机制为下游 NAND4 critical input 的 pin重排 | [单案例结论](../evidence/path_delay_model/R9_SINGLE_CASE_RESEARCH_CONCLUSION.md) |
| R9 局部 arrival | PI闭合 cone 内的原生 mode0 mapper计算通过 baseline sanity及候选对照；只验证一个region root的正相位all-PI arrival | [局部可行性](../evidence/path_delay_model/R9_LOCAL_ARRIVAL_FEASIBILITY.md) |

Gate3F 的原始数值是：

| 配置 | PO11 MIO arrival ps | Global Liberty stime ps | CEC |
|---|---:|---:|---|
| GG | 194.1 | 425.05 | PASS |
| GL | 194.3 | 389.02 | PASS |
| LG | 193.9 | 369.60 | PASS |
| LL | 193.6 | 388.94 | PASS |

两次 replay 的输出哈希相同，并与 Gate3E 相应输出一致。R24 从G换成L的global effect在R9=G时为−36.03 ps，在R9=L时为+19.34 ps；global交互项为+55.37 ps。PO11交互项为−0.5 ps。两种指标分别存在交互，但量值跨模型不能直接比较。

Gate2C 样本来自512条已捕获记录，而非全部1723条 engine records；50个 decision 中47对是 no-op/physical，3对是 physical/physical。因此24/50不能写成整个候选总体的比例，也不能作为LiteCut global accuracy。

更正后的16候选 source-restricted pilot 数据见 [metrics](../evidence/path_delay_model/source_restricted_pilot_metrics.json)：Pearson约0.0247、Spearman约0.2002、raw MAE约22.52 ps、符号一致率0.4375。样本小且按已有候选分位规则选择，仅反映该pilot。

R9 机制案例的 corrected MIO pair delay 为194.169998→193.519989 ps。关键链上的 cell类型和对应逻辑函数保持一致；一个 NAND4 的 critical input 从pin D换到pin C，arc由18.1800003降至17.5300007 ps，解释端点差值。该案例没有观察到逻辑关键路径切换，不能用于证明 Gate3F 四组 global stime 的关键路径身份。

R9 local-arrival 原型使用14个PI闭合的TFI，分别计算49/47个AND；仍构建完整subject graph以保留对象序、结构引用等上下文。正相位mode0 arrival为109.239990→106.219986 ps，两侧与保存的full-map参考一致，delta为−3.020004 ps，见 [validation](../evidence/path_delay_model/R9_LOCAL_ARRIVAL_VALIDATION.json)。它没有实现仅13/11节点窗口的精确增量更新，也没有验证final cover、下游pin决策、端点或global delay。当前证据没有建立通用跨MFFC联合优化器或梯度下降方法。
