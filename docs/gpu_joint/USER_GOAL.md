# Goal：面向低 Runtime 和高 Mapping QoR 的 GPU 加速梯度联合优化

## 研究背景

当前项目基于 MFFC 分区、LiteCut 和梯度优化，实现跨 MFFC 的 AIG Rewrite Candidate 联合选择，目标是改善 Technology Mapping 后的全局 Critical Delay。

现有代码已经完成梯度优化闭环，并在 905 个电路上进行了真实实验，但目前仍存在梯度指导效果不稳定、候选执行失败率较高、GPU 加速不充分等问题。

请首先阅读：

- `docs/research/CODEX_GOAL_梯度联合优化与GPU加速.md`
- `docs/research/评价体系与备选算法_文献记录.md`
- `GRADIENT_OPTIMIZATION_STATUS.md`

结合真实代码、实验日志和已有结果开展工作。

## 核心目标

**在保证 Mapping 后 Delay 具有竞争力的前提下，通过更有效的梯度联合优化和 GPU 并行计算，显著降低整体优化 Runtime，实现更好的 Delay–Runtime Trade-off。**

论文重点研究两个方向：

**1. Gradient-Based Joint Rewrite Optimization**

提高跨 MFFC 离散 Candidate 联合选择的优化效率，使梯度能够更有效地指导实际 Rewrite 决策。

不要求精确预测整个 Mapping 后的时序，但优化信号应当能够有效指导真实 Critical Delay 的改善。

**2. GPU-Accelerated Parallel Optimization**

充分利用 MFFC 的结构化分区和 GPU 并行能力，加速候选评价、梯度计算及其他适合并行的优化阶段。

重点关注单个电路内部的并行能力以及端到端 Runtime，而不仅是某个局部 GPU Kernel 的速度。

## 成功标准

最终方法应尽可能实现：

- **QoR：** Mapping 后的真实 Critical Delay 具有竞争力，不要求每个电路都优于 Native ABC。
- **Runtime：** 相比现有实现显著降低端到端运行时间。
- **Gradient Effectiveness：** 在公平的计算时间或评价预算下，梯度优化相对简单搜索方法体现实际价值。
- **GPU Speedup：** 能够通过合理的 CPU/GPU 对照证明 GPU 并行的独立贡献。
- **Reliability：** 候选组合合法、Rewrite 完整执行、CEC 通过，实验可复现。

最终以真实 ABC Mapping/Timing 结果和端到端 Runtime 评价，不以代理 Loss 下降或 GPU 利用率提升代替实际收益。

## 自主优化要求

请自主分析当前实现与实验结果，识别最主要的性能瓶颈，并据此决定优化方向、实现方案和实验安排。

不预先指定必须使用哪种梯度算法、代理结构、GPU 并行策略或具体参数。

允许在现有框架基础上进行必要的算法和工程改进，但应优先优化已有方法，不要为了使用新论文算法而增加不必要的复杂度。

新梯度算法暂时作为备选方案。只有当现有方法存在明确瓶颈、且替代方案有合理依据时，才考虑引入。

**不要把精确的 Local-to-Global Timing Prediction 重新变成主要研究任务。**相关模型的调整应服务于梯度优化的实际效果。

## 实验与研究要求

- 保持公平、统一、可复现的评价体系。
- 区分算法收益、GPU 加速收益和真实 Mapping QoR 改善。
- 关注优化质量与时间之间的权衡，而不是单独追求最低 Delay。
- 保留原有代码版本和实验结果，不覆盖历史证据。
- 根据实际需要开展测试，避免没有明确目的的大规模实验。
- 不预设结果一定改善；如果效果不好，应分析原因并调整方案。
- 区分已经实现、已经验证和仍未证明的研究结论。

## 最终交付

完成必要的代码改进与实验验证，并提供：

1. 改进后的可复现实现。
2. 与原始版本及相关基线的公平对比。
3. 梯度优化效果和 GPU 加速效果的独立分析。
4. Delay–Runtime Trade-off 的实验结果。
5. 当前方法的有效结论、局限与后续可研究问题。

## 执行原则

这是一个以研究目标为导向的开放式优化任务。

**请围绕核心目标自主推进，不必严格按照预先制定的技术路线实施。**

优先依据真实实验发现问题、改进方法、验证效果；当某条路线无效时，可以调整方案，但始终保持研究主线为：

**跨 MFFC 梯度联合优化 + GPU 并行加速 + 有竞争力的 Mapping Delay + 低端到端 Runtime。**