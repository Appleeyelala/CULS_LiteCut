# GPU 联合梯度优化：当前进度

本任务仍在进行，未完成最终验收。目标保持为联合选择、单电路 GPU 并行、竞争性的真实 mapped delay 和低端到端 runtime；不以组件加速替代完整目标。

## 已实现与已验证

- 保留冻结旧算法控制；新增一次原 DAG adjoint 的局部二次提案器，保留相邻 MFFC 双线性项，批量优化 32 个轨迹，最终做有 edit 预算的全局冲突投影。
- 可关闭的八维成功反馈残差排序，标签只来自当轮之前的 PASS 测量；固定 LiteCut 仍为 CPU 冻结特征，不反传进 Student。
- 稀疏传回 argmax 编辑、复用 CPU 冲突索引、稀疏 hard 组合打分。Slurm 64931 的相关数值、梯度、反馈、合法性和 CPU/CUDA 测试为 27/27 PASS。
- 真实开发闭环两轮各 9/9 PASS；每方法 8 请求，包含失败/修复/基线。新 CPU/CUDA 的最终选择与 QoR 一致。
- 三个大图的同算法 float64 CPU/CUDA 组件对照完成；稀疏版与旧版本代理输入哈希、最终组合和顺序哈希都匹配。div/hyp/voter 的单次提案调用 CPU/CUDA 比值约 27.59/1.70/23.03，均只属于组件范围。
- 原生多配置 replay 的一次枚举复用已编译；no-op/四组跨 MFFC 配置的完整 trace/AIG 与默认和冻结旧工具一致，动态拒绝一致。完整开启复用回归正在继续，不先接入梯度闭环。

源码、数学与边界见 [IMPLEMENTATION.md](IMPLEMENTATION.md)；实际路径、失败记录和数值见 [RUN_LOG.md](RUN_LOG.md)；原生部分见 [PREPARED_REPLAY.md](PREPARED_REPLAY.md)。

## 实际 QoR 与失败证据

种子 20261111 的开发输入结果为：DMA 旧方法 2380.25 ps，基础近似 2345.21 ps，反馈版 2301.57 ps；反馈版最终为 **28 个 rewrites、25 个 MFFC 的显式修复**。ss_pcm 旧方法 323.91 ps，基础近似 332.45 ps，反馈版 328.03 ps，仍退化。c1355 三者均返回 360.11 ps 的 no-op。

反馈版 DMA 的四个直接梯度请求重复因 `(root=2901, cut_slot=5, subgraph_index=4)` 被 REVERTED，实际 saved=2、added=4；修复后才成功。证据是 `20261113T034453Z-joint-feedback-dev-3776f84d/output/feedback_cpu/case_0000/gradient_area_constraint/eval_000{3,4,5,6}/rewrite/raw/traces/choice.jsonl`。这提示安装的正 AIG 节点增益策略与 mapped-delay 目标需分别审查；尚未更改增益策略，也未验证允许该动作后的 CEC/area/timing。

以上不是稳定胜过随机/Native ABC 的证据，端到端 runtime 也未稳定下降。初步 wall 存在同节点并行诊断的影响，不能用单次最好时间作最终加速声明。

## 五项交付的完成边界

| 要求 | 当前证据与剩余内容 |
|---|---|
| 可复现改进实现 | 核心提案器、数学和测试已有；原生批量复用尚需完成回归并接入真实搜索 |
| 公平旧版及相关基线比较 | 有同分配旧 CPU 与新 CPU/CUDA 开发对照；最终 native/greedy/random/CULS 及统一冷/热缓存比较未完成 |
| 梯度和 GPU 独立贡献 | 同工作量 GPU 组件贡献已验证；梯度、LiteCut/交互、反馈/修复的受控消融未完成 |
| Delay–Runtime 实验 | 有三例真实最终点和失败；固定时间截点、time-to-quality、多种子面板及重复时间分布尚未完成 |
| 有效结论与限制 | 当前负结果和范围明确保留；完整最终报告仍需基于后续冻结比较 |

16 输入×3种子的最终面板已按元数据冻结在 [gpu_joint_panel.json](../../config/gpu_joint_panel.json)，尚未执行。不得按开发结果改面板或放宽已固定的主评价口径。

## 当前正在收尾的验证

Slurm 64935 在新目录复用精确源树匹配的编译产物，重验非法候选及全部默认/开启复用回归；64936 用真实 ss_pcm CPU/CUDA 闭环验证稀疏版与新请求完成时间轴。此前 64933/64934 失败目录保留，原因分别是测试夹具覆盖 unittest.run、错误地假设旧 dispatcher 必然非零退出，不能回写为 PASS。

后续依次以这些验证决定是否接入批量 replay，检查增益拒绝策略和重复失败成本，再实施独立消融与冻结面板比较；仍保留完整安装、CEC、真实面积约束和统一 Liberty timing。上述待办不是已经实现或获得优势的结论。
