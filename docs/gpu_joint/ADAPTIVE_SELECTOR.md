# 按计算规模保留全代理或采用局部批量提案

开发证据显示：ss_pcm 的旧全代理 STE 得到 323.908447 ps，而二次近似/残差排序为 328.026855 ps；小图 CPU 梯度成本仅约 0.1–0.2 s，CUDA 冷初始化得不偿失。大图上批量近似可避免大量重复 DAG 传播并获得明确组件收益。因此新增可关闭的 `adaptive_proposals`，不是继续扩大时序预测模型。

固定规则，不读取 benchmark 名称或真实 QoR：

- Decision≤128 且 Region≤4096：调用原 `optimization.gradient_proposals`，CPU float64、最多20步、沿用传入 restarts（主实验2）。使用原全代理 Softmax/STE/Adam，不使用八维残差；参数和采样初始化与原入口一致。
- 其余使用已有局部二次批量提案。`cpu/cuda` 强制同算法对照；`auto` 只在 `Decision×padded_width×trajectories≥500000` 且 CUDA 可用时选 CUDA，否则 CPU。默认32轨迹、40步，其他代理/实际验证规则不变。
- 小图分支即使请求 `cuda` 也明确记录 CPU；不能把这个分支的时间差称为 GPU 加速。profile 只在确实使用 CUDA 提案时读取 Python GPU 设备信息，避免为记元数据额外初始化 CUDA。

阈值来自当前开发规模和初始化/传播成本的工作判断，不是最优阈值证明；在最终冻结面板之前固定，无 circuit-name 特例。原 `feedback_*`、`criticality_*` 入口保留作受控消融，`adaptive_*` 独立记录实际 engine/device。正增益和可选非正增益安装策略另列，不混为提案器本身收益。

单元检查覆盖两个分支与原入口的完整 proposals 一致性及维度/设备规则；真实自由搜索、性能和面积/delay 结果尚需实际对照。仅用小图保留旧方法不能证明大图梯度更有效或稳定优于随机搜索。
