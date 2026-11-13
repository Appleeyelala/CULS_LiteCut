# 单次 adjoint 与批量邻域提案：实现契约

本实现是在现有代理上的可关闭近似提案器，入口为 `global_timing/joint_optimization.py`。历史 Softmax/STE/Adam 入口保留；`Experiment.gradient_search(..., proposal_generator=None)` 的默认行为不变。当前文件描述代码，不替代真实测试或 QoR 证据。

## 一轮反馈内的计算

在当前真实 incumbent 的 one-hot 选择 \(v_0\) 上，使用原有 `TimingSurrogate` 的结构、校准参数、all-PO 目标与面积目标。令 \(c_e\) 是原代理的入口成本，\(g_e=\partial J/\partial c_e\)，\(b_e\) 是其指数成本的输入。一次原 DAG forward/backward 得到：

\[
q_e=g_e c_e(v_0)\,[1-\tanh^2(b_e(v_0)/2)].
\]

把原有线性入口特征和面积一阶项收集为 \(L_i\)，相邻 region 交互系数收集为 \(C_{rs}=\sum_{e:s\to r}q_e w_{int}\)。内层使用：

\[
Q(v)=J(v_0)+\sum_i L_i^T(e_i(v)-e_i(v_0))
+\sum_{(r,s)}C_{rs}^T[F_r(v)\odot F_s(v)-F_r(v_0)\odot F_s(v_0)].
\]

这里 \(e_i=\sum_k v_{ik}f_{ik}\)，\(F_r=\sum_{i\in r}e_i\)。`FrozenCriticalityObjective` 按构造在 anchor 处匹配原目标的值和一阶导数，并保留原有相邻区域的双线性项。**离开 anchor 后不是原代理的等价改写，也不是实际 timing 预测。** 平滑 PO 竞争、指数成本和面积项在该轮局部近似中冻结；后续真实成功反馈仍按原实现校准代理，下一轮重新计算 adjoint。

## 离散选择与 GPU 的范围

- 同一电路默认 32 个独立噪声初始化的 trajectories，float64；CPU 生成相同噪声，CPU/CUDA 执行同一批量目标和 Adam。默认 lr=0.15、eps=1e-12、40 步，温度 1→0.3，各 trajectory 梯度独立裁剪到范数 5。
- 内层使用 Softmax 连续松弛，**没有 STE 或逐步硬投影**。损失为 `Q + 0.05*trust + 0.1*conflict`。trust 是 soft Hamming edit 数超预算的归一化平方惩罚；conflict 沿用静态冲突对的激活概率乘积均值。
- trajectories 在 1/2/4/8/16/32（不超过实际配置与变量数）的 edit 预算间循环。最终一次传回 CPU，以相对 anchor 的 logit 优势贪心投影；保留所有未编辑的 incumbent 选择，任何激活/停用/候选替换都占一个 edit。静态合法不保证动态安装成功。
- 仅按局部二次目标排名 proposals；pipeline 在实际请求前仍用**完整原代理**记录 PO/area 预测。真实安装、失败、修复、CEC、mapping 与请求预算规则保持一致。返回的 best 仍只由真实成功测量决定。
- 最终 argmax 和相对 incumbent 的 logit 优势在选定设备上批量计算，只传回有正优势的编辑；CPU 复用一份冲突索引。合法 hard 组合用其非 no-op 项直接打分，避免为每个组合再构建完整 one-hot。数学目标与投影规则保持一致，需要与此前大图保存的选择哈希再次核对；新增分阶段计时区分 adjoint、初始化/传输、内层、稀疏传回、合法投影和最终排名。
- 新 trace 使用 `joint_criticality_v1`，保存原始目标在 anchor 的梯度范数和含 penalty 的迭代梯度，不能把后者全部解释为时序信号。它不同于旧逐 restart/step 的 trace schema。
- CUDA 只并行内层多轨迹/多候选计算；初始 DAG adjoint、代理校准、最终投影仍在 CPU。传输、CUDA 初始化和这些 CPU 阶段都须计入冷进程时间。多 benchmark 并发不作为单电路 GPU 加速。

## 验证与对照入口

`tests/test_joint_optimization.py` 检查 anchor 值/梯度、非可加交互、批量与单条等价、编辑预算/冲突、可复现性及 CPU/CUDA 一致性；fixture 数字不能用作实验收益。

`scripts/profile_single_method.py --variants legacy_cpu,criticality_cpu,criticality_cuda --legacy-source <冻结旧src>` 在一个分配中顺序运行三种控制，每个案例轮换控制顺序，独立冷子进程。旧方法从明确的旧快照导入；保存实际导入路径和源码哈希。新 CPU/GPU 的算法、精度、噪声、轨迹/步数和预算一致。源码/模型准备成本、实际 EDA 调用和不同参数均需分列；首轮单次计时不能证明稳定加速。

已知限制：局部近似可能忽略路径竞争变化；冻结的梯度可能为零；相邻交互为零时目标可退化；贪心投影可能丢弃有用的组合；最终 proposals 可能重复，触发原 pipeline 的随机探索。没有通过局部测试就宣称梯度的因果收益、QoR 改善或端到端加速。

## 可关闭的成功反馈残差排序

首轮真实开发对照中，基础近似在 DMA 有更低 delay 和更少失败，但更慢；ss_pcm 质量退化。该负结果保留在运行记录中。为检验局部敏感度遗漏实际重映射影响的问题，`JointProposalConfig(feedback=True)` 增加八维标量残差，不另建精确时序模型：

\[
\phi(C)=\left[\sum_i\frac{n_{i0}}{N_{AND}}e_i(C),\quad
\frac{1}{|E_a|}\sum_{(r,s)\in E_a}\frac{F_r(C)\odot F_s(C)}{D_rD_s}\right].
\]

\(E_a\) 是两端都有 Decision 的原始 region 邻接，\(D_r\) 是该 region 的 Decision 数。没有此类边时后四维为零。对此前 PASS 的每次真实测量，用相同标量目标计算 \(J_j^{ABC}\)，定义：

\[
X_j=\phi(C_j)-\phi(C_0),\quad
y_j=J_j^{ABC}-[Q(C_j)-Q(C_0)+J_0^{ABC}].
\]

每列按当前样本 RMS（下限 \(10^{-4}\)）缩放成 \(Z\)，解 \(\beta=(Z^TZ+I)^{-1}Z^Ty\)，再换回原特征尺度。内层目标为 \(Q_f(C)=Q(C)-Q(C_0)+J_0^{ABC}+X(C)^T w\)。系数吸收入同一个线性/双线性目标，GPU 内层不增加 DAG 传播。failed、部分安装及后续尚未发生的测量均不进入拟合；训练样本索引和残差 MSE 明确记录。anchor 与真实标量目标相同按构造成立，不算预测准确性。

该功能的标签是当前成功真实评估，作用是提供低成本的在线排序信号。训练残差下降、非零系数或梯度增大不能代替真实 QoR、泛化或独立贡献检验。`feedback_cpu/feedback_cuda` 为其对照入口；`criticality_cpu/criticality_cuda` 保留不使用残差的基础版本。`profile_joint_kernel.py` 另用保存的大图代理输入比较相同 CPU/CUDA 计算，不运行 EDA，也不把局部计时当端到端加速。
