# 时序代理与实际执行契约

## 变量与可行域

每个 `(root, cut_slot)` decision 有 no-op 和 GPU mask 保留的 replacement candidates。一个 region 的变量由该 region 的各 decision 选择共同组成，允许同 region 内多个兼容 rewrite。不同 cut 共享 root、removable cone 重叠、任一 decision 删除另一 decision 的 cut leaf 都被保守排除。该策略可能排除某些特定顺序可执行的组合，不宣称枚举全部合法状态。

`MinimizeCutSupport` 会删去函数无关的cut变量。跨设计审查还发现cut叶子之间的逻辑关系可能使“边界变量相互独立”的完整真值检查过强。因此验证分两层：结构边界/小型BDD可直接证明的cut先通过；其余在保留原始PI和全部叶子逻辑的上下文中，以ABC CEC比较原root与导出LUT。不能把独立变量检查的差异当作实际反例，也不能跳过证明。未证明cut有独立清单，不进入优化；原始GPU导出始终保留。

所有组合从同一原始 AIG 重放，按原 root 升序执行。必须核对请求与 preselection 完全相等，且每个请求均为 ACCEPTED；否则记录 FAIL，不把部分安装当作请求组合。no-op 显式关闭原生默认选择。原始 AIG 与零选择序列化后 AIG 分开测量，避免把序列化引起的 mapper 差异隐藏在优化收益中。

## 局部到全局的代理

复用 LiteCut 原始 Student、19 维定义、标准化和逆变换。批量推理结果与历史逐候选计算逐项比对。缓存键含输入、模型、rewrite library、特征/模型代码和候选列表的 hash。原实现也提取局部结构特征，以构造相对 no-op 的变化。

对每个 region 的每个边界入口建立结构上的最长 AND 距离。代理传递成本为这个距离乘基线全局尺度，再乘一个由以下信息决定的有界指数校正：可从该入口到达的 decision 的局部 delay、weighted depth、节点数变化和激活状态，以及与直接前驱 region 的交互项。

沿 Region DAG 拓扑顺序用 log-sum-exp 传播所有边界入口的 arrival；没有把所有 region delay 相加。各 PO 的预测以同一 Liberty 基线到达时间锚定，加入 DAG 传播的相对变化。映射面积使用结构变化驱动的独立校准代理。该模型是待评估的近似，不能宣称重现 mapper 的 cover、pin、load/slew 或完整重映射行为；基线锚定精确也不代表预测准确。

## 梯度与真实反馈

候选 logits 经 softmax 松弛；前向用经过冲突投影的合法 one-hot 选择，反向使用 straight-through 近似。每次联合更新全部 decision 的 logits，并保存逐 decision 梯度、投影变化、预测端点和目标轨迹。有限差分测试针对可微代理；STE 不被声称为离散真实目标的精确梯度。

支持两个目标：平滑最大 PO arrival 加面积超限惩罚，以及归一化 delay/area 的 Smooth Tchebycheff。所有 PO 都有正权重；传播中的主导 region 分支和实际 critical PO 可随选择变化。最终面积约束在真实 mapped area 上重新检查，参考明确记录的零选择 replay baseline。

每批真实评估后用已观测的 Liberty PO arrival 和 area 拟合共享校正系数。每个新组合先保存预测，再读取它自己的实际评价；历史拟合误差与新组合误差分别记录。校准 warmup、失败请求、探索回退都计入评估预算，梯度提案来源单独统计，不能把随机回退称作梯度收益。

## 真实评价

统一流程为 `read_lib; read_aiger; map; topo; stime -p; write_verilog`。独立 ABC 构建仅加入只读报告，从同一个 `SC_Man` 导出 area、global delay、每个 PO 的 rise/fall arrival。解析器要求 PO 最大 arrival 与 global delay 一致。CEC 只接受明确的正证明并优先拒绝否定/未决结果。

真实CEC使用`cec -n`，显式按AIGER的PI/PO位置匹配，先检查接口数量。原输入可能包含符号名，而CULS replay未保存名称；名称不同不等于逻辑不同。PI/PO顺序仍是严格接口契约，不能任意重新排序。

原始 baseline、ABC 原生 rewrite、LiteCut 贪心、独立 region、随机搜索和联合搜索都使用同一输入、库、二进制和映射命令。随机与联合搜索有相同真实评估上限；确定性单次方法报告其较小调用数，不能只比较忽略校准成本的优化时间。主结果同时报告相对原始输入和相对 replay no-op 的变化。

## 最大arrival路径审计

`CULS_GLOBAL_TIMING_PATH_REPORT=1`启用独立ABC构建中的可选只读observer；默认关闭。它为arrival达到全局值95%的每个PO输出一条最大arrival链，包括该端点的rise/fall相位、前驱相位、cell/pin、实际load/slew下的arc delay和累计arrival。直接PI/常量端点也有明确终点类型。

原生`stime -p`按全局min-slack回溯，不能直接认作指定PO的最大arrival前驱。新observer按相同SC manager状态和原生Liberty lookup重新选择最大前驱，检查phase连通及逐arc累计误差；解析器拒绝缺失near-critical端点、非有限数值、循环或不能重构arrival的链。它不修改mapping状态。补充分析必须同时验证新CEC、全部PO/area/global数值及mapped网表正文与原记录相同。

PI/PO索引是固定接口的0-based编号；映射节点ID仍仅在本次网表内有效。输出反映ABC静态load/slew合并模型下的最大链，不是布尔路径可敏化证明，也不是signoff物理路径。比较源/端点集合变化时，不宣称跨映射节点身份已经对齐。

独立region基线仍以其他region均为no-op的root arrival作两轮坐标选择。为避免大region重复产生大量Python/Torch微操作，坐标候选的特征累加使用有序`numpy.add.accumulate`；保留标量exp/tanh/LSE计算和原候选平局规则。该辅助函数不参与梯度反传，联合优化的可微传播保持独立。

## 当前限制

代理使用固定原图 Region DAG，真实结构变化由重放验证；跨 region 交互只有直接前驱项，远程 mapper 决策和引脚重排可能预测错误。候选仍受原 GPU 的正节点增益筛选约束。多路径代理指所有 Region DAG 入口及 PO 的传播，不声称每条原 AIG 路径都有精确的映射后路径身份。native min-slack trace与可选最大arrival链分别保存，不跨运行按数值ID对齐。
