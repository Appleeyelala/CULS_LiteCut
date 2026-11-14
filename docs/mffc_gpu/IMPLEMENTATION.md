# MFFC GPU 修订实现与边界

本版实现完整全局代理的GPU梯度与稀疏联合改动。源码入口是 `global_timing/mffc_experiment.py`，运行入口是 `scripts/run_mffc905.py`。真实工具、Student权重、原始/历史结果保留不改；完整905×3正式运行及审计已完成，见[最终结果](../../BENCHMARK905_RESULTS.md)。

## 变量、目标与GPU任务

每个Decision `(root,cut_slot)`选择一个library candidate或no-op；一个MFFC允许多个兼容Decision。沿用原12参数代理的4维相对特征、port incidence、结构深度、相邻Region特征乘积、全PO平滑传播及no-op基线锚定。目标仍为归一化全PO LSE加面积超过1.05倍的平方惩罚；不直接相加LiteCut分数，不求真实mapper梯度。

`mffc_model.MffcGpuSurrogate`把Candidate→Decision→Region的期望特征、稀疏port聚合、相邻耦合和面积计算放在CUDA。`framework/mffc_gpu/region_lse.cu`中一个warp取一个就绪MFFC，多个warp并行处理不同MFFC；CSR保留原始port身份与重复依赖。前向按非PI前驱计数发布就绪项，反向按后继贡献完成计数；设备原子队列和逆向adjoint累加保留整个DAG，深链无需Python逐层发射。大fanin在warp内归约。CUDA11.8的double累加用CAS；warp之间用acquire/release原子及warp同步建立依赖。

输入AIG解析、strict分区、候选枚举准备、Boolean证明、LiteCut特征提取/冻结Student推理、候选复用约束和最终合法投影在CPU；CULS自身的GPU枚举/rewrite分开归属。新原生传播接受Torch当前CUDA stream上的float64缓冲，使用sm80/86/89与PTX构建的独立共享库，运行时不JIT编译源码。设备队列完成数与错误码在真实提交前同步核对，事件计时仅覆盖内核，不代替全方法wall。

## 全局梯度如何产生联合改动

在当前已经真实测量、面积可行的incumbent one-hot \(v_0\) 上，固定代理参数，计算：

\[
g_{ik}=\frac{\partial J(v)}{\partial v_{ik}}\bigg|_{v=v_0},\qquad
\mathrm{gain}_{ik}=g_{i,x_i^0}-g_{ik}.
\]

所有PO、下游MFFC、面积和相邻耦合同时参与这次反传。每Decision选正方向导数收益最大的允许选项；GPU排序，只传最多 `8 × maximum_edits` 个编辑供CPU受控投影。使用1/2/4/8/16编辑预算构造合法联合块，**再次用完整DAG代理前向评分**，每轮最多提交两组。不是各MFFC独立局部贪心，也不使用冻结二次Q或额外随机搜索。

最多2轮、每轮16 edits；总replay预算4含no-op/失败/repair，original为额外付费真实测量。存在新PASS反馈时用12步Adam拟合已有12参数；候选本身使用上述离散方向更新，无内层Adam/STE/多轨迹。梯度不经过Student、离散投影、实际rewrite、CEC或ABC。小任务D<32且R<1024回退同方程CPU参考，保存实际原因；无Decision时不实例化Student/代理。

## 提交前候选约束和真实安装

`feasibility.CandidateFeasibility`按library phase/permutation解码候选，在原AIG结构哈希中检查实际引用的原AND节点。候选复用某原节点时，禁止与删除该节点的其他Decision同时选择；重复原哈希保守包含所有匹配身份。原root/其传递后继被复用的选项先排除。输入叶共享本身不禁止；既有同root、removable overlap和cut leaf deletion约束继续生效。

这个预检没有完整模拟动态引用计数、重写间新结构哈希、顺序实际增益或所有别名环。真实安装仍从原图开始按原root顺序，精确预选并全部ACCEPTED后再CEC/统一Liberty。被拒绝项清零的repair是新组合/新预算，不把失败的部分安装贴成功标签。GPU并行评价与CPU受控提交不意味着并发修改所有MFFC。

original与serialized no-op分别测量、计入Ours成本；只把no-op作为zero-choice校准锚点。最终比较所有真实面积可行成功观察；自己已付费测的original若更好可作为`original_fallback`。不读取外部A/B的结果替Ours选best。

## 当前验证和未证明项

- 65237完成7项原生队列/前反向测试；65238/65239完成32项完整模型、逐候选/参数梯度、校准、相位置换/复用约束和旧默认回归，无SKIP。65236的首次CUDA编译错误原样保留。
- 65239/65240共6个开发输入×3方法=18点全部PASS；65241重新只读审计其38份实际测量与20个成功replay观察，错误0，并通过4项统计分母/缺失/零值/配对测试。没有重新跑EDA。
- c1355/DMA/div/hyp均有完整GPU DAG前反传播；hyp有117509个MFFC、16269层、最大同层12991个，验证设备队列实际处理此类形状。无候选mig实例保留，没有强造梯度贡献。
- 开发数据包含正负结果：c1355改善、hyp返回原图、ss_pcm劣于原生B，整体比baseline耗时更多。它们只用于实现检查，不作为905性能结论。

正式905/2715终态和5750份真实测量已审计：A/Ours各905 PASS，B为901 PASS/4 FAIL；Ours含806 CUDA、5 CPU、94无候选。对A/B的Delay比几何均值0.944392/0.998084，配对baseline/Ours时间比中位0.135011/0.166129，整体Delay–Runtime目标未达到。完整安装3039 PASS/2 FAIL，两个失败是BRAM_SSP同次gpuls batch设置CUDA heap时报OOM，最终no-op成功；不能把final全PASS当所有请求都成功。

正式806个CUDA点原生事件合计逐点中位70.09ms，仅覆盖原生前/反向，不含完整Torch/CPU/EDA成本。本轮未独立验证同算法CPU/GPU因果加速或梯度消融，没有声称精确局部到全局预测、AGD严格复现、所有候选组合可执行、布线后signoff或未见集泛化。
