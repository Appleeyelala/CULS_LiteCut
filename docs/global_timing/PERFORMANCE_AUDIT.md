# 保存输入的梯度性能检查

本记录是工程性能诊断，不是新的时序改善结果。完整源码、输入hash、环境、cProfile与参数在`/data/wushangge/CULS_LiteCut/runs/20261111T142532Z-saved-gradient-profile-5baf3f8e`（64361）中。

## 被复核的实际搜索

输入来自`20261111T134714Z-final-feature-e2e-retry-aa36f9d5/output/suite/case_0020`，即`ACE/div.aig`。它保留51,140个verified decisions、28,306个region、81,276条边界传递边和4,183个DAG层。该搜索在1800 s单例限制下没有完成第二种目标，记录仍为FAIL，不由本诊断改写。

诊断重构面积约束模式的`calibration_0002`，仅用其已记录的更早成功观察（no-op与一次显式修复），30步校准、两次restart各20步梯度提案。校准损失、全部参数和完整梯度trace均与保存结果精确相同。没有调用rewrite、mapping或产生新的timing标签。

| 阶段 | 含cProfile的墙钟秒数 | 直接观察 |
|---|---:|---|
| 分区 | 1.12 | 重建同一原始AIG的partition |
| 模型初始化 | 7.99 | 逐层重新扫描所有region约5.09 s |
| 校准 | 104.46 | 30次backward合计75.67 s；传播forward约26.52 s |
| 梯度提案 | 127.31 | 40次backward合计55.94 s；传播forward约28.53 s |

提案阶段逐decision读取logits的列表构造约14.82 s，合法投影约14.34 s，one-hot构造约10.03 s。这些为同一调用栈的累计时间，不能把父/子项重复相加。125,490/167,320次`index_copy`分别对应校准/提案中的逐层传播；完整反向图是主要成本。

共享分配中原E2E的wall time包含资源竞争；此处又有cProfile开销。两者不用于估算算法独占运行加速，也不把局部函数成本外推为905设计的整体收益。

## 已提交的有限修复与验证边界

`0b03f67`按原拓扑顺序一次建立layer分组，使用scatter构造相同one-hot，整体转换logits再逐行截断padding。没有调整模型、平滑参数、目标权重、候选集合或投影规则。该修改不消除深DAG反向传播本身的开销；验证要求保存的loss、参数、完整梯度trace一致，并补真实回归。

64363（`20261111T143301Z-gradient-overhead-regression-62411fb5`）已完成：46项CPU矩阵为36 PASS、10个真实工具入口SKIP；div的loss、参数与完整40步梯度trace精确一致。初始化2.60 s、校准99.05 s、提案105.82 s，均含profiler。这里只报告相同保存输入上的两次测量，不宣称跨设计或完整E2E的加速；深DAG求导仍是主要成本，真实E2E须单独完成。

64341.4随后通过46项真实核心回归，并开始div/hyp/c1355完整比较。仍使用每搜索8次实际评价预算；根据已有耗时证据给予3600 s单例限制，保留此前1800 s失败。此处增加时间上限并不构成算法性能改善证据，完整结果仍须检查对应case记录。

64361初版把可变`run.json`的开始状态hash放入`input_sha256`，该项不能验证终态manifest。其他输入hash及数值一致性检查仍有效。`27e3741`在后续剖析中排除可变manifest，旧报告保留。准备后未提交的`20261111T143226Z-gradient-overhead-regression-f0d5fdd7`明确标为`SUPERSEDED_NOT_SUBMITTED`，不是失败或完成的实验。

## 有界传播求导存储

`33d067b`增加`global_timing/arrival.py`，保留原Torch传播为`reference`参照，默认使用`bounded_tape`。前向沿原拓扑层计算相同的log-sum-exp，每个region只写入一次；反向显式按逆拓扑传递一阶导数。同一source的贡献先从零归并，再一次加入已有adjoint，避免改变参照的浮点求和分组。模型特征、目标、候选及搜索预算均未改变。

设批量大小为B、边界传递边数为E、region数为R，该传播算子保存B(E+R)个数值元素，静态拓扑计划为O(E+R)。这是arrival算子的存储界，不是整个框架的内存界。优化算子只支持一阶求导；不能用于声称二阶导数或Hessian已验证。

64364（`20261111T144801Z-bounded-arrival-regression-0a0473d0`）完成49项CPU矩阵：39 PASS、10个真实工具入口SKIP。新增测试覆盖重复source、重汇合、跨层边、并列/大数值、有限差分和384层链，并检查实际保存的数值元素数。原模型的数值梯度测试也通过。

同一保存div实例的全部校准loss/参数与完整40步梯度trace仍精确一致。含cProfile的初始化为2.63 s、30步校准21.13 s、40步梯度提案49.05 s；该任务MaxRSS约1.01 GiB。对照64363的99.05/105.82 s，仅描述两次保存输入剖析的测量差异，不将其写成全框架或独占资源的加速比。新的真实工具与完整c1355回归另存于`20261111T150227Z-bounded-arrival-real-regression-b71177d2`，其最终状态以对应报告为准。
