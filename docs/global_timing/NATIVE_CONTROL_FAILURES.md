# 独立Native ABC对照的失败隔离

index108 `Back_End_With_Decode`在`20261111T174715Z-acyclic-terminal-case-retries-7cce9a09`通过准备及no-op CEC/mapping后，Native ABC rewrite的CEC耗时900.21 s仍未结束。旧流程在此抛错，使后续所有搜索未执行，整例1337.65 s FAIL；这不是不等价证据，原结果保留。

`94909a7`将这个独立对照的预期工具、I/O及测量错误保存为`native_abc/attempt.json`。其summary指标为null，不进入排名、不作为训练标签，也不使用no-op代替。其余方法从原始AIG执行自己的严格验证。只要有独立对照或确定性方法未成功，suite就记PARTIAL。

原始输入和no-op测量仍是必需条件，失败继续中止搜索。内部程序错误记录后重新抛出，不当作可忽略的对照失败。报告、重跑比较和路径审计都明确处理native缺失；成功路径只引用取得CEC与mapping证据的观察。

64392的75项CPU矩阵为63 PASS/12 SKIP。64341.8的同源码真实核心75/75 PASS，其中故障注入入口只拦截native对照的CEC，其他命令委托真实ABC。该测试取得其余方法的实际测量，9份映射、12个成功路径引用和3个预测/路径关联均通过复核；没有native数值或路径替代项。此人为故障仅用于集成测试，不作为benchmark性能或自然超时案例。

同一回归的c1355和index197完整8预算双目标比较PASS；c1355全部数值、选择和精确序列化图结构与保存参照一致，index197仍拒绝原循环请求并在另一次13项安装后通过CEC/mapping。全程序矩阵与最终任务状态另见`RUN_LOG.md`。

真实工具回归目录：`/data/wushangge/CULS_LiteCut/runs/20261111T182419Z-native-control-real-regression-2c5d4d0b/`。关键产物为`output/tests.json`、`output/native_control_regression.json`及`output/cycle_regression.json`。它尚未证明index108已在新流程完成完整搜索。

后续自然补验64340.14（`20261111T184033Z-isolated-native-backend-retry-813f3b26`）使用真实ABC，native CEC在900.30 s超时、returncode −9。`native_abc/attempt.json`保持FAIL，流程随后进入`greedy_litecut`；没有故障注入或更换对照数值。此时整例仍在运行，仅证明自然超时已被隔离，不提前宣称其他搜索完成、整例PARTIAL或PASS。
