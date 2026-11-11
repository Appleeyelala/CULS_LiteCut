# 已复现的跨region替换引用环

905输入扫描中的index197 `forgeEDA/Inter_pred_sliding_window.aig`在旧`acd74fc`/113959工具下完成搜索，但独立region基线的首个配置以SIGSEGV退出，整例记录为PARTIAL。该方法保留no-op fallback，不把它记作候选组合成功。两个梯度模式的首次warmup也有同类原生失败，后续成功请求独立保留。

64340.10（`20261111T172127Z-selected-replay-crash-diagnosis-1e0694ff`）用相同输入字节、20项选择、owner表和原gpuls二进制，在GDB下稳定复现SIGSEGV。回溯为重复的`rw::CPUSolver::Fanin(int)`。没有执行新搜索或mapping。

64340.11首次内存检查因集群GDB不支持Python脚本而失败，保留为诊断环境错误。64340.12（`20261111T172704Z-replay-alias-memory-inspection-dd0019ee`）改用GDB原生命令读取第一处Fanin入口的实际内存。字段偏移由同一旧工具源码header编译获得；保存原始二进制内存、字段布局和完整JSON。三个未删除节点的引用形成环，100个活跃fanout/PO引用沿冗余链进入该环。

| 执行序号 | 原始root | 原始region | 内部root | replacement内部节点 | replacement原始root |
|---:|---:|---:|---:|---:|---:|
| 8 | 6868 | 1647 | 7273 | 4916 | 10577 |
| 10 | 8953 | 2256 | 7866 | 7273 | 6868 |
| 17 | 10577 | 2713 | 4916 | 7866 | 8953 |

三项trace均为ACCEPTED、saved=1、added=0。原始root环为`6868 → 10577 → 8953 → 6868`，内部节点环为`7273 → 4916 → 7866 → 7273`。这些是两个明确不同的ID命名空间。最后一项安装时，replacement的直接fanin没有指向当前root，但沿已有替换链两步后会返回root；旧的一层检查因此漏检。

这不是普通工具时限或已证明的GPU初始化问题，也不能用其他请求CEC通过来补充失败请求的时序标签。原图removable/cut-leaf冲突检查不足以保证实际共享节点和顺序替换后的依赖无环。

## 修复及验证边界

当前源码在GPU普通rewrite和wave顺序安装前遍历replacement的实际TFI；若到达当前root，拒绝该安装。普通replay记录`CYCLE_REJECT`，上层仍把不完整请求记FAIL，仅通过另一次显式修复请求移除被拒决策。遍历使用本轮复用的标记数组，同时检查两个fanin，不能只展开直接别名。

`Fanin`改为保留补相奇偶的有界迭代：合法长链不使用C++递归栈，非法引用或超过节点数的环明确抛错。原生CPU回归增加三节点环、fanin0依赖、补相奇偶和100001级合法别名链。64389已构建`d2065ab`工具，并通过链接真实solver对象的上述检查和两次常量输出CEC；CPU矩阵58 PASS/11 SKIP，未据CPU构建单独宣称GPU验证完成。

64341.6（`20261111T173748Z-acyclic-replay-real-regression-e1510460`）随后通过69/69真实核心测试，完成c1355和index197两种目标、每搜索8次预算的完整比较，分别56.80/165.97 s，均PASS。index197首个独立选择与旧崩溃请求逐项相同；它明确FAIL，其中`root10577/cut6/subgraph2`标记CYCLE_REJECT，另6项NO_OP。新的第1号修复请求实际接受13项rewrite并通过CEC/mapping，没有使用no-op fallback。原非法请求没有被改写成成功。

同一回归还确认c1355全部数值、选择、CEC及精确序列化图结构与缓存版参考相同。结果在`output/cycle_regression.json`；原生ABC时间注释不纳入图结构比较。共享分配的耗时不能解释为独立性能加速。该运行的逐程序矩阵终态见RUN_LOG，未据两个设计推断全905设计的修复覆盖。

完整诊断目录都在`/data/wushangge/CULS_LiteCut/runs/`，旧工具和全部失败产物保留。修复不会回写旧的ACCEPTED trace或PARTIAL记录。
