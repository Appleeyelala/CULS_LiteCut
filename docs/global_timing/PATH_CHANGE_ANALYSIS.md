# 已保存的关键端点变化分析

分析使用同一Liberty时序口径下的已保存映射和最大arrival链，不把历史Gate3F的MIO PO列混入。入口为`scripts/analyze_path_changes.py`，纯比较逻辑在`global_timing/path_changes.py`。每次使用新的Slurm数据目录，原搜索和路径审计不改写。

## 已完成的首批分析

运行`20261111T165927Z-saved-path-change-audit-c21e0dbe`，源码`01889c4`，Slurm 64386已完成（6m07s、MaxRSS513212 KiB）。它读取64381冻结的125输入清单及该路径审计当时已完成的39份逐设计报告。67项CPU测试为56 PASS/11 SKIP；逐程序矩阵完成，历史依赖缺项独立保留。

39个设计对应611份映射、728个成功观察引用；重新读取保存的CEC、mapping日志、AIG/netlist、全部PO arrival及逐arc链重构，审计错误0。这里只读取文件，没有再次执行CEC、rewrite或mapping。

原清单的覆盖为39个已分析、77个成功benchmark尚无完整路径报告、9个benchmark非PASS、780个输入尚无终态记录。快照不随后续输出变化；缺失路径报告不等于benchmark失败。三个benchmark源码版本分别为`acd74fc`的37个、`c9a1652`的1个、`5b580a4`的1个；observer二进制相同，SHA256前缀`e492170d5332`。这不是单一最终版本、随机抽样或全905设计结果。

| 方法 | 成功非锚点观察 | 关键PO集合变化 | PO集合不变但代表PI变化 | 近关键PO集合变化 | 保存最终选择的PI/PO对变化 |
|---|---:|---:|---:|---:|---:|
| 面积约束梯度搜索 | 143 | 101 | 11 | 113 | 27/39 |
| 联合目标梯度搜索 | 143 | 83 | 7 | 100 | 24/39 |
| LiteCut贪心 | 39 | 29 | 2 | 34 | 31/39 |
| 独立region | 39 | 30 | 2 | 32 | 32/39 |
| 随机搜索 | 130 | 97 | 12 | 112 | 不设统一final |

上表均相对本设计保存的no-op基线。最终选择可能来自warmup、修复或no-op，不等于直接梯度提案贡献；随机搜索有两种目标排序，不能混作一个final。变化次数本身不是优化收益。

39个设计中27个在original→no-op序列化对照中就出现代表PI/PO对变化，因此搜索效果和序列化影响必须分别报告。面积约束/联合搜索分别有62/59次相邻成功观察跨越了失败请求；分析保留失败间隔，不给失败组合填入时序或路径。

## 解释边界

- critical PO集合采用2e-5 ps绝对容差；近关键阈值是global arrival的95%。每个PO保留一条代表最大链，不枚举所有并列前驱或并列rise/fall链。
- PI/PO索引沿固定接口位置比较。mapped节点ID只在单次映射内有效，不跨映射匹配。
- PO、源PI或相位变化描述观测到的代表链变化，不证明唯一逻辑路径身份切换或布尔可敏化性；集合不变也不证明内部逻辑链相同。
- 完整报告来自正在执行的审计的一个时间前缀，存在完成时间选择效应。观察彼此相关，计数不支持统计显著性、泛化或算法优势。
- 每份成功路径报告必须覆盖该次搜索全部成功观察。其他重跑即使输入hash相同，也不能代替冻结清单选定的benchmark目录。

全部文件位于`/data/wushangge/CULS_LiteCut/runs/20261111T165927Z-saved-path-change-audit-c21e0dbe/output/path_changes/`：`summary.json`含版本分层，`coverage.json`保留905项状态，`observations.csv`和`transitions.csv`保留逐观察及相邻成功转移，`evidence_hashes.json`记录原始证据hash，`input_snapshots/`冻结逐设计报告，PNG/PDF保存对照图。
