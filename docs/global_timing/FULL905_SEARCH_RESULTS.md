# 统一版本905输入的搜索结果

本页报告本轮已完成的框架实现、全905主实验及原始证据审计。完成工程验证不等于方法具有稳定优势，也不表示历史训练环境或所有可选程序均已恢复；明确遗漏与逐项验收见[REQUIREMENTS_AUDIT.md](REQUIREMENTS_AUDIT.md)。

## 实验身份与覆盖

两个预定分片64399/64397使用同一benchmark源码`160b2187eee3b47799e58b658e39f13d40064b61`、相同工具与模型/ASAP7 hash、种子20261111。每种搜索最多8次请求，包含基线、warmup、失败与修复；工具/整例上限900/7200 s。各输入均比较original、replay no-op、native ABC、LiteCut贪心、独立region、随机和两种梯度目标。

905个索引及文件hash唯一，901 PASS、4 PARTIAL。codec、loco_top、mrc3、mrc4仅缺native ABC rewrite对照，保留断言失败及空指标；其他方法的实际结果参与比较，不丢弃整个设计。

64854的分析/编排源码为`fe39364`，与benchmark源码身份分开记录。它在59m40s内正常结束，Slurm MaxRSS978560 KiB；这是保存数据分析时间，不是优化器速度或独占资源benchmark。

## 各baseline与序列化对照

64905已生成`output/performance_overview/`，仅汇总64854冻结cohort的保存数值。下表的胜平负只比较global delay，不作为joint目标胜负或等成本结论。

| 方法 | 有指标/缺失 | 满足no-op面积1.05倍 | 相对原图delay胜/平/负 | 相对no-op delay胜/平/负 |
|---|---:|---:|---|---|
| Native ABC rewrite | 901/4 | 828 | 471/210/220 | 483/167/251 |
| Original AIG | 905/0 | 898 | 0/905/0 | 266/319/320 |
| Replay no-op | 905/0 | 905 | 320/319/266 | 0/905/0 |
| 面积约束梯度 | 905/0 | 905 | 658/158/89 | 608/297/0 |
| joint梯度 | 905/0 | 900 | 656/150/99 | 583/304/18 |
| LiteCut贪心 | 905/0 | 836 | 510/139/256 | 481/185/239 |
| 独立region | 905/0 | 816 | 499/167/239 | 453/213/239 |
| 面积约束随机 | 905/0 | 905 | 658/161/86 | 585/320/0 |
| joint随机 | 905/0 | 891 | 647/158/100 | 568/319/18 |

面积约束/joint梯度相对原图的逐设计delay改善中位数分别为3.6586%/3.8708%，delay几何平均比为0.898432/0.899813（905个正值配对）。这些收益与下面的随机对照一起解释，不能据相对原图的改善归因于梯度或LiteCut。no-op本身相对原图已有320改善、319相同、266恶化，序列化影响必须单列。

面积约束模式的905个最终点均满足真实no-op面积1.05倍限制；joint没有该硬约束，其中5个点超过另一模式的限制。确定性基线调用次数较少；完整面积、ADP、各输入族与比例有效分母见原JSON/CSV，不将它们误写为同预算搜索。

## 相同已消耗预算的随机对照

| 梯度模式 | 配对设计 | 胜 | 平 | 负 | 中位目标改善 |
|---|---:|---:|---:|---:|---:|
| 面积约束下最小delay | 905 | 179 | 464 | 262 | 0% |
| Smooth Tchebycheff delay–area联合目标 | 905 | 252 | 379 | 274 | 0% |

面积模式按满足真实面积约束的delay比较；joint按真实联合目标比较，其胜负不是delay单项胜负。当前结果未建立稳定优于随机搜索的优势；单种子、小预算、相关电路变体也不支持显著性或独立泛化结论。original与no-op的变化分别保留，不将序列化损失的恢复算成相对原图收益。

## 预测与请求来源

| 指标 | 面积约束 | joint |
|---|---:|---:|
| 评估前预测样本 | 3128 | 3197 |
| 有可归一化预测的设计 | 811 | 811 |
| 按设计平均global-delta归一化MAE | 0.117150 | 0.121337 |
| 实际变化样本的按设计平均方向正确率 | 0.436117 | 0.450463 |
| 直接梯度请求成功/总数 | 1567/3339 | 1685/3390 |
| 全部失败请求 | 2418 | 2349 |
| 最终选中直接梯度点 | 153 | 207 |
| 最终选中显式修复点 | 397 | 388 |
| 最终保留no-op | 297 | 243 |
| 最终选中warmup点 | 57 | 66 |
| 最终选中重复投影后的探索点 | 1 | 1 |

归一化误差为`abs(predicted_delta - actual_delta) / measured_noop_delay`，先按设计平均；不是逐样本相对实际delay误差。方向统计只含实际变化的样本。基线锚点、失败请求不计准确性，预测只引用更早成功标签；模型随反馈变化，不能当作固定模型或未见设计测试。

修复点可能继承梯度请求，但其选择来源与直接parent来源分别保存，不推断因果贡献。直接请求的低通过率与模型误差均是当前限制，不能只展示最后CEC通过的点。

## 审计与原始文件

64854的905个设计搜索来源审计错误0；额外读取原始cut证明、候选域/冲突、choices文件、顺序trace、AIG hash及原CEC，执行审计905 PASS。每个成功请求的指定候选都确实ACCEPTED，未将部分安装当成功；失败请求保持无替代指标。64905另行完成了原始timing/路径的全量交叉核验。

完整目录：`/data/wushangge/CULS_LiteCut/runs/20261112T103045Z-final905-search-analysis-v2-32c72d46/`。

| 文件 | 内容 / SHA256 |
|---|---|
| `output/cohort/cohort.json` | 完整固定清单；`fd01816690ea817d250a78a4661a4949adf2bae15e4c77d62f672f0a0406a044` |
| `output/comparison/comparison_all.csv` | 各方法实际delay、area、ADP、预算、runtime、CEC及来源 |
| `output/comparison/paired_statistics.json` | 配对比较；`a90315d3e1a50a7bfa3609bd0e2b2e3ffbcc20161e1038a4f9f8937b71ac03f8` |
| `output/search_quality/summary.json` | 来源与误差；`4e279418811c702aa6641e5cff6106e0e06e5e3f8ba815e04464696b8442a3bd` |
| `output/search_quality/` | 逐输入报告、预测/梯度CSV、失败证据和PNG/PDF图 |
| `output/execution_audit/summary.json` | 原始执行核验；`d2798f89f67b9b0ecd92f56b76f8b83a28736e391b0c2edfaa139ffb015bbeed` |
| `output/execution_audit/evidence_hashes.json` | 原始证明、trace和观察文件的hash |
| `output/completion_audit.json` | 两主实验及前十二批路径的终态核验；`01638a0bd786e9719bb67f028da09673c01d6dc3162b18eb9d70888c466f9623` |

第十三批64852已完成167/167设计、2657份映射、3158个成功引用，Slurm COMPLETED/0:0、4h00m51s、MaxRSS2807740 KiB。完成记录SHA256为`f616a55252bea6769c344e20a2a860b85649d2192c1cdd055fa90d848a709055`；原选择/cohort hash不变。

十三批精确目录互斥、合计905设计的有效测量审计PASS，13981份映射、16696个成功引用，逐例唯一AIG字节计数之和10758；这些不是跨设计的全局去重数量，四项native缺失仍保留。

64905已完成原始CEC/Liberty日志/netlist/arrival链的最终只读交叉核验：905/905 ANALYZED、审计错误0、无损坏快照或未选定报告，只有一个benchmark源码和一个observer版本。Slurm COMPLETED/0:0、1h22m13s、MaxRSS8219292 KiB；四项native缺失仍保留。目录为`/data/wushangge/CULS_LiteCut/runs/20261112T122920Z-final905-path-analysis-61f92cbb/`。

| 最终文件 | SHA256 |
|---|---|
| `output/final_validation.json`（PASS） | `67c346b78f3415017830be96939cbb0cd7c976c65b18990311e6260b62f2db66` |
| `output/completion_audit.json` | `3a835f1476813106382ea387254a31b10f6061871a80e130fc43cb41004cee60` |
| `output/performance_overview/summary.json` | `c78097ea1de57bb70763b42749733b7e4279eee39a6cdb1e1e6b4cd1f4882be0` |
| `output/performance_overview/method_summary.csv` | `d90435a886873414fb42fc77cb8eb594f023d4f65c7baa67667d8431514b5d68` |
| `output/path_changes/summary.json` | `38a6e3e0610ddafff585ccd8efb328c56454ea27820e2a808eb4871460ae8d1d` |
| `output/path_changes/coverage.json` | `9d1c26fe0d6f66891313a3474ab10e295077cf7467f1b608cbad08d33683abcd` |
| `output/path_changes/evidence_hashes.json` | `a32c1c577e3090f57fff98307913cf0ecc5804fe1d54057564035352efb8b368` |

## 关键端点变化与误差

237/905个设计在original→no-op序列化阶段就改变代表PI/PO对。两种梯度模式各有430/905个保存最终选点相对no-op改变代表PI/PO对；不能将这些计数解释为唯一逻辑路径或布尔可敏化路径切换。

在成功非锚点观察中，面积约束/joint分别有1491/1539次critical PO集合变化、245/275次PO集合不变但代表PI变化。按设计等权的归一化Δdelay MAE，在critical PO变化组为0.176418/0.179459，在代表PI/PO对集合相同组为0.044538/0.054245。类别包含不同设计和同一设计的相关观察，差异是描述性关联，不证明路径变化导致误差。

完整逐观察、失败间隔、rise/fall、近关键集合、分组分母及PNG/PDF见`output/path_changes/`和[路径分析说明](PATH_CHANGE_ANALYSIS.md)。随机搜索有两种目标排名，不擅自指定统一final。框架的返回点经过真实校验；静态投影不保证所有梯度提案可执行，代理的准确性及稳定优化优势仍未建立。
