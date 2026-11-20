# 代理预测与联合选择诊断

更直接的证据指向 A：代理对编辑收益的判断有误。38 个电路零编辑延迟的平均绝对相对误差为 43.1%。32 个非空提案全部被预测为改善，ABC 实测有 11 个比原图更慢；改善幅度的 Pearson 为 0.351，Spearman 为 0.586。

联合选择确实能起作用。c5315 的 16 个编辑单独都没赢过零编辑回放，联合后却从 450.51 降到 428.28 ps。四个电路共 28 个梯度步，27 步在路径外也有梯度。路径来自离散候选，梯度来自概率混合状态，两者可能不一致；unreg 的复跑最终选择与历史不同。

用真实单编辑收益相加选组合，仅在 3 例中的 1 例胜过代理加和。三个池里开、关 LiteCut 均选出了相同组合。样本很少，无法定量分摊 A、B 的责任。优先核验代理对同一电路多个组合的排序。

## 表与图

| 任务 | 文件 |
|---|---|
| 1 · 原图 | [38 个电路逐项](baseline_by_circuit.csv) · [两组及合并统计](baseline_summary.csv) |
| 2 · 改写 | [32 个候选逐项](edit_by_circuit.csv) · [统计与误判名单](edit_summary.csv) · [散点图](edit_scatter.pdf) |
| 3 · 梯度与路径 | [逐步计数](gradient_steps.csv) · [入选改写逐项](gradient_path_options.csv) · [复跑对照](trace_parity.csv) |
| 4 · 三组单编辑对照 | [三组与历史方案](oracle_arms.csv) · [42 个单编辑](oracle_single_edits.csv) |

[指标定义、逐电路解读与复现](METHODS.md) · [完整梯度记录索引](gradient_sources.json) · [三组对照的作业和输入](oracle_sources.json)
