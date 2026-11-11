# 新开发阶段的运行记录

所有目录均在 `/data/wushangge/CULS_LiteCut/runs/`；原历史目录及 `evidence/` 未改写。下列“完成”仅指对应运行，不代表整体用户目标完成。

| Slurm ID | 目录后缀 | 结果与原因 |
|---|---|---|
| 64322 | `20261111T102740Z-historical-baseline-cab97f65` | FAIL：新编排脚本传错Gate3D raw trace目录；Gate3A–D完成 |
| 64323 | `20261111T102850Z-historical-baseline-826120e0` | FAIL：历史ABC通用read不识别无后缀Liberty别名；Gate3E完成 |
| 64324 | `20261111T103029Z-historical-baseline-51d97b14` | COMPLETED：Gate3A–F复跑，独立严格CEC通过，四组历史数值复现 |
| 64325 | `20261111T103249Z-framework-build-5e8e7b23` | FAIL：新增字段打乱C++旧聚合初始化顺序；已将字段移到末尾 |
| 64327 | `20261111T103732Z-framework-build-41106501` | FAIL：CULS/ABC均构建成功，测试类路径属性覆盖了unittest.run；已修复，二进制按hash复用 |
| 64328 | `20261111T104247Z-framework-regression-5c16847e` | COMPLETED：14项测试全部通过，包括真实rewrite/CEC/Liberty、326候选批量一致性 |
| 64329 | `20261111T104812Z-gradient-regression-826bf929` | COMPLETED：18项测试全部通过，新增有限差分梯度、联合选择与双目标模式 |
| 64330 | `20261111T105213Z-c1355-global-e2e-7961ed58` | 首次完整闭环产生summary；每种搜索24次预算；贪心与独立region有REVERTED，后续增加显式修复，不把部分安装当作成功 |

## 首次真实闭环的有限结论

64330的原始与no-op replay基线均为360.111877 ps、513.216003 area；ABC rewrite为359.501831 ps、529.778870 area。面积约束搜索最好记录349.582336 ps、513.449280 area；联合目标最好记录345.476715 ps、486.388794 area。两种搜索各有17次直接梯度提案实际评估，其余为基线/warmup；失败请求计入24次预算。

这些仅是一个设计、一个种子的初次结果。尚未据此宣称优于随机搜索、跨设计泛化或整体验收通过；两个确定性基线的失败请求需要修复后重新比较。原日志永久保留。

## 测量边界

新ABC只读插桩输出的PO11 **Liberty** arrival（例如GG=424.422180 ps）与历史Gate3F的PO11 **MIO** arrival（GG=194.1 ps）不同，不能替换历史列。新工具的GG global=425.052216 ps，与历史显示425.05 ps一致；GL/LG/LL也分别对应389.023682、369.597626、388.937592 ps。关键PO并列集合与原生trace各自记录，不把mapper数字ID跨运行等同为原AIG逻辑点。
