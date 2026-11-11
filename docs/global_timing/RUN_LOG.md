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
| 64331 | `20261111T105804Z-c1355-repair-e2e-e04ad899` | COMPLETED：20项回归；修复请求计预算后，面积约束搜索未优于baseline；负结果保留 |
| 64332 | `20261111T105821Z-benchmark-inventory-114a3573` | COMPLETED：905个组合binary AIG逐文件暂存/hash，均唯一；总20,561,771 AND |
| 64333 | `20261111T110315Z-sparse-regression-79327bd5` | COMPLETED：20项稀疏传播与真实工具回归 |
| 64334 | `20261111T111356Z-benchmark-scale-pilot-d516a378` | COMPLETED：22项回归；c1355/c5315/c7552/jpeg_hm_decode四个8预算完整比较 |
| 64335 | `20261111T111703Z-repository-test-matrix-a85876d4` | COMPLETED：2476项源码/程序清单；首轮导入探针过度拦截/dev/null，结果不替代功能测试 |
| 64336 | `20261111T111908Z-integrity-and-scale-regression-7f453a81` | COMPLETED：26项回归，增加负向CEC、ABC失败、原cut真值及独立region快捷计算对照 |
| 64337 | `20261111T112204Z-concurrent-suite-pilot-64109582` | COMPLETED_WITH_FAILURES：c1355通过；中位电路因把功能支持cut当结构边界而误拒绝 |
| 64338 | `20261111T112224Z-repository-import-regression-a19a25d0` | COMPLETED：修正/dev/null与工具搜索路径；83个import PASS，18 FAIL、27受探针约束SKIP；后续允许只读lscpu/fc-list查询 |
| 64339 | `20261111T112634Z-support-cut-regression-4ac56f58` | COMPLETED：29项回归；c1355与中位电路并发比较完成，发现8条score涉及功能支持最小化 |
| 64340/64341 | `20261111T112917Z-full-benchmark-shard0-864a4c8b` / `20261111T112919Z-full-benchmark-shard1-5bb41012` | 旧版本全量扫描进行中；出现命名接口、cut上下文/BDD规模及常量输出问题；不作为最终全量验收 |
| 64342 | `20261111T113159Z-cross-design-integrity-audit-0f3668f1` | COMPLETED：独立读取已导出cut，定位边界自由变量真值差异 |
| 64343 | `20261111T113648Z-context-proof-regression-525333c5` | COMPLETED：31项CPU测试（8个GPU入口SKIP）、4项真实ABC回归；六个保留案例10,281个cut全部证明；两个旧no-op仍因常量翻转不等价 |
| 64344 | `20261111T113959Z-constant-safe-tool-build-5e7be150` | BUILT_NOT_RUNTIME_VALIDATED：修复常量导出/未初始化状态后重新编译；尚待GPU回归 |
| 64345 | `20261111T114605Z-native-writer-and-import-tests-71839ae5` | COMPLETED：链接实际solver对象，连续两次Reset/Reorder/WriteAig的常量CEC与mapping通过；逐程序矩阵完成，导入缺失依赖仍显式记录 |
| 64346 | `20261111T115338Z-imported-native-and-cli-tests-0dd6d0c7` | COMPLETED：Gate1 io_signature、候选选择器和ABC四项Gia原生单元测试全部通过；未下载依赖 |
| 64347 | `20261111T115530Z-ci-cpu-and-existing-reports-bdf90782` | FAIL：CI父进程提前导入绘图库使子进程继承不兼容MKL线程层；调整导入顺序，保留失败日志 |
| 64348 | `20261111T115853Z-ci-cpu-and-reports-regression-c08b3bd8` | COMPLETED：32项CPU矩阵（23 PASS、9个GPU入口SKIP）；6组已保存实验生成图表；重复运行数值/选择一致，Native ABC输出字节hash因时间注释不同，进一步按精确图结构区分 |

## 首次真实闭环的有限结论

64330的原始与no-op replay基线均为360.111877 ps、513.216003 area；ABC rewrite为359.501831 ps、529.778870 area。面积约束搜索最好记录349.582336 ps、513.449280 area；联合目标最好记录345.476715 ps、486.388794 area。两种搜索各有17次直接梯度提案实际评估，其余为基线/warmup；失败请求计入24次预算。

这些仅是一个设计、一个种子的初次结果。尚未据此宣称优于随机搜索、跨设计泛化或整体验收通过；两个确定性基线的失败请求需要修复后重新比较。原日志永久保留。

64331修复后的完整比较：LiteCut贪心359.873505 ps/533.511353 area，独立region399.386475 ps/523.013794 area；面积约束搜索最佳仍为360.111877 ps/513.216003 area，联合目标最佳360.111877 ps/510.416656 area。不能只引用64330的改善。

64334小预算跨设计pilot中，c5315梯度面积约束搜索404.531586 ps，随机414.942444 ps；c7552梯度1020.499270 ps，随机927.469604 ps；jpeg_hm_decode独立region3040.164790 ps，梯度搜索维持no-op 9142.371090 ps。预算每搜索8次、单种子；没有建立稳定的算法优势。

## 全量扫描暴露的接口问题

1. 命名AIG与无符号名replay：旧默认CEC按名称匹配而失败。新接口先检查PI/PO数量，再明确按原AIGER位置证明，保留命名差异。
2. cut自由变量检查过强：支持最小化和原始叶子逻辑关系会造成差异。六个案例的实际上下文CEC证明全部10,281个导出decision，不把旧误报伪装成GPU布尔错误。
3. 常量导出实际错误：`CPUSolver::Reset`已转为AIGER的0=false，但旧`WriteAig`再次翻转0/1。b14_1的前两个PO从0/1变成1/0。新导出直接写solver literal，同时初始化常量/PI phase与reorder状态；旧二进制和错误输出保留供回归。

## 测量边界

新ABC只读插桩输出的PO11 **Liberty** arrival（例如GG=424.422180 ps）与历史Gate3F的PO11 **MIO** arrival（GG=194.1 ps）不同，不能替换历史列。新工具的GG global=425.052216 ps，与历史显示425.05 ps一致；GL/LG/LL也分别对应389.023682、369.597626、388.937592 ps。关键PO并列集合与原生trace各自记录，不把mapper数字ID跨运行等同为原AIG逻辑点。
