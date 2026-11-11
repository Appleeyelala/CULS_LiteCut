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
| 64340.0 | `20261111T121113Z-corrected-tools-regression-538f79e0` | COMPLETED_WITH_EXPLICIT_OMISSIONS：独立step在原分配内运行，33项真实核心测试全部通过，c1355双目标8预算完成、都未改善基线；原生writer/Gate1/Gia及2486项矩阵完成；重复运行选择/数值/结构hash一致，ABC注释导致字节hash不同 |
| 64341.0 | `20261111T121235Z-corrected-cross-design-pilot-13f96501` | COMPLETED：33项真实回归及8/8设计完整比较通过；包括六个此前失败设计、c1355和中位设计；不是代表性随机抽样 |
| 64340.1 / 64341.1 | `20261111T121807Z-corrected-full-shard0-93ae61a2` / `20261111T121809Z-corrected-full-shard1-29165762` | RUNNING：修复后的全905输入双分片；独立step共享已有分配，旧batch未取消；各step先通过33项真实回归 |
| 64349 | `20261111T122037Z-historical-analysis-regression-324c8936` | COMPLETED：四个R9分析入口用隔离历史输入执行通过，依赖文件hash未改变；2487项程序矩阵合入来源匹配的历史Gate3/R9功能证据，120 import PASS、17 FAIL、1 SKIP |
| 64350 | `20261111T122536Z-paired-results-regression-b1f525fb` | COMPLETED：35项CPU测试（26 PASS、9个GPU入口SKIP）；8设计实际配对统计与修复版c1355重复比较通过，包含最终选点检查 |
| 64351 | `20261111T124031Z-arrival-path-observer-build-1ab72688` | COMPLETED：构建可选Liberty最大arrival路径observer；实际ABC路径重构及插桩前后网表/全部PO/面积一致性测试通过；GPU测试由64340.2补齐 |
| 64340.2 | `20261111T124507Z-arrival-path-full-ci-6caaeabb` | COMPLETED_WITH_EXPLICIT_OMISSIONS：38项真实核心回归全部通过，c1355双目标smoke、原生writer/Gate1/Gia及2492项程序矩阵完成；缺失历史训练环境明确保留 |
| 64352 | `20261111T124509Z-saved-selection-path-analysis-2bda1d0e` | FAIL：分析wrapper误按搜索method定位复用的no-op文件；原搜索结果未改动 |
| 64353 | `20261111T124930Z-saved-path-analysis-regression-c851ab7f` | FAIL：首次修复假设了错误的replay路径层数；随后改为按实际observation路径和输入hash定位，并增加回归 |
| 64354 | `20261111T125125Z-path-provenance-regression-1876ee41` | COMPLETED：39项CPU矩阵（29 PASS、10 SKIP）；8设计168个成功观察引用对应124份映射，重新CEC、全部PO/global/area、网表正文与逐arc路径重构全部通过 |
| 64355 | `20261111T130741Z-selection-cost-regression-3cba9919` | COMPLETED：41项CPU矩阵（31 PASS、10 SKIP）；c1355/des_perf/div的完整catalogue均与保存结果一致，24组坐标选项成本与标量参考逐位相同；c1355/des_perf完整选择一致。hyp归一化完成但无旧完整catalogue可比；div无旧完整选择可比，均明确标记 |

## 首次真实闭环的有限结论

64330的原始与no-op replay基线均为360.111877 ps、513.216003 area；ABC rewrite为359.501831 ps、529.778870 area。面积约束搜索最好记录349.582336 ps、513.449280 area；联合目标最好记录345.476715 ps、486.388794 area。两种搜索各有17次直接梯度提案实际评估，其余为基线/warmup；失败请求计入24次预算。

这些仅是一个设计、一个种子的初次结果。尚未据此宣称优于随机搜索、跨设计泛化或整体验收通过；两个确定性基线的失败请求需要修复后重新比较。原日志永久保留。

64331修复后的完整比较：LiteCut贪心359.873505 ps/533.511353 area，独立region399.386475 ps/523.013794 area；面积约束搜索最佳仍为360.111877 ps/513.216003 area，联合目标最佳360.111877 ps/510.416656 area。不能只引用64330的改善。

64334小预算跨设计pilot中，c5315梯度面积约束搜索404.531586 ps，随机414.942444 ps；c7552梯度1020.499270 ps，随机927.469604 ps；jpeg_hm_decode独立region3040.164790 ps，梯度搜索维持no-op 9142.371090 ps。预算每搜索8次、单种子；没有建立稳定的算法优势。

## 修复版的阶段结果

64341.0的8设计检查中，面积约束梯度相对相同已消耗预算的随机搜索为2胜、5平、1负；联合目标为1胜、4平、3负。这里按各自实际目标判断数值胜负，joint胜不等于delay单项更低。样本包含定向选择的失败案例，不能据此推断全数据集效果或统计显著性。

`DSP_comb`的原始delay为4208.673340 ps，零rewrite replay为5567.682620 ps，二者CEC通过；梯度面积约束结果3847.503660 ps，随机3635.908200 ps。此例说明序列化对mapper的影响可能很大；报告必须分别对照original与no-op，不能仅报相对较差no-op的收益。修复版c1355两模式仍为360.111877 ps / 513.216003 area，未改善。

修复版c1355两次完整搜索的选择、最终选点、CEC、全部PO、面积及精确序列化图结构一致；Native ABC输出时间注释导致原始文件字节不同。完整比较见64350运行的`output/paired_report/repeat_comparison.json`。同一c1355运行的12个预先预测样本global-delta MAE约71.03 ps、Pearson约−0.63；这是自适应搜索内小样本诊断，不能宣称代理准确或跨设计泛化。

## 全量扫描暴露的接口问题

1. 命名AIG与无符号名replay：旧默认CEC按名称匹配而失败。新接口先检查PI/PO数量，再明确按原AIGER位置证明，保留命名差异。
2. cut自由变量检查过强：支持最小化和原始叶子逻辑关系会造成差异。六个案例的实际上下文CEC证明全部10,281个导出decision，不把旧误报伪装成GPU布尔错误。
3. 常量导出实际错误：`CPUSolver::Reset`已转为AIGER的0=false，但旧`WriteAig`再次翻转0/1。b14_1的前两个PO从0/1变成1/0。新导出直接写solver literal，同时初始化常量/PI phase与reorder状态；旧二进制和错误输出保留供回归。

## 测量边界

新ABC只读插桩输出的PO11 **Liberty** arrival（例如GG=424.422180 ps）与历史Gate3F的PO11 **MIO** arrival（GG=194.1 ps）不同，不能替换历史列。新工具的GG global=425.052216 ps，与历史显示425.05 ps一致；GL/LG/LL也分别对应389.023682、369.597626、388.937592 ps。关键PO并列集合与原生trace各自记录，不把mapper数字ID跨运行等同为原AIG逻辑点。

新增路径observer不依赖native `stime -p`的min-slack回溯作为指定端点的最大arrival路径。它在相同SC timing状态下读取load/slew、cell arc与rise/fall相位，选择能重构端点arrival的最大前驱，验证每级累计值。64354实际复核的124份映射来自8设计、107份不同AIG字节；同一AIG可能对应不同保存的mapped正文，因而不只按AIG去重。完整证据为该运行的`output/arrival_paths/summary.json`及逐映射`paths.json`。

此处固定PI/PO索引从0开始，只比较源/端点集合及当前映射内的链。它不是原图候选到cell的逐项对齐，也没有证明某条布尔可敏化的唯一物理路径切换。

## 大设计选择开销检查

64355在CPU任务中只读取保留的实际输入，不执行rewrite/mapping。`div`有51,140个executable decisions，最大region含1,016个decision；8组坐标选项的旧标量累计耗时7.41 s，优化后0.327 s，成本逐位一致。完整独立region选择耗时39.61 s；旧全量运行在此阶段超过1800 s单例上限，没有完成的旧选择可比较。`des_perf`完整选择与已保存请求完全相同，当前22.45 s；历史90.54 s来自不同并发负载，只能作诊断。

`hyp`有117,509个region、979,378个legal cuts。共享只读region集合后，完整catalogue耗时4.52 s；512条实际cut的归一化参考耗时1.027 s、优化后0.00304 s，内容相同。此处不把子集计时外推为整个优化框架的加速，也不把纯CPU诊断记作完整benchmark通过。后续须补齐超时设计的真实E2E。
