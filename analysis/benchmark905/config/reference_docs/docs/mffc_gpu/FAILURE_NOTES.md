# 正式905：失败与缺失测量记录

最终全905/2715已完成并审计：A/Ours各905 PASS，B为901 PASS/4 FAIL，方法级TIMEOUT为0；另有Ours内部2个失败请求，均发生于同一次BRAM_SSP CUDA分配失败，最终回退成功。下文保留运行时观察的时间顺序，早期“尚非最终”指当时截点；最终数量以此段及[全量结果](../../BENCHMARK905_RESULTS.md)为准。冻结实验源为`ecc85e6`，Slurm65245，正式目录为`/data/wushangge/CULS_LiteCut/runs/20261113T191340Z-mffc-full905-frozen-6a45b8db`。不重跑失败点、不替换输入或工具、不用其他方法的delay填补缺项。

## index392：codec的基线B原生Rewrite断言

2026-11-14约07:33观察到`forgeEDA/codec.aig`的B方法终态FAIL。输入SHA256为`3898206a7d9a6d64835aa8ab6cced17f9dd073762f300c9c7b114cffba7e35a9`。

| 项目 | 原始记录 |
|---|---|
| 阶段 | `read_aiger; strash; rewrite; write_aiger`中的原生Rewrite |
| 退出码 / 原生调用时间 | −6 / 0.214344236s |
| B冷启动端到端时间 | 0.518344442s，失败耗时仍保留 |
| 断言 | `src/base/abc/abcAig.c:1134: Abc_AigUpdateLevelR_int: Assertion Abc_ObjIsNode(pNode) failed` |
| CEC / Mapping / Delay / Area | 未取得该方法的成功验证及测量；指标不可用，不填零 |
| 相同输入A | PASS，冷E2E 2.072187981s |
| 相同输入Ours | PASS，冷E2E 18.102447693s，来源`original_fallback`；不能据此称该输入有梯度收益 |

原始文件（均相对正式目录）：

- `output/records/0392_{A,B,Ours}.json`：三个方法的输入hash、终态及实际时间。
- `output/case_0392_B/work/native_rewrite.{command.json,log}`：完整冻结命令、退出码与断言文本。
- `output/case_0392_B/cold_method.{command.json,log}`：独立子进程失败与调用栈。
- `output/case_0392_{A,Ours}/child_result.json`：各自独立取得的成功结果；Ours使用自己付费测量的original回退。

旧版记录也曾在该输入的native rewrite阶段观察到同名断言，见[历史对照失败隔离](../global_timing/NATIVE_CONTROL_FAILURES.md)。相同断言位置不证明内部根因完全相同。本次没有修改ABC版本或recipe，也没有把旧结果作为新运行指标。

该异常被正式驱动隔离为B的FAIL，原始结果保留；检查时Slurm65245仍RUNNING，已继续发布后续输入的方法终态。最终统计保留该输入的A/Ours数据；涉及B的成功质量/时间配对排除此缺失测量，全部终态时间仍包含失败耗时。方法终态与Ours内部候选请求的失败分开统计。

## index562：loco_top的基线B原生Rewrite断言

2026-11-14约09:23观察并读取`forgeEDA/loco_top.aig`的B方法终态FAIL。输入SHA256为`144a42f71a7b37d3a7e3afb91fdd7b368cce38a7f0847b25ba422df5c2845e56`，A/B/Ours记录一致；仍为同一冻结源码、ABC和单次rewrite命令。

| 项目 | 原始记录 |
|---|---|
| 阶段 | `read_aiger; strash; rewrite; write_aiger`中的原生Rewrite |
| 退出码 / 原生调用时间 | −6 / 0.214363001s |
| B冷启动端到端时间 | 0.569874013s，失败耗时仍保留 |
| 断言 | `src/base/abc/abcAig.c:1134: Abc_AigUpdateLevelR_int: Assertion Abc_ObjIsNode(pNode) failed` |
| CEC / Mapping / Delay / Area | 未进入CEC/mapping；没有对应日志或成功测量，指标不可用 |
| 相同输入A | PASS，冷E2E 2.171938984s |
| 相同输入Ours | PASS，冷E2E 18.661857780s，来源`mffc_global_gradient`；B缺失不构成相对B的性能胜出证据 |

原始证据为正式目录下的`output/records/0562_{A,B,Ours}.json`、`output/case_0562_B/work/native_rewrite.{command.json,log}`和`output/case_0562_B/cold_method.{command.json,log}`。这与codec具有相同断言位置；旧版也有同名输入记录，但不能由此断言内部根因已经完全定位。

该次读取时65245仍RUNNING，驱动已经继续后续输入，两个后处理仍PENDING/Dependency。本次只记录新观察，不修改冻结算法、ABC或参数，不重跑失败点；最终B的有效质量配对排除此缺失测量，全部终态时间保留失败成本。09:23观察到的方法失败为codec/B和loco_top/B，尚非全905最终计数。

## index610/611：mrc3、mrc4的基线B原生Rewrite断言

2026-11-14约09:49逐项读取两例B的终态、原生命令和日志，确认均为FAIL；仍使用冻结源码`ecc85e6`、相同ABC与单次原生rewrite流程。断言均为`src/base/abc/abcAig.c:1134: Abc_AigUpdateLevelR_int: Assertion Abc_ObjIsNode(pNode) failed`，native返回−6。未进入CEC/mapping，无相应日志，delay与area缺失，不补零或旧结果。

| 输入 / index | 输入SHA256 | native调用s | B冷E2E s |
|---|---|---:|---:|
| `forgeEDA/mrc3.aig` / 610 | `dfbbab07cbddac9577af876a37c0c6a0ab5ce20c5345ffcb5927f84c42249ce6` | 0.164326474 | 0.519633666 |
| `forgeEDA/mrc4.aig` / 611 | `894c573a03ba2fe416245f11cab49f26d28212bb2881ee4166ad0c5be044cb3b` | 0.164231263 | 0.518854996 |

两例三个方法各自的输入hash一致。A/Ours均为PASS：mrc3冷E2E分别1.874341234/21.274892242s，Ours来源`mffc_global_gradient`；mrc4分别1.876943022/17.267691858s，Ours来源`replay_noop`，不计为梯度质量收益。B缺失不构成Ours相对B的性能胜出证据。

证据相对正式目录：`output/records/0610_{A,B,Ours}.json`、`0611_{A,B,Ours}.json`；`output/case_0610_B/work/native_rewrite.{command.json,log}`及`case_0611_B`对应文件；两例外层`cold_method.{command.json,log}`保留失败传播。旧版同名输入曾出现该断言，仍不能据相同位置宣称内部根因已经完全定位。

检查时65245实际RUNNING并继续后续输入，两个后处理仍PENDING。至此已观察到四项方法级失败：392/B、562/B、610/B、611/B；这仍是运行中的已知清单，不是全905最终成功率。没有重跑失败点、替换工具或修改参数；最终保留全部失败wall，涉及B的成功质量配对排除缺失测量，Ours内部请求失败另行统计。

## index104：BRAM_SSP的两个内部请求共享CUDA分配失败

全量审计确认`forgeEDA/BRAM_SSP.aig`的Ours最终PASS，但`optimization/eval_0001`和`eval_0002`均FAIL、metrics为空。两者共享`optimization/_batch_0001/rewrite.command.json`同一次gpuls调用：返回1，3.269543231930584s；原始`rewrite.log`明确为`aig_manager.cu:29`的`cudaDeviceSetLimit(cudaLimitMallocHeapSize, (size_t)2048 * 1024 * 1024)`报告out of memory。两个配置分别请求`(26397,3,2)`及`(8319,3,2)+(26397,3,2)`。

两份`observation.json`的inclusive时间为3.272750240052119/3.272956053027883s。它们包含同一批次调用，合计6.545706293080002s只作请求inclusive统计，不等于两次独立失败调用或独占wall。没有成功安装、CEC或mapping标签；失败部分接受计数为0、没有repair。更深分配原因与显存峰值未核实，不能用Slurm CPU MaxRSS代替显存证据。

最终选择已付费且成功的replay no-op：4243.653320ps、area22461.130900、CEC PASS；整个Ours冷E2E为25.109973201993853s。相同输入A为4243.653320ps/area22462.531200，B为4268.202150ps/area22584.070300。回退使方法终态成功，但不能声称这两个梯度组合已验证或取得质量收益。没有重跑或更换工具来填补失败。

证据：正式目录的`output/records/0104_Ours.json`、`output/case_0104_Ours/child_result.json`与`work/optimization/{eval_0001,eval_0002}/observation.json`、`rewrite/replay.json`，共享`work/optimization/_batch_0001/rewrite.{command.json,log}`。全量3039个完整安装PASS、2 FAIL、0 UNKNOWN；2134个非空完整组合，2个失败都在该共享调用，不能把最终905 PASS解释为请求零失败。
