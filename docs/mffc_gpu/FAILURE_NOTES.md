# 正式905：失败与缺失测量记录

本页记录已观察到的失败，最终数量以全905终态与原始审计为准。冻结实验源为`ecc85e6`，Slurm65245，正式目录为`/data/wushangge/CULS_LiteCut/runs/20261113T191340Z-mffc-full905-frozen-6a45b8db`。不重跑失败点、不替换输入或工具、不用其他方法的delay填补缺项。

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

读取时65245仍RUNNING，驱动已经继续后续输入，两个后处理仍PENDING/Dependency。本次只记录新观察，不修改冻结算法、ABC或参数，不重跑失败点；最终B的有效质量配对排除此缺失测量，全部终态时间保留失败成本。当前观察到的方法失败为codec/B和loco_top/B，尚非全905最终计数。
