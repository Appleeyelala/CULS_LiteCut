# Runtime重新记账：扣除第二次ABC验证命令

本次只计算既有日志，不执行性能优化。定义：

`optimization_runtime_s = measured_full_runtime_s - second_mapping_abc_time_s`

- `measured_full_runtime_s`直接取每电路独立冷进程原始record的 `end_to_end_runtime_s`。
- 67个非空候选的扣除值取 `work/final/eval_0000/evaluation/mapping.command.json.elapsed_s`；全部存在、退出码0且无timeout。
- 33个没有第二次Mapping的电路，扣除值为0；这表示该命令未执行，不是用0填补缺失计时。
- 只扣这一份命令时钟一次。原图Mapping、最终Replay、CEC、准备、LiteCut、GPU优化、传输/同步、必要输出及进程开销都保留。

真实时钟由 `global_timing/common.py:run_tool` 的 `time.monotonic()` 在进程启动前至进程退出后采集。这里的“第二次ABC Mapping耗时”是整个 `read_lib; read_aiger; map; topo; stime -p; write_verilog` ABC命令进程wall time，包含命令内加载/时序/网表输出；现有数据不支持把纯`map`内核与命令其余步骤精确拆分。没有使用CPU时间或日志估计替代wall time。

|指标（100电路）|中位数 s|总和 s|
|---|---:|---:|
|`measured_full_runtime_s`|13.175791104|1647.197537762|
|`second_mapping_abc_time_s`|1.315912822|164.036815200|
|`optimization_runtime_s`|12.003044787|1483.160722562|

只看确有第二次Mapping的67个电路，其验证命令耗时中位数为 **1.616543823 s**。全100的扣除中位数包含33个0。优化Runtime中位数来自逐电路扣除后的100个值，不是两个中位数相减。

|同一100电路逐对比较|完整Ours/完整基线Runtime比中位数|扣除验证ABC后的研究优化Runtime/完整基线比中位数|
|---|---:|---:|
|A|5.281315483|4.495928093|
|B|4.045126345|3.653274284|

A、B仍使用各自完整冷进程Runtime，没有调整。A/B完整Runtime中位数分别为2.172834/2.702031 s。新研究口径的Ours仍明显更慢。

逐电路三种时间、基线时间、第二次Mapping原始路径和调用次数见[RUNTIME_ACCOUNTING.csv](RUNTIME_ACCOUNTING.csv)，100行完整，缺失计时0项。总扣除164.036815秒约占Ours完整时间总和的9.96%；这只是当前正式100实验的时钟记账，不能替代详细瓶颈归因，也不套用历史905的阶段占比。

**这不是已经实测的一次Mapping端到端部署性能。** 完整程序实际执行过第二次ABC验证及后续回退/输出；此处保留其余实际开销，仅按研究规则扣除验证命令时间。未来真正删去第二次Mapping的程序还可能改变控制流、缓存、输出与选择逻辑，需要另行实测才能作部署性能声明。现阶段只报告这份明确标注的研究记账数据。

第一优先级根因解决前，不进行并行化、缓存、GPU/CPU切换或新耗时实验。尚待回答：哪类准备工作占据主导、一次Mapping程序实际运行时间如何变化；当前记录可作下一步定位依据，但本次不进行额外性能改造。
