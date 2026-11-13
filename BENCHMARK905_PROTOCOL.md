# MFFC GPU 全 905 三方法评估协议

状态：**开发检查完成、最终入口预检中的草案，尚未开展正式2715点实验**。机器可读规则为 `config/mffc905_protocol.json`；正式冻结时记录Git源码/配置、native CUDA与CULS/ABC二进制、Student/ASAP7、完整输入manifest和运行命令的身份。不得把本页当执行完成证明。

## 输入、方法与正确性

使用既有完整905清单，输入SHA256唯一，来源ACE40/OpenABC47/forgeEDA805/iwls2022 5/iwls2024 8；不按旧收益、速度或候选数排除。全部输入各运行A/B/Ours，共2715个终态方法点。无候选、FAIL、TIMEOUT保留，不补造指标。

- A：原始AIG直接走统一ABC mapping/timing，不能替换为replay no-op。
- B：`read_aiger original; strash; rewrite; write_aiger candidate`，仅单次原生rewrite，再由相同Evaluator重新读入、mapping/timing。
- Ours：strict MFFC/Region DAG、候选Boolean证明、冻结LiteCut、候选原图哈希复用约束、全局可微代理方向与合法联合块更新、真实顺序rewrite及反馈。只以此前成功观察校准已有12参数，不训练Student。

各成功替换组合必须精确预选并完整ACCEPTED，CEC明确等价，再 `read_lib; read_aiger; map; topo; stime -p; write_verilog`。使用同一ABC/ASAP7和同一SC_Man的全部PO/global/area。旧MIO/pair/mode0指标不进入本轮。

## GPU职责与固定搜索规则

Candidate相对特征按Decision和MFFC在GPU归约；一个warp处理一个就绪MFFC，设备队列与依赖计数保留完整Region DAG前向和逆向依赖。所有PO/area的同一代理目标对候选one-hot权重求一阶梯度，按负方向导数构造跨Region合法联合块，再用完整GPU代理评分。不是多轨迹并行或CPU冻结adjoint。静态/候选复用预检不能保证动态增益，原生安装/CEC仍兜底。

固定seed20261114（当前方向法无需随机采样）；最多2轮、每轮最多2提案、每轮最多16个Decision编辑。replay总预算4，含no-op、失败和repair；另独立付费测original，普通mapping最多5次。真实反馈后校准12步。D<32且R<1024时CPU回退，记录原因；无Decision时不加载Student模型/代理。没有额外random/greedy或旧版对照。

真实面积限制为`A <= 1.05 × replay_noop_area + 1e-8`。Ours可返回自己已付费测量且面积可行的original，标注`original_fallback`；original/no-op不同网表不作为同一zero-choice拟合标签。A/B真实面积越界仍保留测量，不替换。

## 冷启动时间、硬件与超时

正式运行拟固定Compute/eda、RTX4090、4CPU/64GiB/1GPU；提交前重新检查实际partition与本用户RUNNING/PENDING资源，合计≤64CPU/256GiB/2GPU。同一作业内逐方法独立冷进程、按输入轮换A/B/Ours顺序；恢复也须保持同一硬件。Torch/OMP/OpenBLAS/MKL各1线程，真实验证最多2 workers。CPU参考仅用于实现正确性，不另设905 CPU组。

主runtime从各方法第一次读取输入到最终文件/必要CEC/真实timing可用，计入一次独立input staging、解释器/导入、所需候选/模型/准备、传输、GPU同步、选择/校准、全部失败/修复和在线工具。baseline不加载不用的ML。一次性build/源码归档、排队、分析不计单方法时间。页缓存不清空、CUDA缓存独立；GPU event时间只算内核阶段，不替代冷wall。工具上限900s、每冷方法1800s；失败/超时有实际耗时，无虚构delay。

## 统计与交付口径

Delay胜平负容差为`max(2e-5ps, 2e-6 × max(abs(D1),abs(D2)))`；runtime容差为`max(0.01s, 0.01 × max(T1,T2))`。改善为`100*(D_baseline-D_ours)/D_baseline`，速度比为`T_baseline/T_ours`。零分母/缺失/失败均N/A，几何均值仅使用正的已定义比值并报告分母，不能把0/0填1。

分别对A/B报告全部905的尝试/成功/失败/timeout/无候选、有效配对、胜平负、改善中位数、delay ratio几何均值、runtime中位/P90/总记录时间/逐对速度比、面积与可行性。来源及AND分组边界预设为0/128/4096/8192/16384/65536；描述MFFC数、DAG层/宽和实际设备。A/B/Ours共同成功且面积可行集合另列delay×runtime完整关系表，不事后挑有利分母。

保留逐请求/安装/mapping/CEC数、final来源、失败修复与主要inclusive阶段成本；重叠阶段不相加当wall，整批elapsed单列。输出`benchmark905_manifest.json`、`BENCHMARK905_RESULTS.csv`、`BENCHMARK905_SUMMARY.json`、`BENCHMARK905_RESULTS.md`及Delay/Runtime相对A/B、按规模的质量—时间图。最终报告首段一页内给结论，包含负结果及限制。
