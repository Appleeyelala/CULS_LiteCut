# GPU 联合优化运行记录

原始 905 运行与状态报告保留；本页只记录本轮新增证据，路径均在 `/data/wushangge/CULS_LiteCut/runs/` 下。真实 Timing 统一使用冻结 ABC/ASAP7 的 `ABC_SCL_LIBERTY`，不引用撤回的 pair 标签。

## 正式面板执行完成，独立后处理尚在进行

Slurm **64983** 已由实时accounting确认 **COMPLETED/0:0，Elapsed 1:42:27**，2CPU/32GiB/1GPU；batch MaxRSS为2083396K，不是GPU显存峰值。目录 `20261113T062048Z-frozen-joint-final-panel-b9c390b3`，快照2257998。精确16设计×3seed×9方法共 **432/432冷方法点PASS**，原始保存证据审计 **1119次成功测量、错误0**；方法进程PASS不代表其全部候选请求成功或QoR胜出。

实际完成文件 `output/final_panel_completion.json` 为 `EXECUTED_AND_AUDITED`；其记录的 `profile_summary.json` SHA256为 `f9b89532eb9d9e7521930bcfcf370ff3d11107f210691917cdfa6fceb7cb7f31`，协议SHA256仍为 `3999f37580ac0923b19d5c4766d26fec53b57ea46f19869461a7e303e325f326`。未修改算法、面板、种子、预算或评价门槛。

依赖后处理 **64984** 已开始；其10项统计/原始审计测试PASS，26开发方法点/108成功测量新来源审计PASS。正式1119测量的独立复核尚在运行，暂不报告完整统计或图表完成。65004仍等待该后处理终态，没有重复提交已有作业。

只读状态报告已单独提交为9b5b065，阅读截点15:57:12；其当时的RUNNING/PENDING状态保留为时间快照，不用于覆盖这里后续的实际终态。新增[复现索引](REPRODUCIBILITY.md)明确区分正式2257998、冻结旧5025a6c和补充控制db64308，源归档/工具构建与每电路完整wall分别说明。

## 已完成的诊断与旧方法基线

- `20261113T031521Z-gpu-joint-saved-diagnostics-63af3c1f`，Slurm 64920，源码 `5025a6c`，COMPLETED/0:0。12 个历史样例的只读耗时、重复 AIG 字节和 trace 体积诊断；不是新优化或 GPU 性能测量。
- `20261113T031523Z-gpu-joint-cold-baseline-934606e2`，Slurm 64921，源码 `5025a6c`，COMPLETED/0:0。三个旧梯度面积模式独立冷进程；DMA/c1355/ss_pcm wall 为 24.109/14.457/12.641 s，真实最终 delay 为 2380.250980/360.111877/323.908447 ps，全部最终 CEC PASS。完整子进程计时不含父进程输入复制和 Slurm 排队；各阶段嵌套计时不能相加。

## 单次 adjoint / 批量邻域实现验证

`20261113T033219Z-joint-criticality-validation-7014e76c`，Slurm **64922**，源码 **17ce395**，2 CPU、8 GiB、1 GPU，COMPLETED/0:0，Elapsed 6 s。`output/joint_tests.json` 为 **24 PASS/0 FAIL/0 SKIP**：8 项新增局部模型/邻域/CPU-CUDA 测试，4 项 pipeline 回归、9 项原代理回归、3 项 arrival 数值回归。`test_cpu_cuda_objective_gradient_and_final_selection_agree` 实际执行并 PASS；这些为 analytical fixture，不能作 QoR 改善证据。

## 真实闭环开发对照

`20261113T033348Z-joint-criticality-dev-0a56c7ea`，Slurm **64923**，源码 **17ce395**，4 CPU、16 GiB、1 GPU。已提交顺序冷进程对照：三个开发输入 × `legacy_cpu/criticality_cpu/criticality_cuda`，每个方法 8 请求，种子 20261111。旧方法明确从 5025a6c 的数据区快照导入；新 CPU/CUDA 使用相同 32 trajectories、40 步和 float64。每例轮换方法顺序，没有多 benchmark 吞吐加速声明。

本条提交时作业尚在运行，不能把初步已落盘的个别结果当作完整比较。终态以该 run 的 `run.json`、`output/profile_summary.json`、逐例 profile/choices/trace/CEC/timing 及 Slurm accounting 为准。新的数学机制见 [IMPLEMENTATION.md](IMPLEMENTATION.md)，评价规则见 [PLAN_AND_EVALUATION.md](PLAN_AND_EVALUATION.md)。

更新：64923 已 **COMPLETED/0:0，Elapsed 2m46s，9/9 PASS**。所有最终点 CEC PASS；以下 wall 为父进程记录的完整子进程时间，单次诊断不作稳定加速声明：

| 输入 | 旧 CPU：delay ps / wall s / 失败数 | 基础近似 CPU | 基础近似 CUDA |
|---|---|---|---|
| DMA | 2380.250980 / 22.307 / 4 | 2345.209470 / 32.689 / 1 | 2345.209470 / 29.580 / 1 |
| c1355 | 360.111877 / 13.014 / 1 | 360.111877 / 12.492 / 1 | 360.111877 / 14.650 / 1 |
| ss_pcm | 323.908447 / 12.692 / 0 | 332.451324 / 12.705 / 0 | 332.451324 / 13.561 / 0 |

新 CPU/CUDA 的最终离散选择和数值一致；DMA 的更少失败使更多请求继续做 CEC/mapping，因此较低 delay 不等于更低 runtime。ss_pcm 最终点来自 warmup，不能归因于梯度；CUDA 在小图上增加冷初始化成本。基础近似尚未达到总体目标。后续增加可关闭的八维成功反馈残差排序，是对这组失败证据的具体调整；尚不能预先声称有效。

## 残差排序验证与真实对照

`20261113T034339Z-joint-feedback-validation-cf6f9ad3`，Slurm **64928**，源码 **97d27f7**，COMPLETED/0:0，**26/26 PASS**；包括反馈只使用既往 PASS 标签、anchor 对齐、训练残差和反馈版 CPU/CUDA 选择一致性。fixture 的训练拟合下降不是验证集预测或 QoR 结果。

`20261113T034453Z-joint-feedback-dev-3776f84d`，Slurm **64929**，源码 **97d27f7**，COMPLETED/0:0，Elapsed 3m26s，**9/9 PASS**。最终点均完整安装并 CEC PASS：

| 输入 | 旧 CPU：delay ps / wall s / 失败数 | 反馈 CPU | 反馈 CUDA |
|---|---|---|---|
| DMA | 2380.250980 / 29.234 / 4 | 2301.566650 / 34.961 / 5 | 2301.566650 / 22.023 / 5 |
| c1355 | 360.111877 / 17.147 / 1 | 360.111877 / 17.005 / 1 | 360.111877 / 21.663 / 1 |
| ss_pcm | 323.908447 / 16.612 / 0 | 328.026855 / 22.067 / 0 | 328.026855 / 22.006 / 0 |

反馈版 CPU/CUDA 的最终选择和 QoR 相同。DMA 最终来源为 **explicit_feasibility_repair**，不是直接梯度完整成功；ss_pcm 的最终点为直接梯度，比基础近似更好、仍劣于旧方法。DMA 的单次 CUDA/CPU wall 差异不构成稳定端到端加速证据：作业前段同时存在本用户另一 GPU 组件作业，共享节点资源会影响 CPU/内存/IO；本轮只作为诊断，最终公平计时需单独控制。没有因此宣称已达到 QoR 非劣或 runtime 总体目标。

## 大图同工作量 CPU/CUDA 组件对照

`20261113T034455Z-joint-large-cpu-cuda-85b293a7`，Slurm **64930**，源码 **97d27f7**，4 CPU、32 GiB、1 GPU，COMPLETED/0:0，Elapsed 4m39s，MaxRSS 约 2.90 GiB。只从保存的原输入、verified catalogue、LiteCut 分数和校准系数构建代理，没有执行新的 rewrite、CEC 或 mapping。32 trajectories × 40 步，CPU/CUDA 均 float64，两个 seed 对应 20261111/12，轮换设备顺序；每次调用包含一次新 adjoint 和传输，模型构建/导入另列。

| 输入 | Decision 数 | repeat 0：CPU s / CUDA s / 比值 | repeat 1：CPU s / CUDA s / 比值 |
|---|---:|---|---|
| ACE/div | 51140 | 106.628 / 16.523 / 6.453 | 97.662 / 9.746 / 10.021 |
| ACE/hyp | 2760 | 4.682 / 3.566 / 1.313 | 4.596 / 2.752 / 1.670 |
| ACE/voter | 7007 | 5.505 / 1.204 / 4.572 | 5.085 / 1.178 / 4.318 |

6 个配对的最终候选向量及排序均一致。该证据支持新近似提案器内的单电路 GPU 并行计算收益，**不支持整个优化闭环也快 6–10 倍**；CUDA context 在作业内共享，首次调用/缓存成本与后续调用不同，少量重复不是严谨分布估计。hyp 的一次 adjoint 约 2.45–3.26 s，占调用较多，说明剩余 CPU 部分限制加速。原始 JSON 入口为 `output/kernel_summary.json` 及各 `kernel_case_NNNN.json`；每项保留实际代理输入文件哈希，校准数据不能用作新方法的在线免费历史标签。

## 稀疏传输/投影版本

`20261113T035458Z-joint-sparse-validation-d7e72b17`，Slurm **64931**，源码 **f209a70**，COMPLETED/0:0，**27/27 PASS**，含 sparse hard-value 与 dense one-hot 数值对照。

`20261113T035708Z-joint-sparse-cpu-cuda-a65c4ca7`，Slurm **64932**，源码 **f209a70**。RTX4090/eda，同工作量 32×40，三例各 1 个配对；`output/kernel_summary.json` 的 CPU/CUDA 顺序一致与旧版选择哈希一致两项均为 true。旧版和本版代理输入文件哈希严格相同：

| 输入 | CPU 调用 s | CUDA 调用 s | 比值 | CUDA adjoint / 内层 / CPU合法投影 s |
|---|---:|---:|---:|---|
| div | 96.932 | 3.513 | 27.589 | 0.969 / 0.596 / 0.452 |
| hyp | 4.362 | 2.564 | 1.701 | 2.445 / 0.086 / 0.010 |
| voter | 4.227 | 0.184 | 23.034 | 0.030 / 0.053 / 0.054 |

稀疏传回与投影索引复用降低了大变量数时的非 GPU 开销，保存的选择没有改变。div 的第一次 CUDA 初始化/传输仍占 1.443 s；hyp 受 CPU adjoint 限制。这里只是组件和一次配对，未建立对应端到端加速，更没有新 QoR 结论。64932 前段与原生构建共存，计时仍是诊断范围。

## 原生命令内枚举复用的构建与回归

`20261113T040333Z-prepared-replay-build-499f58ff`，Slurm **64933**，源码 **b30bdaa**。CMake Release 编译和链接完成，但回归入口失败：新测试夹具把类属性命名为 `run`，覆盖了 `unittest.TestCase.run`，导致 `TypeError: PosixPath object is not callable`，**测试没有实际执行，不能记为 PASS**。原始失败目录保留。

已在后续源码中改为 `run_root`。重新验证使用新 committed-source 目录；构建入口允许在精确比较全部 CULS 源文件哈希后复用已编译二进制，明确只复用编译，不改变原作业失败状态。原生命令能力的契约见 [PREPARED_REPLAY.md](PREPARED_REPLAY.md)，尚不能据编译成功宣称执行语义或加速已验证。

`20261113T040910Z-prepared-replay-validation-9244db37`，Slurm **64934**，源码 **6eed787**，失败保留：首组 **14 PASS/1 FAIL**，后续全量开启复用的回归因前组失败未执行。已通过的是 no-op/四组联合配置与当前默认路径、冻结旧二进制的全部 AIG/trace 精确一致，以及动态拒绝一致、12 项默认真实工具回归。失败项为负向夹具错误地期待 gpuls 必然非零退出；原 CLI 实际打印候选越界错误、没有成功 AIG/trace，却可能返回 0。后续修正检查明确错误与成功产物缺失，未修改原 dispatcher，也未把旧作业改成 PASS。复用编译前核对了 125 个 CULS 文件，源树 SHA256 `04d5f0135e5bd0c4d27fb49ecddc460b205d8bbfd338c5d9bea7d38c2eea20f4`，新 gpuls SHA256 `b7fb8732e3cc9a25a20176660e0ba48f365b21999f1acf1df41fb9e2a3fdcf5d`。

`20261113T041250Z-prepared-replay-negative-fix-3ce36193`，Slurm **64935**，源码 **85acd9f**，COMPLETED/0:0，Elapsed 1m11s。`output/replay_tests.json` **15/15 PASS**；随后开启复用的 `output/prepared_regressions.json` **12/12 PASS**，覆盖常量、重复实例、接口、CEC/Liberty、一致性与错误处理。新编译二进制仍为上述 b7fb8732…；只复用精确匹配的编译产物，两个旧失败作业保持 FAILED。

## 后续计时证据的格式

`profile_single_method.py` 新增独立 `case_NNNN_timeline.json`，记录真实请求完成时的 monotonic 时间、来源和已验证 delay/area。父进程记录子进程启动时间，供相同时间截点与 time-to-quality 统计使用；不从文件 mtime 推断，也不回写上述旧 run。旧诊断 run 没有这些结构化时间戳，只能引用其总 wall 和工具计时，不能补造 30/120/600 秒处曲线。

`20261113T041347Z-joint-timeline-check-8ad162dc`，Slurm **64936**，源码 **85acd9f**，COMPLETED/0:0，Elapsed 33s。真实 ss_pcm 反馈 CPU/CUDA 均 PASS、最终 328.026855 ps；cold wall 13.402/16.574 s。每模式保存 8 个真实请求完成事件，时间均在父进程的启动/退出区间内，timeline 文件哈希匹配。该检查验证了稀疏版的真实安装与时间轴，仍没有使 ss_pcm 优于旧方法，也不是 GPU 冷进程加速结果。

## 批量执行接口的集成验证

`20261113T043158Z-joint-batch-validation-b62c7dd9`，Slurm **64939**，源码 **43f81b8**，COMPLETED/0:0，Elapsed 11s，**7/7 PASS**。3 项真实 batch fixture 与 4 项既有 pipeline 回归。混合批次的动态拒绝明确 FAIL、不送 CEC；另两个完整成功配置各自实际 CEC/Liberty mapping，与单独 replay 的 AIG 字节一致。另验证静态非法输入在启动原生工具前拒绝、预算/索引/预测训练历史保持，以及固定配置夹具不被误标为梯度来源。这里尚未运行自由搜索的端到端 batch 对照。

后续冷进程比较将 CUDA cache 放到各 child 独立的 `/data/.../cache/cuda_<variant>_<index>`，记录 LiteCut cache hit、Torch 线程数和实际 GPU；不清空共享文件系统页缓存。输入校验/复制计时单列，并给每个独立部署方法计入一份 staging 成本。旧对照可显式同时指定冻结源码和冻结 tool-run；CPU/CUDA 新方法与 batch 工程控制使用相同的新工具。这些约束用于后续比较，不回写此前诊断的缓存条件或时间。

## 同分配、隔离缓存的自由搜索对照

`20261113T043907Z-joint-batch-dev-controls-73353bd5`，Slurm **64945**，源码 **6504546**，COMPLETED/0:0，Elapsed 3m02s，12/12 PASS。旧算法使用冻结 5025a6c 源码和原 741990f3… 工具，新方法统一使用 b7fb8732… 工具；同一分配 2 CPU、16 GiB、1 GPU，顺序冷进程、各方法 8 请求，所有 LiteCut cache_hit=false，CUDA cache 每个 child 独立。

| 输入 | 旧 CPU：delay ps / wall s | 反馈顺序 CPU | 反馈 batch CPU | 反馈 batch CUDA |
|---|---|---|---|---|
| DMA | 2380.250980 / 20.584 | 2301.566650 / 18.152 | 2301.566650 / 17.401 | 2301.566650 / 17.999 |
| c1355 | 360.111877 / 13.641 | 360.111877 / 11.803 | 360.111877 / 10.989 | 360.111877 / 11.994 |
| ss_pcm | 323.908447 / 12.292 | 328.026855 / 12.343 | 328.026855 / 18.623 | 328.026855 / 13.141 |

`20261113T045111Z-batch-control-analysis-99c88f36`，Slurm **64951**，源码 **e43a3c8**，COMPLETED/0:0。只读核对全部 12 个最终原始 AIG、trace 精确安装、CEC 和 Liberty delay/area/全 PO 日志；6 个工程配对的**逐请求向量、PASS/FAIL、全部成功真实指标完全一致**，timeline 完整且范围正确。报告/CSV/哈希入口为 `output/profile_analysis/{REPORT.md,results.csv,summary.json,input_hashes.json}`。

batch 将 DMA/c1355 的实际 replay 命令从 8 减至 5，ss_pcm 从 8 减至 7；配对的 CEC/mapping 次数不变，未省略验证。但单次端到端时间不稳定、ss_pcm batch CPU 还更慢，没有达到稳定的整体加速或更好梯度价值结论。DMA 四个直接提案仍全部 REVERTED，best 为修复；新增安装策略将按 [INSTALL_POLICY.md](INSTALL_POLICY.md) 单独控制检验，不与本表混为纯工程加速。

## 冻结评价面板

### 独立自然成本基线与回归

### 最终面板提交与原始审计完成

补充贡献控制已准备为 **`20261113T073402Z-channel-repair-ablation-c9634975`**，源码 **db64308**，Slurm **65004**，2 CPU/16GiB/1GPU、2h有限墙钟，依赖 **afterany:64984**。因此主正式计时与其后处理完成后才开始，不与64983/64984并发。提交时三个RUNNING+PENDING任务共6 CPU/56GiB/2GPU，仍在用户上限内；未取消、重启或改动两个既有作业。

用户研究说明要求区分LiteCut/显式交互/修复。按原冻结面板每个AND区间的第一行固定65/96/625/587，三个原seed、默认+no-score/no-pairs/no-repair，共48方法点；选择时主面板在运行，未查看指标来选择补充输入，且不把它冒称为完全预结果选择。协议在 `config/gpu_joint_ablation_protocol.json`，作用与边界见[CHANNEL_ABLATIONS.md](CHANNEL_ABLATIONS.md)。

该job先运行新开关与默认数值回归，再用新建 `20261113T073404Z-ablation-real-smoke-3c7df183` 和 `20261113T073406Z-ablation-default-reference-bd1bb4e0` 进行同分配c1355四请求的当前默认/冻结2257998真实对照，并验证三个开关；这些标签不进入研究反馈。兼容检查不通过就停止，不自动改写参考结果。随后才执行48点及逐点原始审计。当前仅静态语法/元数据规则检查，实际测试和贡献结果仍未发生。

后处理已独立准备为 **`20261113T070050Z-frozen-joint-final-synthesis-f895a375`**，源码 **acdf988**，Slurm **64984**，2 CPU/8GiB/无GPU，有限45m，依赖 **afterany:64983**。提交时确认64983仍RUNNING、64984为PENDING/Dependency；累计RUNNING+PENDING请求4 CPU/40GiB/1GPU，未干扰主实验计时。此任务只做统计fixture、开发证据回归、正式原始证据复核、表格/中文报告与PNG/PDF图；不运行EDA或模型，且不改写正式快照。

新分析工具区分外层运行commit与实际旧源码import commit，并核对原生ABC/CULS命令、工具与模型hash；严格记录相同工作量/请求的GPU配对，CPU分支不计GPU贡献。完整来源见[FINAL_ANALYSIS_CONTRACT.md](FINAL_ANALYSIS_CONTRACT.md)。测试和图表尚待任务实际执行，提交与语法检查不是PASS证据；若64983失败，afterany也不会伪造有效最终数据，而应停止并保留失败原因。逐项完成审查在[GOAL_DELIVERY_AUDIT.md](GOAL_DELIVERY_AUDIT.md)，当前仍未完成。

`20261113T062030Z-complete-developer-audit-2d3f409e`，Slurm **64982**，源码 **2257998**，2 CPU/4GiB/无GPU，COMPLETED/0:0，24s。上述64976/64978/64980合计26方法点、108成功测量均通过完整原始审计。`output/panel_analysis/comparisons.json` 保存div实际CUDA/CPU全请求一致性和同分配E2E比值；未将它计作更好的QoR或稳定总体速度分布。

最终运行 **`20261113T062048Z-frozen-joint-final-panel-b9c390b3`**，Slurm **64983**，已按前瞻协议提交，源码 **22579982d22d741d47600a6406b09f00f4a4b46b**，本地tag **gpu-joint-evaluation-freeze-20261113**。请求2 CPU、32GiB、1GPU、Compute、12h有限墙钟，提交时无其他本用户RUNNING/PENDING请求，总额仍在64 CPU/256GiB/2GPU内；任务为16输入×3种子×9方法共432冷child，顺序执行而非多benchmark吞吐。

此记录仅代表提交，不代表完成或已取得优势。权威状态需检查该job的squeue/sacct与run.json；精确命令/集群检查在 `input/{requested_command.json,command.json,submission.json}`、`job.slurm`。预检/最终覆盖/实际配置/逐点原始审计由 `scripts/run_final_joint_panel.py` 强制，成功后应生成 `output/final_panel_completion.json` 和 `output/panel_analysis/`。未生成或审计失败时不得以意图、部分目录或旧测试替代最终结果。

最终入口验证 `20261113T061111Z-final-entry-validation-3cdd96be`，Slurm **64981**，源码 **7087364**，2 CPU/12GiB/1GPU，COMPLETED/0:0，51s。`output/entry_tests.json` **36/36 PASS**，含新增3项no-inner更新的无Adam/backward与CPU/CUDA一致性检查；随后c1355的2seed×4方法8个冷进程全部PASS，22次成功测量完整原始审计。初始化控制的请求没有误标梯度，未生成不存在的梯度范数；同seed各方法数据/候选/库/ABC和面积参考一致；多seed输出及cache相互隔离，没有覆盖既有文件。

冻结最终协议为 `config/gpu_joint_final_protocol.json`，主面板sha不变，16输入×3seed×9方法共432个冷child：旧CPU、adaptive CPU/auto、no-inner控制、自然成本random/greedy/original/native ABC/native CULS。主方法默认正增益，8逻辑请求和原评价阈值不变，prospective协议不是已经执行的结果。新入口 `scripts/run_final_joint_panel.py` 提交前核对精确面板/清单/输入hash，运行后核对完整覆盖并逐点审计，显式保留失败；不因少数开发案例调整主面板或评价阈值。

后续审计 `20261113T060125Z-timing-curve-audit-validation-a1e40c0b`，Slurm **64979**，源码 **a6d7f84**，2 CPU/4GiB/无GPU，COMPLETED/0:0，7s。3/3 analytical反例与时间截点测试通过；原始审计覆盖64976/64978全部24个方法点、94次成功测量，错误0。所有成功点均核对AIG、精确请求全ACCEPTED、CEC、Liberty、mapped.v、资产/命令hash和完成事件。产物在 `output/panel_analysis/`，包括results.csv、audited_records.json、comparisons.json、evidence_hashes.json；没有新EDA或timing标签。

### 真正端到端的单电路CPU/CUDA对照

`20261113T060301Z-adaptive-large-e2e-pair-aa596854`，Slurm **64980**，源码 **a6d7f84**，2 CPU/32GiB/1 RTX4090，COMPLETED/0:0，8m21s，MaxRSS约2.65GiB。固定开发ACE/div（index20），每方法8请求，32×40/float64，CPU Torch线程1、验证workers2、同一新gpuls/ABC/ASAP7，顺序独立冷child及隔离CUDA/Student cache。

| 方法 | cold wall s | 最终真实delay ps | 实际area | 最终rewrite | 请求/失败 | 首个已验证no-op完成 s |
|---|---:|---:|---:|---:|---|---:|
| adaptive_batch_cpu | 327.205694 | 76414.984400 | 12459.951200 | 0 | 8/1 | 64.434139 |
| adaptive_batch_cuda | 169.601841 | 76414.984400 | 12459.951200 | 0 | 8/1 | 62.618616 |

两个最终点CEC PASS；八个请求逐项choices、PASS/FAIL、所有成功delay/area/全部PO精确相同。CPU提案累计164.492s，CUDA两个提案调用约3.696+2.888s，其余准备/校准/真实EDA仍付费。因此该次完整成本约1.929倍差异，不再只是kernel或代理组件计时；但仅单个开发设计、一次配对，没有稳定分布证据。

**两者均返回no-op，没有新的QoR改善。** 按预先质量阈值只需baseline，此时达标时间约64.43/62.62s，不能把完成全部搜索的1.929倍称为同倍更早找到改善点。原始证据在 `output/<variant>/case_0020/`、逐profile/timeline和父 `output/profile_summary.json`；完整raw逐点审计仍要单独落盘。该设计不在最终面板中。

`20261113T055037Z-standalone-controls-validation-97d74575`，Slurm **64977**，源码 **dc95c8c**，2 CPU/12GiB/1GPU，COMPLETED/0:0，53s，`output/controls_tests.json` **51/51 PASS**。包括6项新control、3项真实batch、4项pipeline、14项joint、9项surrogate、3项arrival、12项真实工具。random/native导入路径不加载未使用的Torch/NumPy/NetworkX；共享Boolean parser与原canonical完整返回结构一致；实际random/greedy/native各自CEC+Liberty验证。新共享执行模块由原已验证函数提取，不改变默认安装政策。

`20261113T055431Z-standalone-controls-dev-0e1ce7d6`，Slurm **64978**，同源码，2 CPU/16GiB/1GPU，COMPLETED/0:0，1m10s，15/15独立冷child PASS；seed20261111。

| 输入 | original delay ps / wall s | native ABC | native CULS | random batch（8请求） | LiteCut greedy |
|---|---|---|---|---|---|
| DMA | 2463.366460 / 2.519 | 2491.453120 / 2.869 | 2328.766110 / 3.170 | 2371.904300 / 13.391 | 2352.961910 / 10.538 |
| c1355 | 360.111877 / 1.366 | 359.501831 / 1.467 | 353.219391 / 1.668 | 360.111877 / 7.078 | 359.873505 / 5.374 |
| ss_pcm | 350.016113 / 1.316 | 319.565643 / 1.416 | 323.908447 / 1.618 | 319.488068 / 8.381 | 338.486664 / 4.723 |

各方法最终点CEC PASS。随机/greedy最终点按真实no-op面积1.05倍选择，原图/native的同约束可行性需对照同输入no-op分析。greedy实际3/3/2请求（含no-op及修复），不凑满8。所有random/native profile明确`torch_loaded=false`，greedy为true；没有不计时的免费ML或人为给简单基线增加unused fit。方法范围和自然成本见[BASELINE_PROTOCOL.md](BASELINE_PROTOCOL.md)。本表与64976不同分配，不能据此作严格同硬件E2E配对加速声明；保留作基线能力和后续统一计时的依据。

ss_pcm原生CULS恰好取得与旧/自适应梯度相同的323.908447ps，但自然成本显著较低；random/ABC的delay还更低。DMA新正增益梯度可找到更低delay，但绝不能据三个样例宣称全局竞争力或低runtime。最终冻结面板仍未运行。

补充实际搜索：`20261113T054320Z-adaptive-gain-search-dev-2896af18`，Slurm **64976**，源码 **7e5bbc1**，2 CPU、16 GiB、1 GPU，COMPLETED/0:0，2m20s，9/9冷进程闭环PASS。各方法8请求、seed20261111；旧5025a6c+741工具，新方法使用G的83d22440工具。只在开发0/47/80验证，没有使用最终面板调参。

| 输入 | 旧CPU delay ps / cold wall s | adaptive batch auto正增益 | adaptive physical batch auto |
|---|---|---|---|
| DMA | 2380.250980 / 22.161 | 2301.566650 / 18.222 | 2396.672360 / 27.917 |
| c1355 | 360.111877 / 11.810 | 360.111877 / 9.581 | 360.111877 / 14.042 |
| ss_pcm | 323.908447 / 11.693 | 323.908447 / 11.102 | 323.908447 / 11.035 |

均实际CEC PASS且面积可行。ss_pcm/c1355实际选择full_ste/CPU，DMA为frozen_criticality/CPU；没有CUDA内层贡献。正增益DMA best依然28rewrite显式修复、5失败；physical DMA则138rewrite直接点、0失败，但QoR更差。允许非正增益改变了早期warmup是否成功与后续校准/选择，不能用固定四向量probe的最佳点预言自由搜索胜出。保留该负结果，主方法仍选默认正增益；metadata自适应小图保留原STE有效避免已观察到的ss_pcm退化，三例单次计时仍不足证明稳定加速。

逐例证据在 `output/<variant>/case_NNNN/gradient_area_constraint/{summary.json,gradients_*.json,eval_*/}`，元数据/完整命令在 `run.json`、`job.slurm`、`output/profile_summary.json`；父进程完成时间和每请求timeline均保留。后续独立审计不能在缺失或失败时补造成功标签。

[config/gpu_joint_panel.json](../../config/gpu_joint_panel.json) 固定 16 个输入与种子 20261111/12/13；生成器为 [freeze_joint_panel.py](../../scripts/freeze_joint_panel.py)。只使用冻结输入清单的 AND 规模、来源、文件名家族和 SHA，不查看候选数、delay 或运行时间。排除开发家族，四个区间各选 4 项：128–4095、4096–8191、8192–16383、16384–50000 AND。第一区间原拟分成 128–511 与 512–4095，但输入清单前者只有两个设计且一个属于开发集，因此在成功生成/冻结之前合并，仍未使用结果信息。

面板是已分析过的 905 数据中的冻结子集，不能称未见测试集；basename 规范化不证明所有功能相关设计都已去重。最终比较尚未执行，不能将固定清单视为已完成验证。
