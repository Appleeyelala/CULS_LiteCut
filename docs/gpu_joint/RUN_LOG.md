# GPU 联合优化运行记录

原始 905 运行与状态报告保留；本页只记录本轮新增证据，路径均在 `/data/wushangge/CULS_LiteCut/runs/` 下。真实 Timing 统一使用冻结 ABC/ASAP7 的 `ABC_SCL_LIBERTY`，不引用撤回的 pair 标签。

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

## 冻结评价面板

[config/gpu_joint_panel.json](../../config/gpu_joint_panel.json) 固定 16 个输入与种子 20261111/12/13；生成器为 [freeze_joint_panel.py](../../scripts/freeze_joint_panel.py)。只使用冻结输入清单的 AND 规模、来源、文件名家族和 SHA，不查看候选数、delay 或运行时间。排除开发家族，四个区间各选 4 项：128–4095、4096–8191、8192–16383、16384–50000 AND。第一区间原拟分成 128–511 与 512–4095，但输入清单前者只有两个设计且一个属于开发集，因此在成功生成/冻结之前合并，仍未使用结果信息。

面板是已分析过的 905 数据中的冻结子集，不能称未见测试集；basename 规范化不证明所有功能相关设计都已去重。最终比较尚未执行，不能将固定清单视为已完成验证。
