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

## 冻结评价面板

[config/gpu_joint_panel.json](../../config/gpu_joint_panel.json) 固定 16 个输入与种子 20261111/12/13；生成器为 [freeze_joint_panel.py](../../scripts/freeze_joint_panel.py)。只使用冻结输入清单的 AND 规模、来源、文件名家族和 SHA，不查看候选数、delay 或运行时间。排除开发家族，四个区间各选 4 项：128–4095、4096–8191、8192–16383、16384–50000 AND。第一区间原拟分成 128–511 与 512–4095，但输入清单前者只有两个设计且一个属于开发集，因此在成功生成/冻结之前合并，仍未使用结果信息。

面板是已分析过的 905 数据中的冻结子集，不能称未见测试集；basename 规范化不证明所有功能相关设计都已去重。最终比较尚未执行，不能将固定清单视为已完成验证。
