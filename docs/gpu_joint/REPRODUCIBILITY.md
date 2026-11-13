# 联合梯度与 GPU 实验复现索引

本页说明已提交实验的输入、版本与入口，不启动实验。正式结果是否完整以原始作业终态和审计为准，不能仅凭本页或 `run.json` 中的命令判断通过。

## 固定来源

以下共同根目录为 `/data/wushangge/CULS_LiteCut/runs/`。

| 用途 | 已保存目录 / 身份 |
|---|---|
| 正式算法与协议 | `20261113T062048Z-frozen-joint-final-panel-b9c390b3/src/`；commit `22579982d22d741d47600a6406b09f00f4a4b46b`，本地tag `gpu-joint-evaluation-freeze-20261113` |
| 冻结旧算法 | `20261113T031523Z-gpu-joint-cold-baseline-934606e2/src/`；commit `5025a6c863f0a7766c41707bcc1828723afce447` |
| 新 gpuls | `20261113T050552Z-replay-gain-policy-probe-1d5e6f0a/build/framework_tools/culs/gpuls`；SHA256 `83d22440e2846054ea80e5c1cced97fefe8a25d51a499380d76225a9bb936776` |
| 旧 gpuls 与统一 ABC | `20261111T173355Z-acyclic-replay-tool-build-64cf764e/build/framework_tools/`；具体路径与hash见下文 |
| 905输入清单 | `20261111T105821Z-benchmark-inventory-114a3573/output/benchmark_manifest.json`；SHA256 `e4211308a06fc5b2bb02614b965c4b93dcf08f6f7ea2a73af2f2e1405da1e167` |
| 正式分析源码 | `20261113T070050Z-frozen-joint-final-synthesis-f895a375/src/`；commit `acdf98867e5afe628ba487c5361eb7d9aa36e6af` |
| 补充消融源码 | `20261113T073402Z-channel-repair-ablation-c9634975/src/`；commit `db64308d53dcdd37b075d35b0da08f1f5cf9f458` |

固定资产（不进入Git）：

- Student：`/data/wushangge/CULS_LiteCut/assets/models/litecut_student_best.pt`，SHA256 `b3f499d2973bdb36fb42690794e9838d4c8359f05d6e141aa8cbe2b9fe3ae64d`。
- ASAP7：`/data/wushangge/CULS_LiteCut/assets/libraries/asap7.lib`，SHA256 `819ecfc2609a1f53b9d298bf0cba75af0f1a8c737f33d4bbc5c9fb1156e8fa55`。
- ABC：旧工具目录下 `abc/abc`，SHA256 `5327dacd8df3b98caf6cea3ab4f01134413c0a8dc8dcda6a181a59e2c83f2bbb`。
- 旧gpuls：旧工具目录下 `culs/gpuls`，SHA256 `741990f3e8a9f270c0a9b3fa3a8889296f5c153749ad09bb91c83aeda59dda2e`。

新工具构建/默认与prepared回归的入口为 `scripts/build_replay_tools.py`；G的 `run.json`、`job.slurm`、`build/framework_tools/build_report.json` 和 `output/{replay_tests,prepared_regressions,batch_regressions}.json` 保存实际构建和验证。Release CULS及上游优化ABC的口径见 `docs/global_timing/PERFORMANCE_AUDIT.md`。旧二进制和新二进制分列，不承诺不同编译环境逐字节重建同一binary。

## 正式命令与参数

实际顶层argv在正式目录 `input/command.json`；driver为：

```text
python -B <正式src>/scripts/run_final_joint_panel.py
  --run <新的已准备run>
  --tool-run <G目录>
  --legacy-tool-run <旧工具目录>
  --legacy-source <冻结旧算法src>
```

driver从其自身 `src/config/gpu_joint_final_protocol.json` 和 `gpu_joint_panel.json` 读取16输入、三seed、九方法，检查每个输入SHA和实际child配置；不从当前工作树临时拼接算法。全部展开argv在 `input/final_panel_preflight.json`、`run.json:command`、日志和child记录中。

关键参数：每搜索8逻辑请求（含no-op/失败/修复），warmup参数2、batch4、校准30步、旧STE20步×2 restarts、新内层40步×32轨迹、编辑上限32、正增益安装、真实面积≤1.05no-op。主目标只用 `area_constraint`。固定seed为20261111/12/13。小图自适应保留CPU STE，不因方法名含auto就计CUDA收益。

工具评价为 `cec -n original candidate`，再 `read_lib; read_aiger; map; topo; stime -p; write_verilog`；`CULS_GLOBAL_TIMING_REPORT=1` 从同一个SC_Man输出global与全部PO rise/fall。所有正式标签为 `ABC_SCL_LIBERTY`，不使用旧pair标签或MIO arrival。

正式Slurm配置为Compute、2CPU、32GiB、1GPU、12h；不是必须在未来继续假定的集群配置。解释器 `/home/share/process/anaconda3/bin/python`，CUDA11.8；Torch1.13.1、选择/代理float64、Student原float32，Torch与OMP/OpenBLAS/MKL各1线程，CPU验证workers2。完整软件/节点环境在原 `run.json:environment`、`job.slurm` 及每例profile；系统页缓存没有清空。

## 在新的运行目录复现

1. 独立工作树检出需要的commit，保留当前研究分支和所有历史目录。`prepare_run.py`只归档其所在工作树的干净HEAD，不支持通过参数任意替换commit。
2. 在该干净工作树执行 `python -B scripts/prepare_run.py --name <新名称> --purpose <明确目的>`，获得全新 `/data`目录；它只准备源码、输入和资产引用，不执行实验。
3. 将上述完整argv写为新目录的JSON数组，使用该目录自身的 `src/`。不要复用原实验输出目录或修改其源码快照。
4. 按 `slurm-data-workflow` 检查当前partition和本用户RUNNING/PENDING总资源；用 `scripts/submit_checked_run.py` 记录资源、环境、argv并提交。总量≤64CPU/256GiB/2GPU。此处没有执行任何提交，也不建议把相同大面板无目的重跑。
5. 确认真实Slurm终态/退出码、精确432个终态方法点、完整安装与CEC/Liberty原始审计，才进行独立新目录的 `scripts/summarize_final_joint.py`。记录源码、输入/模型/库/工具hash；不要将旧算法实际import commit替换成外层profiler commit。

单例开发入口为 `scripts/profile_single_method.py --indices ... --variants ... --seed/--seeds ...`；实际可用名称以冻结脚本 `PROFILE_VARIANTS` 为准。补充消融用 `scripts/run_joint_ablations.py`，其配置和默认兼容验证单独记录，不与2257998正式结果混成一个算法版本。

## 读结果时的边界

每种搜索返回经真实验证的best，可能来自warmup、repair或no-op。代理loss、无量纲Q、Student局部分数和真实delay分别报告；不将训练残差下降当作泛化或QoR改善。固定30/120/600秒只使用已完成验证点；达到质量的条件中位时间须连同未达到分母一起读。不同seed不视为独立设计，16输入属于已分析905集合，不是未见集。

完整wall包含实际需要的初始化/准备/推理/校准/传输/失败/修复/工具调用/保存，并另计一次input staging；不含排队、一次性源码归档与工具构建。阶段inclusive时间会重叠。新增GPU贡献只统计实际CUDA、相同配置/工作量/完整请求序列的CPU配对，不能把原生CULS已有GPU能力归为新贡献。
