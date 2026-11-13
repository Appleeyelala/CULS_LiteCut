# MFFC 级 GPU 改造与全 905 评估方案

依据：[本轮 Goal](docs/research/CODEX_GOAL_MFFC并行改造与905电路评估.md)。工作分支 `work/mffc-gpu-905`，历史432/48方法点、旧905结果与冻结tag保持原样。本文是实施前方案，不是实现或测试通过报告；后续以真实验证和冻结协议为准。

## 改进依据与计算组织

当前新版主要并行32条轨迹，DAG adjoint仍在CPU。旧组件hyp有16269个Region层，CPU adjoint约2.445s/完整CUDA提案2.564s。候选还会因顺序安装/结构复用产生动态失败。新实现直接按单电路的Region、Decision和Candidate组织计算：

| 阶段 | CPU职责 | 新GPU职责 |
|---|---|---|
| 输入与结构 | strict maximal MFFC、owner、Region DAG及CSR计划；保留原始ID空间 | 一次传入不可变拓扑/归属数据；不把region视为独立 |
| 局部信息 | 沿用冻结Student的19维特征和标签变换；不重训 | Candidate特征选择/聚合按Decision和owner执行张量归约 |
| 全局目标 | 准备PI/PO位置及真实基线标签 | 按Region就绪依赖执行前向LSE传播；反向就绪队列累积所有PO敏感度 |
| 联合更新 | 候选可执行性索引和受控离散提交 | 对所有Decision的Candidate选择求同一个全局目标梯度，批量排序/评分联合改动 |
| 真实执行 | 原生顺序replay、严格全ACCEPTED、CEC、统一ABC测量 | 复用CULS既有GPU候选/重写能力，归属与新增传播内核分开说明 |

计划以warp线程组处理一个就绪MFFC，多个线程组并行处理不同MFFC；大fan-in在组内归约。前向计数所有非PI依赖，反向计数后继依赖，使用设备队列与明确的原子内存顺序。深窄图避免每层由Python发射/同步CUDA kernel。Candidate→Decision→Region聚合和相邻region交互也在GPU，不只并行多条轨迹。

保留旧CPU传播作数值参照，小任务可以按固定元数据条件回退CPU，记录原因、规模、实际设备、前反向调用和内存。不在主实验增加一个905电路CPU方法组。不直接移植第三方二进制内核或CCS标签。

## 梯度选择与可执行性

变量仍为每个固定 `(root,cut_slot)` 的Candidate/no-op选择；一个MFFC允许多个兼容Decision。复用现有4维局部相对特征、port incidence、相邻region耦合和全PO平滑目标，面积约束下最小化delay。Student结果为常量，不反传经过Student、replay或ABC。

初步选择实现采用全代理的GPU梯度和受限联合块更新：一次计算所有候选的全局方向，构造不同编辑预算的合法组合，再用完整GPU DAG目标批量评分。无需32条搜索轨迹充当主要并行，也不另跑随机搜索。真实成功观察可校准现有12个共享参数；不建立新的大型精确时序模型。

静态冲突之外，检查库候选在原AIG结构哈希中的复用依赖：候选使用的原节点若被另一Decision移除，提交前保守限制该组合。复用依赖、候选级限制与排除原因显式保存，不把它宣称成完整动态可执行性预测。实际顺序saved/added/leaf/cycle检查和CEC继续兜底；拒绝部分安装结果。

开发初值为最多两轮、每轮最多两组合，总真实replay预算4；修复也是新请求并占预算，不能把失败部分电路直接送mapping。最终预算、校准步数、编辑上限、CPU回退条件与timeout在正确性验证后统一冻结。不会按905测试结果逐设计挑参数。

原图与内部replay no-op都由Ours自己真实测量、计入成本；全局代理基线与面积阈值沿用 `1.05 × replay no-op area`。若原图在该阈值内且优于搜索best，可作为明确记录的有成本最终回退，不把它当梯度收益，也不将两个不同的zero-choice网表混作同一校准标签。不会免费注入外部基线B的结果。

## 公开代码参考与边界

已查看固定版本的公开文件、README和可找到的license，文件hash及服务器路径见 `provenance/mffc_gpu_references.json`：

| 项目 / commit | 实际参考 | 采用边界 |
|---|---|---|
| [CULS](https://github.com/cuhk-eda/CULS) `5d62b64563b29354151638b67e510eeaa9cf88ed` | MFFC局部任务与原生顺序安装；BSD-3-Clause LICENSE可读 | 继续复用本地研究fork，不把cut-limited discovery当strict partition，不更换固定ABC |
| [INSTA](https://github.com/NVlabs/INSTA) `a91b7edb3c9b96717e19be96357673848775321f` | `graph/levelization.py`、`timing/propagation.py`和主接口 | NVIDIA Source Code License；只借鉴正反依赖/可微接口，独立实现本项目LSE内核，不依赖未公开的预编译内核 |
| [GCS-Timer](https://github.com/cuhk-eda/GCS-Timer) `77170ab831777fe7a620e3cf83eb6cfd0c1a5c6b` | `gpu_timer.hpp`的任务/索引组织与GPU归约 | README声明BSD-3-Clause，固定树未发现根目录独立license；不复制内核或引入CCS模型 |

公开参考仅小型静态阅读副本保存在 `/data/wushangge/CULS_LiteCut/references/mffc-gpu-20261113T173934Z-8742151d/`，未执行其安装/训练/实验。

## 正确性与冻结门槛

1. 原生GPU传播对照CPU：PI/constant入口、重复依赖、重汇合、深链、宽层、不同batch、极值稳定性，前向和一阶梯度/有限差分；检查队列完成计数、错误和内存。
2. 库相位/置换、原图哈希复用及候选级冲突的人工fixture与真实候选核对；小范围真实顺序安装、CEC和固定Liberty结果。保留失败，不用fixture标签当QoR。
3. 验证计时同步、真实设备与CPU回退、baseline A原图、B单次ABC rewrite、原图/no-op差异、失败/timeout与请求计数。
4. 通过后冻结源码/内核/工具/配置/seed/905输入SHA和协议。既有ABC与ASAP7版本保持不变，CUDA构建、测试和实验只在/data通过Slurm执行。

## 正式 905 评估与交付

仅A/B/Ours，全部905输入，共2715方法点；不加random、greedy、旧版或native CULS对照。优化seed预定20261114，一个种子；方法顺序按输入轮换，冷进程、独立缓存，统一主机/设备与线程配置。先检查集群与本用户RUNNING/PENDING，总资源≤64CPU/256GiB/2GPU，不自动取消已有作业。

沿用 `20261111T105821Z-benchmark-inventory-114a3573/output/benchmark_manifest.json` 的完整905输入身份，不按旧收益/速度/候选筛选。Ours仍在自己的wall内重新准备结构/特征；manifest可引用已存输入结构元数据，不能免费重用旧预测或timing标签。无候选、失败、timeout全部保留。基线B固定单次原生rewrite和已记录的读取/序列化命令。

逐方法保存冷wall、真实delay/area、CEC、选择与来源、请求/安装/mapping次数、MFFC/DAG形状、GPU任务与阶段时间；定义零分母不可用，胜平负容差/面积可行性/统计分母评估前冻结。断点续跑只补无终态点，完成点不挑最佳重跑；不同硬件不拼成纯GPU加速。

最终交付根目录同名说明与/data完整产物索引：`BENCHMARK905_PROTOCOL.md`、`benchmark905_manifest.json`、`BENCHMARK905_RESULTS.csv`、`BENCHMARK905_SUMMARY.json`、`BENCHMARK905_RESULTS.md`以及Delay/Runtime相对A/B和规模质量—时间图。报告先用不超过一页给出真实核心结论；负结果允许交付，但本轮不得以开发小样本代替全905终态与全部指标。
