# AIG 层加权改写实验的四电路验收

**第二道验收在 c5315 的原图 root 460、cut slot 2 停止。** 三个原生 cut 叶子给候选 0 的预测层数为 6，原图上该 root 的实际层数为 7。独立路径枚举与冻结 ABC 均复现了差异。按批准的停机条件，本轮结束于候选结构核对。[失败记录](/data/wushangge/CULS_LiteCut/runs/20261121T053811Z-level-four-candidates-14732d0f/output/failure.json) · [独立复核](/data/wushangge/CULS_LiteCut/runs/20261121T054029Z-level-c5315-witness-2335d93b/output/witness.json)

## 两个研究问题的结果

| 研究问题 | 本轮结果 |
|---|---|
| 相同预算下，叶子加权能否比等权或一次 ABC rewrite 得到更低的全局深度？ | 待验证。五种评分及两种安装安排均未进入比较，ABC rewrite 对照也未运行。 |
| 全局深度实测下降的电路，映射后关键延迟是否也下降？ | 待验证。本轮没有安装改写或运行技术映射，因而没有可配对的深度和延迟变化。 |

## 原图深度与 strict MFFC roots

Python 在以 PO 为终点的 AIG 上传播 AND 层数，并从所有 PO 的要求层数 D 反推最紧要求。冻结 ABC 对同一输入文件执行 `read_aiger` 和 `print_stats`。不可达节点包括没有到达 PO 的常量、PI 或 AND。四个电路的 strict MFFC root 均可到达 PO。

| 电路 | 原图 AND | Python D | ABC D | strict roots | 零 slack roots | 深度等于 D 的 PO | 不可达节点 |
|---|---:|---:|---:|---:|---:|---:|---:|
| c1355 | 512 | 29 | 29 | 62 | 56 | 32 | 1 |
| c5315 | 1,613 | 38 | 38 | 264 | 8 | 2 | 1 |
| intra_pred_PE | 5,407 | 71 | 71 | 1,163 | 48 | 1 | 3 |
| cabac_ucontext | 37,490 | 77 | 77 | 5,249 | 220 | 186 | 1 |

各电路的 `roots.csv` 包含原 root、level、req 和 slack。`unreachable.csv` 保留节点身份，`level_check.json` 给出全部 slack 和 PO 层数的并列分布。[四电路输出](/data/wushangge/CULS_LiteCut/runs/20261121T053338Z-level-four-baselines-9506eb5c/output)

## 候选身份、叶子深度与 AND 增减

GPU 在放宽收益的 Gate3C 枚举中直接记录 `Eval` 的 nodesAdded，并核对 GPU 的 saved 与主机导出的 saved。保存的 AND 增减为 added 减 saved。新增明细和参考二进制在相同 AIG、相同策略上逐条核对原生行，随后按同一合法性过滤结果核对候选身份和数量。choice_index 0 表示原结构，其 subgraph_index 为空。原生 subgraph_index 0 仍是物理替换。[枚举实现](../../global_timing/level_candidates.py) · [工具哈希](/data/wushangge/CULS_LiteCut/runs/20261121T053811Z-level-four-candidates-14732d0f/output/native_tools.json)

| 电路 | 合法决策 | 合法物理候选 | 已写原结构行 | 已写物理候选行 | 候选 0 核对 |
|---|---:|---:|---:|---:|---|
| c1355 | 1,815 | 9,726 | 1,815 | 9,726 | 全部通过 |
| c5315 | 6,301 | 27,004 | 1,053 | 4,395 | 最后一条原结构行失败，表截止于此 |

c1355 的 root 151、cut slot 5、物理 subgraph 0 使用置换 `[2,3,0,1]`。有序叶子 `[21,34,145,146]` 对应的库输入顺序为 `[145,146,21,34]`。手算得到叶子 21 和 34 经库顶点 4 或 5、6、8 或 9、10，最长深度为 4。叶子 145 和 146 经 7、8 或 9、10，深度为 3。导出的深度向量为 `[4,4,3,3]`，原结构为 `[6,6,4,4]`。该物理候选预测 root 从 6 层降至 5 层，原生 added 为 7，saved 为 9。[完整候选表](/data/wushangge/CULS_LiteCut/runs/20261121T053811Z-level-four-candidates-14732d0f/output/c1355/pool/candidates.csv.gz) · [同池核对](/data/wushangge/CULS_LiteCut/runs/20261121T053811Z-level-four-candidates-14732d0f/output/c1355/pool/pool_check.json)

## c5315 的反例

原图 SHA-256 为 `d1c0cea55baf5a407495d82eb4385a9ddea8528cde21a762f46bb0774fc93afd`。新旧原生枚举对 root 460、cut slot 2 均给出相同的三个叶子、真值 3341 和物理子图编号 `[0,1]`。该决策通过 INTRA_REGION 和原有布尔合法性检查。下表中的深度全部来自候选 0 的原始 AIG。[原始决策和失败行](/data/wushangge/CULS_LiteCut/runs/20261121T053811Z-level-four-candidates-14732d0f/output/c5315/pool/noop_discrepancy.json) · [截至失败的明细](/data/wushangge/CULS_LiteCut/runs/20261121T053811Z-level-four-candidates-14732d0f/output/c5315/pool/candidates.csv.gz)

| 原叶子 | 叶子 level | 到 root 的最长 AND 深度 | 两者之和 |
|---:|---:|---:|---:|
| 83 | 0 | 4 | 4 |
| 446 | 2 | 4 | 6 |
| 450 | 3 | 2 | 5 |

按真实 cut 叶子取最大值得到 6。原始 AIG 另有一条 7 层路径经过 `153 → 439 → 440 → 442 → 443 → 457 → 458 → 460`，绕过这三个叶子。原生支持集最小化后，三叶真值已经足以表达 root 的布尔函数。把这三个叶子和另外四个 PI 作为独立变量，128 种赋值都与原生真值相同。函数验证因此可以通过，而叶子边界仍会漏掉决定原结构深度的一条路径。独立复核保留全部原图 AND 边，把 root 460 单独接到 PO，ABC 报告 7 层。[逐路径与真值复核](/data/wushangge/CULS_LiteCut/runs/20261121T054029Z-level-c5315-witness-2335d93b/output/witness.json) · [ABC 日志](/data/wushangge/CULS_LiteCut/runs/20261121T054029Z-level-c5315-witness-2335d93b/output/abc_levels.log)

本次验收要求用批准的叶子公式复原候选 0 的实际 root 层数。现场保留了这个决策及其失败的原结构行。逐决策评分、逐轮安装、最终映射三类表尚无数据。14 个开发电路和 24 个外部电路均未启动。本分支包含可选的原节点至序列化 literal 的回放导出，四电路实验停在实际回放之前。

## 可复现运行

基点为 `28523b9a`，分支为 `work/aig-level-weighted-rewrite`。原生枚举和 ABC 的命令、环境、输入哈希、Slurm 请求及输出索引随运行保存。本轮未加载 LiteCut 模型。[运行清单](runs.json) · [试验入口](../../scripts/aig_level_experiment.py) · [反例复核入口](../../scripts/audit_level_noop.py)

| Slurm ID | 源码快照 | 工作 | 请求 CPU / GiB / GPU | 墙钟时间 | 终态 |
|---:|---|---|---|---:|---|
| 66256 | `b43292df` | 四电路原图核对 | 2 / 8 / 0 | 4 s | COMPLETED |
| 66257 | `ed0b3658` | 编译原生 GPU 工具 | 8 / 32 / 0 | 49 s | COMPLETED |
| 66258 | `4105a0b2` | 同池候选与原结构验收 | 3 / 48 / 1 | 23 s | FAILED |
| 66261 | `ac0cbc53` | 已保存反例的独立复核 | 1 / 4 / 0 | 3 s | COMPLETED |
