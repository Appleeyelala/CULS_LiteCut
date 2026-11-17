# 代码职责与数据流

本仓库保留原始实现语义，以独立子目录汇总来源。导入版本对应原工作区当前文件，包含此前未提交的研究修改；原仓库 HEAD 本身不足以描述这些内容。逐文件关系见 [导入清单](../provenance/import_manifest.jsonl)，仓库状态见 [来源记录](../provenance/source_repositories.json)。

| 功能 | 本地源码 |
|---|---|
| AIG 读取、命令入口、GPU/CPU 状态 | `projects/culs/src/aig_manager.cu`、`src/command_manager.cpp` |
| MFFC cut 扩展与 PO 到 PI discovery | `projects/culs/src/aig/mffc.cuh`、`src/algorithms/refactor_mffc.cu` |
| GPU cut 枚举、预评估、replacement DAG、顺序安装 | `projects/culs/src/algorithms/rewrite.cu`、`rewrite.h`、`rewrite_library.inc` |
| Gate3 审计与 replay 接口 | `projects/culs/src/algorithms/gate3a.h`、`gate3c.h`、`gate3d.h`、`gate3e.h` |
| strict partition 和 Region DAG | `projects/culs/tools/gate3a_analyze.py`、`gate3b_analyze.py` |
| candidate 归属、执行审查 | `projects/culs/tools/gate3c_analyze.py`、`gate3d_analyze.py` |
| LiteCut 与固定配置 replay、CEC | `projects/culs/tools/gate3e_run.py` |
| 固定配置 ABC mapping/timing | `projects/culs/tools/gate3f_run.py` |
| mapping-free candidate 物化 | `projects/litecut/gate2a_candidate_inference.py` |
| 特征及 Student | `projects/litecut/src/data_preprocess/feat_extraction_v3.py`、`src/extract_cutlib_student/modeling.py` |
| checkpoint 加载、归一化 | `projects/litecut/src/validate/eval_prediction_accuracy.py` |
| 局部 abc_delay 标签生成 | `projects/litecut/src/subutils/Parser.py` 的 `stimeDelayProcess` |
| 后续 timing/mapper/R9 研究 | `projects/culs/experiments/path_delay_model/` |
| 历史 Gate1 多 decision replay | `legacy/gate1_aigrewrite/src/algorithms/rewrite.cu` 及 `gate1*.slurm` |
| ABC 干净基线与历史修改 | `third_party/abc/`、`history/abc_variants/` |

真实处理顺序如下：

1. 原始 AIG 通过 Gate3A 生成 CULS discovery cone；CPU unlimited ref/deref 参照计算全部 standard MFFC，再选择 inclusion-maximal 集合，建立 strict owner map。K=12 是 discovery 的限制，不是 strict MFFC 的定义。
2. Gate3B 逐原图边建立 quotient Region DAG，用 `owner[node]` 压缩结构路径；每个 PO 保存一条最长节点路径。它没有建立 mapped timing path。
3. GPU 枚举保留的 3/4-leaf cuts，使用 rewrite library 评估。Gate3C 导出满足本次正增益条件的候选身份和 cut-bounded removable nodes，再用 owner 归属 region。输出恢复到原 AIG ID。
4. 正常 rewrite 为每个 root 预选一个最优 cut/candidate，GPU 构建 DAG，CPU 顺序检查 cut leaf 并重算实际 saved/added。预选、executable 和 ACCEPTED 是不同状态。
5. Gate3E 从原 AIG 获取 context，按 phase/permutation 物化候选 DAG，提取25列但仅给 Student 前19维，使用 checkpoint 统计量标准化。输出经逆标准化、log clamp 和 expm1 还原为局部 abc_delay 预测。
6. 指定配置每次从原始 AIG 开始，只启用指定 decision，复用顺序安装与回退逻辑，输出完整 AIG 并 CEC。
7. Gate3F mapping 后分别读取 `stime` global delay 和 `print_delay po11` 的 MIO arrival。

| 概念 | 精确含义 |
|---|---|
| MFFC Region | strict maximal standard MFFC，用于分组、覆盖和结构路径表示 |
| Rewrite Decision | 固定 `(root, cut_slot)`，同一 region 可包含多个 decision |
| Rewrite Region | 此 decision 的 cut-bounded removable cone |
| Replacement Candidate | 此 decision 的 `subgraph_index` |
| LiteCut Score | 训练流水线局部 mapped-supergate cone 标签的预测代理 |

PO11 例子的 strict region path 是 R6 → R9 → R24 → R36 → R37 → R38。R9 的 region root=213，但被选择的 decision root=197；R24 的 region root=388，但 decision root=386。两者 cut_slot=2，G=subgraph0，L=subgraph2。

后续 ABC 副本中没有 `.git` 元数据。整理时将其实际源码与 `third_party/abc` 对比，保存完整的修改文件而非猜测补丁叠加顺序。每个变体的 changed/extra/missing 文件清单在 [abc_variants.json](../provenance/abc_variants.json)。在未来 `/data` 工作区中先复制基线，再覆盖选定变体，才是对应源码；不能把多个变体叠加。干净基线与 Gate3F 冻结二进制具有日期和源码关联证据，但历史二进制的精确构建链未完全记录。

## Joint-gradient v3 研究入口

以下是后续研究分支实现，不替代上方历史处理顺序或生产STE入口。

| 文件 | 职责 |
|---|---|
| `global_timing/phase_ports.py` | literal正/反相通道，候选边界相位组合 |
| `global_timing/conditional_arrival.py` | 条件端口DAG的分层GPU前向和一阶反向 |
| `global_timing/learned_ports.py` | 冻结LiteCut特征、共享候选端口网络和全局残差 |
| `scripts/joint_gradient_learning_data.py` | 家族隔离、冻结合法组合、受控测量及历史标签复用 |
| `scripts/train_joint_gradient_ports.py` | 离线PO/critical delta与排序学习 |
| `scripts/evaluate_joint_gradient_v3.py` | 不读取候选真实标签的固定预算搜索对照 |
| `scripts/audit_replay_noop_v3.py` | 原图与serialized no-op结构指纹审计 |
| `analysis/joint_gradient_v3/` | 实现、原始记录索引、失败质量门槛和后续决定 |
