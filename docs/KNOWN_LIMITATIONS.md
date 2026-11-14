# 结论边界与已知问题

本页主要记录初始历史导入的解释边界；后续MFFC GPU完整905的当前状态见[正式结果](../BENCHMARK905_RESULTS.md)和[梯度闭环报告](../GRADIENT_OPTIMIZATION_STATUS.md)。现有结果的主要风险是指标、输入快照和节点身份混用。整理保留原始文件，通过本文件记录解释更正，不回写历史日志和报告。

| 项目 | 当前解释 |
|---|---|
| Gate1输入 | 502 AND的`c1355.aig`，SHA256 `88df0a75b9328378b869843ba675c316d05a8bd812b7de22effc12af51f430ea` |
| Gate3输入 | 512 AND的`c1355_orig.aig`，SHA256 `2602b10f131a6ba5996521036ba130aab128a4aa15e3ab49ce44f2f02047f0b0` |
| LiteCut checkpoint | SHA256 `b3f499d2973bdb36fb42690794e9838d4c8359f05d6e141aa8cbe2b9fe3ae64d`；Student推理已恢复，精确历史训练数据/训练重建未恢复 |
| `predicted_local_abc_delay` | 训练流水线的局部cone标签预测代理，不能作为可加的MFFC delay或准确全局贡献 |
| `print_delay po11` | 固定MIO pin-delay模型的all-PI PO arrival；日志保留0.1 ps |
| `stime` | Liberty load/slew timing，global摘要保留0.01 ps；输出格式不等于内部量化精度 |
| `print_pair_delay` | corrected source-restricted MIO端点对最大delay；不是原Gate3B节点序列的固定path delay |
| R9 local arrival | 初始delay-DP mode0 phase reconciliation后的region-root正相位all-PI arrival |

Gate3F 的 `timing_precision_ps=0.01` 不适用于 `print_delay` 文本列。GG/LG/LL 的PO11 MIO回溯起点是pi22，GL是pi04；现有Gate3F `stime`日志没有建立全局critical path switching。候选局部supergate对齐被保守标记为NOT_ALIGNABLE；runner直接写入该分类，并非自动逐项对齐证明。

Gate3C summary 的 `artifact_sha256` 中自身文件的哈希与最终文件不同：脚本先计算哈希，再写入哈希表改变了文件。其余列出的8项artifact哈希匹配。导入清单保存当前真实字节的SHA256，不修补旧摘要。旧审计报告里的源码行号也可能因后续插桩变化，应按函数名定位。

旧15/67候选的`single_candidate_calibration.csv`及衍生相关性、校准、path-dependence分析保留为调试历史，不能继续引用为严格PI/PO pair实验结论。有效替代数据见`source_restricted_*`文件及 [标签更正说明](../evidence/path_delay_model/LABEL_CORRECTION_NOTE.md)。R9 2026-10-10停止的报告已被2026-10-12成功复现结论更新，不能把两者当作同时有效的最终结论。

导入保留历史源码绝对路径，没有迁移运行入口、重建软件环境或验证构建。冻结ABC二进制的精确source/build链没有完整记录；`third_party/abc`是审查所用已知基线，不宣称可以逐字节重建该二进制。Teacher/DeepGate训练依赖、完整训练数据及模型不属于已恢复的Student推理闭环。

Git沿用ABC上游`.gitattributes`：三个Windows IDE文件以LF存储、CRLF检出，两种哈希见 [git_normalization.json](../provenance/git_normalization.json)。`git archive`也会按上游`export-subst`展开ABC的`.gitcommit`占位符；ABC原始upstream commit以`source_repositories.json`为准。

`history/links`是对原有大运行目录的符号链接，不是独立备份。`history/imports`和`assets`是逐文件校验过的实际副本。初始整理仅复制固定c1355输入；随后按用户要求，将完整数据集迁至`/home/wushangge/CULS_LiteCut/open-sourceDataset`，旧位置保留兼容链接。后续统一通过`config/workspace.json`选择该数据集中的输入，准备工具将选定设计复制到新`/data`运行目录并记录哈希。历史输入副本与实验记录保持原样。
