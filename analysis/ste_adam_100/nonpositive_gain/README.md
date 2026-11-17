# 非正节点 gain 候选实验

先读 [候选覆盖结论](ZERO_CANDIDATE_RESOLUTION.md)，再读 [100电路逐项计数](CANDIDATE_COUNTS_100.csv)、[结果汇总](RESULTS_SUMMARY.md) 和 [实现与证据](IMPLEMENTATION_AND_EVIDENCE.md)。

这是 `allow_nonpositive_candidates` 独立策略。原 STE100 正式源码 `9a199bf0ea0dcf20bf55e40da8caa8b8bbff5fac` 及历史905方向导数数据均未改写。本轮候选验证源码 `61ec4fcef316f20ff5fed1d8aadbe4932748a48b`，Slurm `65897`。独立 gpuls SHA256 `66b2aa811d759dde57aecb0847241fd7c735cfcb0e98da0dfcfd855dd4c60af6`。

固定100沿用 seed=20261116、原100 ID及逐输入SHA；清单和工具身份见 `RUN_PROVENANCE.json`。`REPLAY_PROBES.csv` 是少量实际安装/CEC检查，不是 Adam 搜索质量。真实原始证据位置及哈希见 `EVIDENCE_INDEX.json`。

原17例门槛：17/17。GO_905=true。全量新执行见 full905/；100子集结果来自该905运行，没有另跑100或A/B。

完整905实际运行源码：`bd87a176024d681140305d2aa21ba20defdba4bd`。完成分母、失败/超时、实际Slurm环境与资源见 `full905/FULL905_RUN_STATUS.json`；后续报告发布提交与实验运行提交分别记录。

部分已完成的新run LiteCut缓存已无损归档；读取这些缓存前请参照 [恢复说明](STORAGE_ARCHIVE.md) 和 [逐文件哈希索引](STORAGE_ARCHIVE_INDEX.json)。原始结果和计时未改写，归档期间的并行I/O在汇总中披露。
