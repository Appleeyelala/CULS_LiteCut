# STE100只读根因诊断

先读[未提交Rewrite的根因](NO_REWRITE_ROOT_CAUSES.md)，再查[33例逐项表](NO_REWRITE_CASES.csv)。第一阶段区分17个GPU导出域为空和16个有候选但未激活的情况；证据不足的解释明确保留。

- [67个不回退候选的真实质量](RAW_CANDIDATE_QOR.md)及[CSV](RAW_CANDIDATE_QOR.csv)。
- [Runtime记账口径](RUNTIME_ACCOUNTING.md)及[100电路CSV](RUNTIME_ACCOUNTING.csv)。
- [最小后续建议](RECOMMENDED_FIXES.md)：未实施、未调参。
- [83例轨迹概览](TRAJECTORY_OVERVIEW.csv)、[16例逐候选解析灵敏度](SAVED_PROXY_SENSITIVITY.csv)、[统计JSON](DIAGNOSTIC_STATS.json)、[原始证据路径与SHA256](EVIDENCE_INDEX.json)。

正式实验源码为`9a199bf0ea0dcf20bf55e40da8caa8b8bbff5fac`。原始实验文件和先前发布的结果保持历史语义。本目录来自已有文件的读取、统计和no-op处公式解析；没有新增实验、Mapping、推理或优化器运行。

`SAVED_PROXY_SENSITIVITY.csv`是基于保存特征和拓扑的独立解析值，不是当时保存的分候选梯度。读取脚本只导入冻结源码中的AIG解析器，不导入torch、不执行模型。它不能恢复当时未保存的logits、softmax概率或枚举拒绝细项。

需要复核表格时，在可访问原始run的服务器上，可对新空目录运行`python -B recompute_tables.py --output /data/wushangge/CULS_LiteCut/analysis/<new-directory>`。脚本拒绝覆盖已有输出；仅生成CSV和JSON，Markdown为基于这些数据撰写的诊断。大型数据仍留在服务器，公开目录只有派生小文件、脚本和路径/哈希索引。
