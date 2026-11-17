# Joint-gradient v3

已实现并完成受控验证；质量门槛未通过，停止扩大实验。研究分支
`work/joint-gradient-v3`，生产入口和历史实验保持原状。

- [学习结果与决定](PILOT_RESULTS.md)：12训练＋12验证、三种子消融、失败门槛。
- [开发集搜索对照](DEVELOPMENT_SEARCH.md)：96个固定预算选择的真实QoR。
- [实现说明](IMPLEMENTATION.md)：相位端口、GPU前反向、共享网络和学习口径。
- [序列化no-op诊断](diagnostics/SERIALIZED_NOOP.md)：相同结构下的Mapping漂移。
- [最终复核](verification/final_checks.json)：9份关键汇总重新生成一致，17项测试通过记录另存于同目录。
- [执行契约](PLAN.md)、[冻结电路清单](learning_manifest.json)。

`structure/` 是旧180组合＋12原图的根因验证；`pilot/` 是新增216次离线
Mapping及复用24个历史标签的汇总，包含192个非空组合和24个serialized
no-op；`development/` 只关联旧12电路已有真实结果，没有新增Mapping。
逐文件原始路径和SHA256见各目录的 `evidence_index.json`。

所有新增测量均CEC PASS、面积合格；但开发搜索选择仍有面积超限，学习
模型的独立验证排序/方向门槛也失败。没有启动后续独立24电路、100或905。
LiteCut Student未训练；没有内部mapped pin对齐、候选真实反馈或真实Delay
回退。大图、权重、库、二进制和完整日志只保存在 `/data`。

复核入口：`scripts/report_joint_gradient_v3.py` 和
`scripts/report_joint_gradient_v3_search.py` 从已完成记录生成小型报告。
计算型脚本必须从已提交源码快照在 `/data` 中通过Slurm执行。当前没有
待续实验，新的算法/测量改动需要下一阶段任务。
