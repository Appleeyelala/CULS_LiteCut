# 运行状态与阶段决定

状态：`DEVELOPMENT_GATE_FAILED`。存储阻碍已解除，本次续跑完成剩余87次独立Mapping和12×6方法消融。179个非空组合具备完整CEC／Mapping记录；885/mask03仍是先前磁盘故障导致的INCOMPLETE_RECORD，未重复测量。12份原预测SHA256保持不变，累计180次非空Mapping尝试。当前无本轮作业仍在运行。

排序、方向及面积门槛未通过，**不启动独立24、100或905**。这不是研究目标完成；下一步应先处理开发诊断暴露的结构表示、预测并列及面积漏判，见[结果与下一阶段决定](STAGE1_2_RESULTS.md)。

| 工作 | 源码 | Slurm | 实际状态 |
|---|---|---:|---|
| 首次恢复／旧构建对照 | 68bde516 | 65981 | 主动停止本轮自建作业；5例消融完成，471构建存在二次复杂度；0新增Mapping |
| 节点集合最小复现 | 68bde516模块＋独立测试 | 65983 | 预期失败：4节点触发14次全图set创建 |
| 修复检查首次启动 | 文件校验快照 | 65984 | 测试依赖暂存不完整，缺少truth模块；未执行检查 |
| 修复后4项检查 | 随后提交6ee8d8af | 65985 | COMPLETED，4/4通过 |
| 完整12例六方法对照 | 6ee8d8af | 65986 | 12例、72选择全部完成；批作业FAILED源于首条unittest包路径错误，原始失败不隐藏 |
| 修正启动方式的完整检查 | 6ee8d8af | 65987 | COMPLETED，15/15通过 |
| 剩余87次Mapping及汇总 | 6ee8d8af | 65988 | COMPLETED，9分13秒；87/87新增完整记录 |
| 数据／CEC／工具hash／回归审计 | 63acabe7 | 65982 | COMPLETED；5个修复前后对照的全PO预测、面积、目标与选择完全一致 |
| 180个已安装AIG结构审计 | ee7b7004 | 65989 | COMPLETED，20秒；检查成功，不代表模型结构匹配或QoR门槛通过 |

计算均在`/data`通过Slurm。GPU作业各4 CPU、32 GiB、1 GPU，最多同时2 GPU；只读审计各2 CPU、8 GiB。没有取消本轮启动前的用户作业。CPU测试与审计均计入64 CPU／256 GiB／2 GPU的总额度。

## 原始证据位置

- 原面板：`/data/wushangge/CULS_LiteCut/runs/20261117T042705Z-joint-gradient-panel-00895eff`
- 已停止的旧构建对照：`/data/wushangge/CULS_LiteCut/runs/20261117T053158Z-joint-gradient-resume-20123cb8`
- 修复后消融／15项检查：`/data/wushangge/CULS_LiteCut/runs/20261117T053917Z-joint-gradient-construction-fix-81e9357c`
- 实际测量续跑：`/data/wushangge/CULS_LiteCut/runs/20261117T054155Z-joint-gradient-panel-complete-f757fa1e`
- 数值与证据审计：`/data/wushangge/CULS_LiteCut/runs/20261117T053543Z-joint-gradient-audit-resumed-da0e92d8`
- 显式图结构审计：`/data/wushangge/CULS_LiteCut/runs/20261117T055017Z-joint-gradient-structure-audit-da4f4e57`

各run.json记录源码、命令、资源、环境和退出状态；小型副本在[verification/resumed_20261117.json](verification/resumed_20261117.json)，数据路径和SHA256在[stage1_2/evidence_index.json](stage1_2/evidence_index.json)。旧记录中的FAILED或CANCELLED不会改成COMPLETED。

## 当前边界

- 数据只用于开发诊断，不是独立测试，不是新的905总体结论。
- 所有选择在本次新增Mapping前冻结；算法不读取候选实测标签，不回退真实Delay退化组合。
- 面积超限记录保留；主表的原始Delay几何比包含超限候选，不能称约束内部署成绩。
- 新代理未证明准确，联合梯度未证明在完整候选域更优，完整优化Runtime和GPU加速优势尚未评估。
- 不重跑A/B、不修改LiteCut、不覆盖旧100/905；仍在研究分支，未切换生产入口、未推送远端。

此前存储中断与部分结果见[STAGE1_PARTIAL.md](STAGE1_PARTIAL.md)，用户授权清理见[LATEST_ONLY_CLEANUP_20261117.md](LATEST_ONLY_CLEANUP_20261117.md)。
