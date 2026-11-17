# 全局可微代理与多 MFFC 联合梯度研究

**状态：阶段0完成；12电路组合诊断及六方法消融已执行完毕（179完整测量＋1历史残缺）。开发质量门槛未通过，停止扩大到24／100／905。论文目标仍未达成；见[最新结果](STAGE1_2_RESULTS.md)。**

研究契约见 [PLAN.md](PLAN.md)，冻结配置见 [joint_gradient_v2.json](../../config/joint_gradient_v2.json)。本轮以非正 Gain 905 为历史参照，和旧 STE100、历史方向导数905分别统计。

- [阶段0汇总](stage0/summary.json)、[家族分组清单](stage0/cohort_manifest.json)、[逐电路审计](stage0/evidence.csv)：905条证据及训练/验证/测试划分。
- [当前完整开发诊断](STAGE1_2_RESULTS.md)、[72个方法选择](stage1_2/selected_results.csv)、[审计](stage1_2/audit.json)、[结构参照](stage1_2/structure_summary.json)。
- [阶段1部分结果](STAGE1_PARTIAL.md)、[完整统计JSON](stage1_partial/summary.json)、[192组合表](stage1_partial/combinations.csv)、[选择结果](stage1_partial/methods.csv)、[两两交互](stage1_partial/pair_interactions.csv)：包含缺失、未执行和不完整记录。
- [实现与边界](IMPLEMENTATION.md)：目标对齐、端口组合代理、联合可行梯度搜索及2×2入口。
- [运行和恢复](RUN_STATUS.md)：Slurm、源码、失败原因、剩余工作。
- [本轮15项检查及Slurm记录](verification/resumed_20261117.json)：构建复杂度回归和全部聚焦检查通过；新代理准确性门槛仍失败。
- [先前13项检查](verification/unit_v3.json)：结构压缩参照、相位、梯度、联合搜索、预算、合法性和分组/流式读取；恢复协议另有1项轻量检查。

所有新 Mapping 是预先冻结请求的离线研究验证。选择器没有候选真实标签入口，没有真实 Delay 回退。本轮没有修改 LiteCut、历史优化入口、905或100实验文件，没有运行新A/B。A/B数字从相同输入SHA256的旧基线读取；它们不是本轮重测。

12个开发电路的域限制为历史已选组合中的4个兼容二选一Decision，跨至少2个MFFC。它用于机制诊断，不能代表完整候选域、独立验证24例或测试100例。
