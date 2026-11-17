# 全局可微代理与多 MFFC 联合梯度研究

**状态：阶段0完成；阶段1部分完成；阶段2原型及单元检查完成，电路消融尚未运行。存储故障曾阻断验证；[清理后写入恢复](CLEANUP_20261117.md)，研究尚未续跑，不能宣称计划全部完成或新算法已有 QoR 优势。**

研究契约见 [PLAN.md](PLAN.md)，冻结配置见 [joint_gradient_v2.json](../../config/joint_gradient_v2.json)。本轮以非正 Gain 905 为历史参照，和旧 STE100、历史方向导数905分别统计。

- [阶段0汇总](stage0/summary.json)、[家族分组清单](stage0/cohort_manifest.json)、[逐电路审计](stage0/evidence.csv)：905条证据及训练/验证/测试划分。
- [阶段1部分结果](STAGE1_PARTIAL.md)、[完整统计JSON](stage1_partial/summary.json)、[192组合表](stage1_partial/combinations.csv)、[选择结果](stage1_partial/methods.csv)、[两两交互](stage1_partial/pair_interactions.csv)：包含缺失、未执行和不完整记录。
- [实现与边界](IMPLEMENTATION.md)：目标对齐、端口组合代理、联合可行梯度搜索及2×2入口。
- [运行和恢复](RUN_STATUS.md)：Slurm、源码、失败原因、剩余工作。
- [已执行13项检查](verification/unit_v3.json)：结构压缩参照、相位、梯度、联合搜索、预算、合法性和分组/流式读取；恢复协议另有1项轻量检查。

所有新 Mapping 是预先冻结请求的离线研究验证。选择器没有候选真实标签入口，没有真实 Delay 回退。本轮没有修改 LiteCut、历史优化入口、905或100实验文件，没有运行新A/B。A/B数字从相同输入SHA256的旧基线读取；它们不是本轮重测。

12个开发电路的域限制为历史已选组合中的4个兼容二选一Decision，跨至少2个MFFC。它用于机制诊断，不能代表完整候选域、独立验证24例或测试100例。
