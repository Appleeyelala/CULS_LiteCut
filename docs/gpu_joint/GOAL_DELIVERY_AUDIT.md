# 目标交付审查：尚未完成

目标依据为`USER_GOAL.md`及用户两份研究说明。最终目标仍是跨MFFC梯度联合优化、单电路GPU并行、有竞争力的真实mapped delay与低端到端runtime；不以通过单元测试、组件加速或提交正式作业替代完成。

用户研究说明§6允许在尚无优势时交付负结果与原因；这不免除完整基线、真实验证、贡献分析和诚实覆盖的要求，也不允许事后修改评价口径。正式协议/面板在结果前冻结，使用已有905集合中的16设计、三个预定种子，不能称未见数据集。

## 五项交付与权威证据

| 交付/要求 | 当前已有证据 | 完成时必须检查的证据 | 当前状态 |
|---|---|---|---|
| 1 可复现改进实现 | `joint_optimization.py`的局部二次/反馈/批量/自适应/内层控制；原生prepared replay与Python完整验证；快照2257998及冻结tag | 核对实际源码、所有开关默认语义、模型/库/二进制hash、构建记录；51/36项各自范围，不把重复回归求和 | 代码与开发回归已有；最终来源核对待完成 |
| 2 公平旧版及相关基线 | 固定旧5025a6c+741工具控制；原图、ABC、CULS、LiteCut greedy、自然成本random；64983精确432终态、64984独立1119观察原始审计 | 候选域/输入/技术库/ABC一致；精度、线程、cache与编译优化；调用数、失败/修复和native较少请求如实记录；最终人工结论须按完整分母 | 正式比较与审计已完成；全部432方法点PASS，native面积不可行保留 |
| 3 梯度独立分析 | 正式48配对：对random/no-inner的delay比1.026663/1.043888；直接请求45/161成功，final直接6/repair28/no-op12，其余来源分列 | 最终结合通道研究解释作用；不能把无内层控制叫全无梯度，也不能把repair/warmup收益全归梯度 | 主面板结果与人工复核已有，当前为负结果；补充控制待完成 |
| 3 LiteCut/显式交互/修复贡献 | 四设计补充65004已通过37回归、同分配真实默认兼容与4配置smoke原始审计，主面板不变 | 48方法点完整覆盖、原始审计与逐项结果；结论仅限四设计，no-score仍保留Student成本、no-pairs仍有DAG耦合 | 前置验证已完成，正式补充正在运行 |
| 3 GPU独立分析 | 正式9设计27个CUDA同工作量配对中位E2E加速1.326246；21对CPU/无提案分列；div单例另列1.929倍且最终no-op | 源码/模型/域/配置/调用数/全部请求指标及分配相同已由hash核对；整体对旧版1.0355与纯GPU贡献分开解释 | 正式统计和四图人工复核已完成；总体1.5倍未达 |
| 4 Delay–Runtime实验 | 完整S统计/CSV及四组PNG/PDF；30/120/600真实截点、达标分母、original/no-op、实际工具成本分列 | 四PNG已查看；需在独立D修正曲线图顶端遮挡和offset，并补来源分组/阶段摘要；明确固定8请求曲线不是期限内持续增预算实验 | 正式结果已有，显示修正与附加分组65009待65004终态 |
| 5 有效结论/限制/问题 | FINAL_RESULTS已人工解释QoR非劣、runtime未达、梯度负结果、GPU独立贡献；保留开发退化/非正增益负结果 | 补充通道数值与来源/阶段摘要、修正图人工核验、最后逐项审查，不能以负结果许可免除缺项 | 主面板说明已有；目标保持未完成，待最后补充证据 |

## 不可省略的科学与执行核验

- 每个成功指定组合必须精确全ACCEPTED、CEC明确证明，再使用统一`ABC_SCL_LIBERTY`和同一ASAP7；拒绝部分安装或替换标签。旧MIO/已撤回PI→PO标签不进入正式表。
- source commit区分外层运行快照和旧/新算法实际导入来源；编译二进制以真实build报告/hash定位，不因Python源码相同就宣称二进制来源一致。
- MFFC、Decision、cut-bounded removable region和library candidate不同；一MFFC可含多Decision。所有节点ID空间和映射后ID不强制对齐。
- 梯度只经过可微代理/选择，不经过Student、CEC或ABC。no-inner控制不等于完全无梯度；训练残差下降、非零梯度或global预测下降不当作真实优化证据。
- 面积以真实no-op的1.05倍为准；最终非劣门槛保持原48配对定义，缺项不得按有利子集判通过。相同时间的替代条件只作前瞻已指定口径的描述，不事后挑阈值。
- 原生CULS/GPU能力和新增GPU提案/复用分开归属；整个方法对旧版加速和相同算法CPU/CUDA纯贡献分开。
- 真实结果、source快照、模型与原仓库保持不变；大数据/新输出均在`/data`。无发布、force-push或外部remote配置。
- 正式作业64983与依赖后处理64984均需用sacct检查终态、退出码与原始日志。PENDING/RUNNING或旧run.json状态不证明完成，也不允许重复提交主实验。

## 预期最终证据位置

- 正式实验：`/data/wushangge/CULS_LiteCut/runs/20261113T062048Z-frozen-joint-final-panel-b9c390b3/`，源2257998，Slurm64983。
- 后处理：`/data/wushangge/CULS_LiteCut/runs/20261113T070050Z-frozen-joint-final-synthesis-f895a375/`，源acdf988，Slurm64984，afterany:64983。
- 补充控制：`/data/wushangge/CULS_LiteCut/runs/20261113T073402Z-channel-repair-ablation-c9634975/`，源db64308，Slurm65004，afterany:64984；实际产物应在`output/{ablation_tests.json,default_compatibility.json,smoke_audit.json,ablation_analysis/}`，尚待生成。
- 最终汇总/显示修正：`/data/wushangge/CULS_LiteCut/runs/20261113T081810Z-joint-final-delivery-d22ea3ab/`，源1994e57，Slurm65009，afterany:65004；`output/delivery/`仅消费已完成原始审计产物，不执行EDA/模型。尚待实际生成及人工检查，不能凭代码记完成。
- 正式原始审计/覆盖应在前者`output/{final_panel_completion.json,panel_analysis/}`；新来源核对、统计测试、中文报告与图应在后者`output/{post_tests.json,panel_analysis/,final_synthesis/}`。这些是预期路径；须在实际生成后核验，缺失不能补造。

只有完成上述实际证据检查、导出/人工复核和最终说明后，才能把本轮研究实现与验证任务标为完成。数值门槛未达时必须明确写负结果；不可用工程完成冒充算法优势。
