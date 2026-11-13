# 最终交付审查：验证完成，含明确负结果

目标依据为`USER_GOAL.md`及用户两份研究说明。研究性能目标仍是跨MFFC梯度联合优化、单电路GPU并行、有竞争力的真实mapped delay与低端到端runtime。现有完整证据支持对旧版QoR非劣及GPU独立收益，未支持整体1.5倍加速或梯度胜过简单搜索。交付验收和性能评价分别列明，不以单元测试、组件加速或提交作业替代算法收益。

用户研究说明[§6](../research/CODEX_GOAL_梯度联合优化与GPU加速.md)明确要求：“若尚无优势，交付负结果与原因，不修改评价口径掩盖问题。”本轮按此交付完整改进、基线、贡献分离、质量—时间结果及限制；没有放宽门槛宣布性能达标。正式协议/面板在结果前冻结，使用已有905集合中的16设计、三个预定种子，不能称未见数据集。

## 五项交付与权威证据

| 交付/要求 | 当前已有证据 | 完成时必须检查的证据 | 当前状态 |
|---|---|---|---|
| 1 可复现改进实现 | 局部二次/反馈/批量/自适应/内层控制；prepared replay默认/真实工具回归；正式2257998及tag、REPRODUCIBILITY索引；source audit覆盖432方法点3120模块引用和65快照文件，记录/实际/Git一致 | 资产/二进制hash经原始审计核对，Release CULS/优化ABC构建记录已读；51/36/37项范围分列；补充48点420模块引用与Git匹配，12个默认完整真实轨迹也与正式源相同 | 交付与来源核验完成；不承诺外部import或binary逐字节重建 |
| 2 公平旧版及相关基线 | 固定旧5025a6c+741工具控制；原图、ABC、CULS、LiteCut greedy、自然成本random；64983精确432终态、64984独立1119观察原始审计 | 候选域/输入/技术库/ABC一致；精度、线程、cache与编译优化；调用数、失败/修复和native较少请求如实记录；最终人工结论须按完整分母 | 正式比较与审计已完成；全部432方法点PASS，native面积不可行保留 |
| 3 梯度独立分析 | 正式48配对：对random/no-inner的delay比1.026663/1.043888；直接请求45/161成功，final直接6/repair28/no-op12，其余来源分列 | 实际summary确认28个最终repair父请求为17梯度/11warmup；no-inner保留校准/adjoint，不能叫全无梯度；所有父请求存在、失败且先于子请求 | 分析完成，梯度优势未成立，保留负结果 |
| 3 LiteCut/显式交互/修复贡献 | 65004为COMPLETED/0:0；37回归、真实默认兼容与4配置smoke通过，48方法点/115成功观察审计完成；12个默认轨迹与F一致 | no-score/no-pairs对照各2胜7平3负、比值1.002822/1.002929；no-repair为9胜3平、0.952208；每配置3个no-Decision点保留，时间只在同一A40分配内比较 | 补充控制与分析完成；通道收益不成立，修复质量价值仅限该子集 |
| 3 GPU独立分析 | 正式9设计27个CUDA同工作量配对中位E2E加速1.326246；21对CPU/无提案分列；div单例另列1.929倍且最终no-op | 源码/模型/域/配置/调用数/全部请求指标及分配相同已由hash核对；整体对旧版1.0355与纯GPU贡献分开解释 | 正式统计和四图人工复核已完成；总体1.5倍未达 |
| 4 Delay–Runtime实验 | S的完整统计/CSV，四AND规模组；D的五来源分组/阶段摘要及四组PNG/PDF；30/120/600真实截点、达标分母、original/no-op分列 | S/D的18/14项产物hash全部复核；修正曲线图已查看，另三PNG与已查看S图逐字节相同；四PDF可读取、各1页；固定8请求曲线及条件中位数边界明确 | 结果、导出与人工复核完成；未事后挑时间截点判赢 |
| 5 有效结论/限制/问题 | FINAL_RESULTS解释QoR非劣、runtime未达、梯度/通道负结果、GPU与修复作用；保留开发退化/非正增益负结果 | 已结合全部设计、来源/规模/阶段和祖先关系复核；记录源码/数据/命令/环境/资源/Slurm/hash与路径；CPU内存和组件Torch显存分开，整闭环显存未记录 | 完成报告与局限/研究问题交付，未宣称论文优化优势 |

## 性能目标逐项判定

| 原目标 | 当前完整证据 | 判定 |
|---|---|---|
| 有竞争力的真实QoR | 相对旧版48/48配对geom=0.999398、P90=1.007710；对随机geom=1.026663，对native有效子集也未总体胜出 | 达到预定对旧版非劣；更强竞争优势未建立 |
| 显著减少整体runtime | 配对wall比中位1.0355，预定目标≥1.5；30/120/600截点均完整展示，未新增二元阈值 | 未达到 |
| 梯度相对简单搜索有价值 | 对random12胜19平17负，对无内层控制7胜17平24负；失败、修复/warmup来源均分列 | 未得到稳定收益支持 |
| GPU独立贡献 | 9设计27对实际CUDA，同算法/工作量/请求及实测指标一致，E2E中位1.326246；21对CPU/无提案不计 | 该面板有独立证据；不是整体1.5倍达成 |
| 可靠、可复现 | 正式432点/1119成功观察与补充48点/115成功观察按完整安装、CEC、统一Liberty审计；失败请求不给标签，源码/资产/命令与版本可追溯 | 本轮限定输入与版本内已验证，不是所有rewrite组合均可执行 |

## 不可省略的科学与执行核验

- 每个成功指定组合必须精确全ACCEPTED、CEC明确证明，再使用统一`ABC_SCL_LIBERTY`和同一ASAP7；拒绝部分安装或替换标签。旧MIO/已撤回PI→PO标签不进入正式表。
- source commit区分外层运行快照和旧/新算法实际导入来源；编译二进制以真实build报告/hash定位，不因Python源码相同就宣称二进制来源一致。
- MFFC、Decision、cut-bounded removable region和library candidate不同；一MFFC可含多Decision。所有节点ID空间和映射后ID不强制对齐。
- 梯度只经过可微代理/选择，不经过Student、CEC或ABC。no-inner控制不等于完全无梯度；训练残差下降、非零梯度或global预测下降不当作真实优化证据。
- 面积以真实no-op的1.05倍为准；最终非劣门槛保持原48配对定义，缺项不得按有利子集判通过。相同时间的替代条件只作前瞻已指定口径的描述，不事后挑阈值。
- 原生CULS/GPU能力和新增GPU提案/复用分开归属；整个方法对旧版加速和相同算法CPU/CUDA纯贡献分开。
- 真实结果、source快照、模型与原仓库保持不变；大数据/新输出均在`/data`。无发布、force-push或外部remote配置。
- 四作业真实终态均已检查：64983 COMPLETED/0:0（1:42:27）、64984 COMPLETED/0:0（7:01）、65004 COMPLETED/0:0（30:01）、65009 COMPLETED/0:0（13s）。不存在把PENDING/RUNNING当完成或重复提交原实验的情况。
- A四配置ALU_mod末seed的原始trace/CEC/GT_META再作人工抽查：默认422、no-score420、no-pairs419个请求全部ACCEPTED/CEC通过，no-repair为0个改写/no-op、同样等价；115个A成功观察的完整覆盖由原始审计提供，抽查不代替全覆盖。

## 实际最终证据位置

- F（正式实验）：`/data/wushangge/CULS_LiteCut/runs/20261113T062048Z-frozen-joint-final-panel-b9c390b3/`，源2257998，Slurm64983。
- S（后处理）：`/data/wushangge/CULS_LiteCut/runs/20261113T070050Z-frozen-joint-final-synthesis-f895a375/`，源acdf988，Slurm64984，afterany:64983。
- A（补充控制）：`/data/wushangge/CULS_LiteCut/runs/20261113T073402Z-channel-repair-ablation-c9634975/`，源db64308，Slurm65004；`output/{ablation_tests.json,default_compatibility.json,smoke_audit.json,ablation_analysis/}`均实际生成并检查，后者有records/paired_results/evidence_hashes/REPORT/completion。
- D（最终汇总/显示修正）：`/data/wushangge/CULS_LiteCut/runs/20261113T081810Z-joint-final-delivery-d22ea3ab/`，源1994e57，Slurm65009；`output/delivery/`实际包含来源/阶段/会计/来源hash/报告/完成标记及四组PNG/PDF，只消费完成审计产物，不执行EDA/模型。
- 正式原始覆盖在F的`output/{final_panel_completion.json,panel_analysis/}`；独立来源核对、10项统计测试、中文报告和原图在S的`output/{post_tests.json,panel_analysis/,final_synthesis/}`。这些文件均已读取核验。
- 仓库[FINAL_RESULTS.md](FINAL_RESULTS.md)、[REPRODUCIBILITY.md](REPRODUCIBILITY.md)和`provenance/gpu_joint_{source_audit,ablation_source_audit,repair_ancestry,component_structure}.json`给出人工解释与静态来源证明。

上述交付证据均已实际检查。按用户§6，本轮必要代码改进、真实对照及分析报告以包含负结果的形式完成；算法的整体显著加速和梯度优势仍判为未达。未见集泛化、物理signoff、精确local→global模型、全部905新版重跑、外部方法严格复现不属于本轮已证明结论，没有以这些未验证主张填补性能差距。
