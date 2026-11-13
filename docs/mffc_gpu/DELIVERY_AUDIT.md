# 完整905交付核验入口

主实验固定为源`ecc85e6`、tag `mffc-gpu-905-freeze-20261114`、Slurm65245。此表记录所需证据及位置；主作业未终态前不判定最终完成。后续纯统计/交付代码与该冻结算法源分别记录，不更换正在执行的源码。

| 要求 | 权威证据 | 当前范围 |
|---|---|---|
| MFFC/Candidate GPU组织与完整DAG依赖 | `framework/mffc_gpu/region_lse.cu`、`global_timing/mffc_model.py`、`mffc_search.py`；65244的41项回归；65239/65240真实原生调用 | 已验证实现；不是仅多轨迹或多设计并发 |
| 候选约束、真实完整安装、CEC | `feasibility.py`、共用`execution.py`；每请求`rewrite/replay.json`和原trace、CEC；65241审计38测量/20成功replay | 开发已验证；正式待全量原始审计 |
| 同一ABC/库、指标与A/B语义 | 冻结机器协议、输入/工具hash、各请求command/result；统一`ABC_SCL_LIBERTY` | 预检通过；正式每点保留身份 |
| 全905和精确2715终态 | `benchmark905_manifest.json`、正式`output/records/`与`formal_completion.json`、Slurm实际终态 | 65245运行中；不得用小样本替代 |
| Delay/Area/cold Runtime与A/B配对 | 冻结分析的`BENCHMARK905_RESULTS.csv`、`BENCHMARK905_SUMMARY.json`，空值/零分母N/A | 自动分析安排在全部方法终态后 |
| 完整安装成功/失败、修复及部分工作量 | 补充`installation_details.json`与新增CSV列；四项`test_mffc_delivery`及65246验证 | 仅读原始小型replay/观察摘要；部分安装不给成功timing标签 |
| 来源/AND与MFFC/DAG形状、四类取舍 | 原始summary分组保持原样；补充`delivery_supplement`；同一冻结九格关系派生具名统计 | 补充为描述性分析，不引入新对照或改门槛 |
| 计时与GPU边界 | cold方法wall、inclusive阶段、原生CUDA事件、整作业accounting | 不用重叠分项相加，不把A/B相对速度说成GPU因果加速 |
| CEC/Mapping/Replay各自的实际耗时 | CSV的`tool_seconds_inclusive`；补充SUMMARY的`delivery_supplement.tool_costs`和MD工具表 | 65248在eda2验证5/5及两开发批次；包含失败/超时点的已有时钟，缺项不补零 |
| 三组图/导出 | 原始`figures/`的PNG/PDF及hash；最终交付逐字节复制 | 实际全量图需完成后人工检查 |
| 六个命名交付与可复现版本 | 根目录方案/协议/manifest、最终CSV/JSON/MD；Git与/data完整运行索引 | 当前前三项已存在；最终三项待终态审计后保存 |

补充入口为 `scripts/finalize_mffc905.py`，验证原始审计产物hash、实验COMPLETED/0:0与精确A/B/Ours覆盖，读取额外安装/时钟摘要，并保持原始比较统计逐项不变。正式运行时还核对47个冻结源码文件及每例Student/native库/gpuls/input实际hash。没有新增rewrite、mapping、模型推理或训练。

失败/中断的缺失时钟、命令文件或观察保持显式N/A/UNKNOWN，不能填零。PASS replay与PASS CEC/mapping分别统计；failed部分接受数只说明真实工作量。具名取舍的前两项可能重叠，原九格矩阵仍是互斥完整划分。

最终发布时，先核对正式/data交付的所有hash、精确905/2715及Slurm终态，再把小型CSV/JSON/中文报告复制到根目录规定名称，PNG/PDF置于`docs/mffc_gpu/figures/`并修正根报告图片链接。根报告首段需用一页以内回答真实Delay/Runtime/Area/CEC/GPU边界与已确认不足；原始/data报告保持原字节。另记录复制来源、源/目标hash和文稿呈现调整，人工检查全量图后才能判定完成。当前仍未满足正式全量终态，不能只因后处理已排队提前发布结果。

最新完整成本交付安排为65249（源b1afd34），`/data/wushangge/CULS_LiteCut/runs/20261113T200926Z-mffc-final905-tool-cost-delivery-67ebf285/output/final_mffc905_delivery/`；依赖65247再依赖65245。这里是待形成的路径，不是已经完成的全量结果。65248的5项测试与开发数据验证只支持该后处理实现，不能代替正式2715点/原始审计和最终人工验收。
