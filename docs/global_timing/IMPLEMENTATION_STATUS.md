# 全局时序优化系统：验收与进度

本次开发由 `USER_REQUIREMENTS.md` 授权，覆盖完整实现、真实实验、全仓库程序测试及所有可运行 benchmark。此前 README 中“方向留待讨论”描述的是导入时的状态。历史记录与原始仓库不改写。

## 2026-11-12 全量运行后的状态

统一源码 `160b218` 的两个预定分片已经结束：64399记录453个偶数输入（450 PASS、3 PARTIAL），64397记录452个奇数输入（451 PASS、1 PARTIAL）。合计905个唯一输入、901 PASS、4 PARTIAL；全部索引和文件hash唯一。四项缺失均为native ABC rewrite对照：codec、loco_top、mrc3、mrc4，原始断言日志和空指标保留。完整运行不等于优化改善。

64854已完成全905保存搜索与原始执行审计：905设计来源审计错误0、顺序replay执行核验905 PASS。与同预算随机对照的面积约束/joint胜平负为179/464/262、252/379/274，均未建立稳定优势；相对原图delay的改善及预测/修复来源见[FULL905_SEARCH_RESULTS.md](FULL905_SEARCH_RESULTS.md)。失败编排64853及原始实验记录完整保留。

第十三批64852已正常结束，167/167设计通过；与前十二批精确互斥、合计905设计、13981份映射和16696个成功引用。64905已COMPLETED/0:0，905项全部ANALYZED，原始CEC/Liberty日志/netlist/arrival链审计错误0，无损坏快照或未选定报告，final_validation为PASS。本轮工程实现、完整测试记录、全量实验及结果分析已完成；已知遗漏和研究限制保留，详见REQUIREMENTS_AUDIT。各阶段旧前缀见RUN_LOG，不替代最终状态。

## 验收矩阵

| 要求 | 完成证据 | 当前状态 |
|---|---|---|
| 代码、依赖、数据库、数据集清单 | 64395的2515项程序矩阵、905唯一输入暂存/hash、provenance | 全部程序有记录；未检出数据库API/文件；import/历史训练遗漏明确 |
| 原始baseline复现 | 64324 Gate3A–F、64399/64397全905各方法 | 901 PASS/4 PARTIAL，仅四项native对照缺失，无替代数值 |
| MFFC覆盖、唯一owner、无环DAG及传播距离 | 64395全905独立结构证书、80小图oracle及错误反例 | 905 PASS；结构距离不冒充mapped timing |
| Candidate、no-op、冲突、顺序与回退 | 64854原始cut证明、候选域、choices/trace/CEC/AIG审计905 PASS；原生引用环拒绝及修复回归 | 成功请求均完整ACCEPTED；静态投影不保证动态安装，失败/新修复请求分开 |
| LiteCut语义、批量及缓存 | 326候选函数与历史逐项预测；123286条19维特征/score精确对照；全905实际推理及hash | 冻结Student已接入；local abc_delay不是global/path delay |
| 局部→region/global可微传播及校准 | surrogate/arrival、905设计来源审计错误0；按设计平均归一化Δdelay MAE 0.11715/0.12134 | 可运行近似已评估；未建立准确性或未见设计泛化 |
| 多PO、关键/近关键链及变化 | 十三批905设计、13981份映射、16696个引用的fresh CEC/测量/网表一致及代表链记录 | 采集及原始证据交叉核验全905完成、错误0；237个序列化对照改变代表对，不证明唯一逻辑路径身份 |
| 面积约束与delay–area联合目标 | 全905两模式实际测量；scalar_objective及有限差分/双目标测试 | 面积模式905点满足no-op面积1.05倍；joint无该硬约束，5点超出另一模式上限 |
| 离散联合梯度、真实反馈与收敛 | Softmax/STE/Adam、冲突投影、先预测后标签、905份梯度/观察/收敛报告 | 同预算random胜平负179/464/262及252/379/274；直接梯度成功1567/3339、1685/3390，修复来源明确 |
| 全仓库静态/单元/集成/回归/E2E | 最新86项CPU矩阵74 PASS/12 SKIP，两主分片各75项真实核心PASS；2515程序矩阵、5个ABC变体/Gate1真实证据 | 命名测试PASS并集覆盖86名，按相同源码复用；17 import FAIL/1 SKIP及逐文件覆盖边界见TEST_COVERAGE |
| c1355及所有可运行benchmark公平比较 | 64399偶数453、64397奇数452；同一160b218/工具/库/种子/预算；64854固定cohort，64905 baseline总览 | 全905终态已核验；原图/no-op分列，native缺失保留，未以旧开发重跑替换结果 |
| 复现、异常、CI和文档 | c1355同源同参数独立重复：完整数值/选择/25项图结构相同；本地CI及来源匹配回归，USAGE/模型契约/限制/验收 | c1355搜索复现与全905保存测量再检分开，二者均有实际通过证据；不声称全905搜索轨迹已重复 |

状态只由实际证据更新。读取附件、产生计划、通过单个 smoke 均不代表系统完成。

## 复用与依赖

输入 AIG → Gate3A strict MFFC/owner → Gate3B Region DAG → Gate3C GPU cuts/candidate masks → Gate2A materialization + Student → 合法配置管理/可微代理/搜索 → Gate3E sequential replay → 严格 CEC → 统一 ABC Liberty mapping/timing → 校准反馈。

- `projects/culs/tools/gate3a_analyze.py` 提供独立 deref/ref 参照；其全根两两包含检查需要关注规模。
- `gate3b_analyze.py` 提供 binary AIG 解析、DAG 和结构路径；结构路径不作为已映射 timing path。
- `src/algorithms/rewrite.cu` 的 Gate3C/3E 接口恢复原 AIG ID，并保留正节点增益筛选及顺序安装语义。
- `gate3e_run.py` 的 predictor 复用 LiteCut 的 19 维 Student、归一化和逆变换；现有实现逐候选调用。
- 历史 CEC 辅助函数的宽松字符串匹配必须在新系统中被严格验证替代。
- Gate3F 的 frozen binary 用于历史基线对照；新增统一 endpoint timing 需要可构建源码和独立二进制，不能冒充历史二进制。

## 设计约束

Region variable 表示该 region 的合法局部配置（允许包含多个 decision），或用受约束的 decision variables 作分解；不能把 region 与单个 rewrite 等同。全局组合必须在下发前进行结构冲突检查，下发后核对每个请求是否 ACCEPTED。

代理模型与真实评估明确分离；最终 delay/area 只能来自实际 mapping。预测校准记录训练样本与未参与拟合的评估样本。所有不可用依赖、不可运行输入、失败执行均有状态与原因，不用数值填补。

## 运行与版本管理

开发分支 `work/global-timing-framework` 从数据迁移提交 `5af5cae` 分出。使用 `scripts/prepare_run.py` 准备已提交快照，所有构建、批量测试与实验在 `/data/wushangge/CULS_LiteCut/runs/` 中经 Slurm 执行。每个阶段保留提交、job ID、命令、输入/模型/库/二进制哈希与运行日志。
