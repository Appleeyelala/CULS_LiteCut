# 用户要求逐项验收

要求来源为[USER_REQUIREMENTS.md](USER_REQUIREMENTS.md)，与原附件SHA256 `6c3aae48ad73c854f00f1db2f26fc9aeb87f0130124f91b391a2677b2af2bb2a`一致。当前尚有最后路径分析待完成；本文件记录可核验的完成证据及明确缺项，不用阶段成功替代整个目标。

所有运行路径的共同前缀为`/data/wushangge/CULS_LiteCut/runs/`。

## 代码检查与六阶段实现

| 要求 | 当前证据与范围 | 状态 |
|---|---|---|
| 2.1–2.3 文件/依赖/数据库盘点、复用与接口缺陷检查 | `provenance/`、PROJECT_MAP、全程序矩阵及TEST_COVERAGE；2515项源码/入口，数据库API/文件未检出。源码与历史差异保留 | 有证据；未恢复的训练依赖明确列出 |
| 2.4 原始baseline、环境、命令和差异 | 64324历史Gate3A–F复现；64399/64397固定源码全905比较，每例config、工具命令、环境和hash；四项native缺失不填值 | 完成并保留缺项 |
| 2.5–2.6 复用与依赖关系 | projects/culs与projects/litecut接口复用；PROJECT_MAP、MODEL_AND_EXECUTION和USAGE解释新旧数据流 | 有源码与运行证据 |
| Phase1 覆盖、唯一owner、精确无环DAG、结构传播与身份映射 | 64395独立结构证书905 PASS；fanout闭合与边界不可吸收、逐边DAG、全部入口反向AND距离；80小图对照历史oracle及错误反例 | 完成；结构距离不是mapped timing |
| Phase2 root/cut/candidate/no-op、冲突、顺序、回退 | candidates.py、cut_validation.py、pipeline.py；64854原始执行审计905 PASS，含cut证明、choices、trace全ACCEPTED、顺序、AIG及CEC。动态失败请求保留并单独修复 | 成功请求完整验证；不保证所有静态投影请求可执行 |
| Phase3 Student语义、特征、批量、缓存与局部/全局区分 | 19维Student冻结；64357的123286条特征/score精确对照及真实缓存测试；全905 litecut.json及模型hash保留 | 完成推理接入；未重建Teacher训练 |
| Phase4 DAG时序、多PO、近关键、加权LSE、面积及联合目标 | surrogate.py/arrival.py：端口结构成本、局部/直接前驱交互、no-op锚定、全部PO传播；scalar_objective支持正PO权重、面积惩罚和Smooth Tchebycheff；真实面积最终复核 | 已实现/执行；代理误差与路径身份边界明确 |
| Phase5 可微松弛、联合更新、离散化、反馈、梯度与收敛 | optimization.py的Softmax/STE/Adam及冲突投影；pipeline.py反馈拟合；905份gradient目录及report/observations.csv、gradients.csv、convergence_and_prediction.png/PDF；64854预测先于标签审计错误0 | 完成可执行近似；不是精确离散导数或AGD复现 |
| Phase6 Rewrite→CEC→统一Mapping/Timing→反馈及失败记录 | 全905 current-source比较、64854执行审计；真实CEC-n按端口位置、统一Liberty与全部PO/global一致；失败没有替代指标 | 主闭环完成；最终原始路径证据核验待64905 |

## 测试要求

| 4.2关键检查 | 自动化及真实证据 |
|---|---|
| 1–2 分区覆盖、Region DAG | test_structure、test_partition_validation；64395全905独立证书 |
| 3–4 cut枚举与candidate等价 | test_truth、test_real_tools上下文cut与全部candidate函数；每例cut_validation原始记录及64854核验 |
| 5 LiteCut推理接口 | test_student_features、test_real_tools批量/缓存；326历史候选与123286特征/score对照 |
| 6 选择合法性 | test_candidates的同region多decision、冲突/空候选/投影；test_pipeline动态拒绝修复；实际原生环拒绝与13项修复CEC |
| 7–9 目标、梯度稳定与多目标 | test_surrogate的多端点锚定、有限差分、双目标、联合梯度/投影；test_arrival深链和显式一阶反传对照 |
| 10–11 多region真实rewrite与ABC | 两个主分片分别75项真实核心PASS；全905保存比较、原始trace及CEC审计；Gate3A–F与5个ABC变体/Gate1的独立真实复现 |
| 12 保存读取、复现与异常 | test_cohort、test_path_changes、test_failure_analysis篡改/来源/标签拒绝；c1355同源同参数独立复现，25项图结构及完整数值/选择相同；异常CEC、空候选、冲突、非有限梯度与native故障有命名测试 |

最新CPU矩阵64395是86项中的74 PASS/12 SKIP；统一奇数主分片真实矩阵为75 PASS。逐测试名核对，两者PASS并集覆盖最新86个测试名；75项真实矩阵涉及的16份测试模块与最新CPU快照逐字节一致。不能把CPU SKIP写成本次CPU执行了GPU测试；证据按各次实际运行保留。

4.1/5的全程序矩阵来自64395：2515项、177入口，1618语法PASS、1395个编译TU、147 import PASS/17 FAIL/1 SKIP、59 CLI smoke PASS。2456项runtime smoke记SKIP并有原因；113个文件有来源匹配的命名功能证据，2402项未测逐文件功能。它不是全仓库行覆盖率或所有历史训练程序功能PASS。缺DGL/einops/DeepGate、无权限原训练数据、导入即写历史目录和需参数入口，见[TEST_COVERAGE.md](TEST_COVERAGE.md)与逐文件test_matrix.json。按用户允许的遗漏报告条款保留，不安装环境、伪造训练数据或覆盖历史输出。

4.3的全905主实验已运行；各方法mapped delay、area、ADP、实际预算、rewrite数、runtime、CEC与来源在64854的comparison_all.csv，完整预测/梯度/收敛在每例report及汇总search_quality。各确定性基线调用数较少，随机与梯度按同输入/同已消耗预算配对。硬件不同的分片不作为同硬件速度比较；单种子/相关变体不证明统计显著性。

## 八项交付与剩余检查

| 交付 | 文件或证据 | 当前状态 |
|---|---|---|
| 1 可执行完整框架 | global_timing、构建/运行脚本、全905真实比较 | 有证据 |
| 2 多目标梯度逻辑与复现 | optimization.py、surrogate.py、保存梯度、c1355独立重复及全905结果 | 有证据；无稳定优势 |
| 3 自动化测试系统 | tests/、scripts/ci.py、run_tests.py、test_repository.py | 有执行证据 |
| 4 测试报告及遗漏 | TEST_COVERAGE、64395矩阵、两主分片核心报告 | 有证据；FAIL/SKIP明确 |
| 5 基线比较 | FULL905_SEARCH_RESULTS、64854 comparison/；64905 performance_overview/ | 主比较完成；各baseline额外总览待64905 |
| 6 预测/梯度/收敛/路径变化分析 | 64854 search_quality/、逐例report/；64905 path_changes/ | 搜索分析完成；全量路径待64905 |
| 7 使用说明 | USAGE、MODEL_AND_EXECUTION、OPERATIONS与各run/job.slurm | 已有具体命令、依赖与参数 |
| 8 限制与失败报告 | LIMITATIONS、NATIVE_CONTROL_FAILURES、NATIVE_REPLAY_CYCLES、PERFORMANCE_AUDIT及原始failure_evidence | 已有证据；最终路径限制仍待核验 |

任务完成前还须确认：64852的167项全部终态、与前十二批互斥合计905；64905成功结束、全905原始路径/CEC/timing/netlist证据复核无遗漏或审计错误；最终报告及源码/测试来源保持一致。未满足时保留目标active。
