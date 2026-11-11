# 全局时序优化系统：验收与进度

本次开发由 `USER_REQUIREMENTS.md` 授权，覆盖完整实现、真实实验、全仓库程序测试及所有可运行 benchmark。此前 README 中“方向留待讨论”描述的是导入时的状态。历史记录与原始仓库不改写。

## 验收矩阵

| 要求 | 完成证据 | 当前状态 |
|---|---|---|
| 现有代码、数据库、依赖、数据集和程序清单 | 程序清单、依赖及数据库扫描、输入清单 | 已有逐程序矩阵；905个唯一组合AIG完整暂存/hash；未检出数据库API/文件 |
| 原始程序真实 baseline | 新 Slurm 运行、环境、命令、完整输出、与历史差异 | 64324完成，Gate3A–F复现 |
| MFFC 全覆盖、唯一 owner、正确无环 DAG | 自动化结构及回归测试、实际 benchmark 检查 | c1355与80个生成小图参照通过；跨设计pilot通过，全量正在扫描 |
| candidate、no-op、冲突、顺序与回退 | 注册表、合法性测试、实际执行 trace | 上下文cut证明、常量导出修复及46项真实核心回归通过；初始静态冲突不能保证动态安装，拒绝及显式修复记录完整 |
| LiteCut 语义、批量推理与缓存 | 特征一致性、checkpoint/hash、缓存回归 | 326候选函数与历史逐项预测通过；新增19维提取快捷入口，c1355/div全部123,286条原实现/候选score与旧结果完全一致 |
| 局部到 region/global 的校准与传播 | 可微模型、真实标签、独立验证与误差报告 | 端口DAG与反馈校准已执行；44设计中间预测审计有明显误差，未建立准确性或未见设计泛化结论 |
| 多路径、关键/近关键路径、变化记录 | 统一时序口径的端点与路径输出、测试 | 可选只读observer重构每个95% near-critical PO的最大arrival链；8设计124份映射逐项CEC/时序/网表一致性通过，全量路径分析未完成 |
| 面积约束及 delay-area 联合模式 | 两种模式测试、mapped area/delay/ADP | 单元检查及c1355首次真实搜索已执行 |
| 梯度优化、合法离散化、真实反馈 | 梯度/收敛日志、实际 rewrite/CEC/mapping | 真正joint logits/STE、反馈拟合及两目标已执行；跨设计结果有正有负，未建立稳定优势 |
| 全仓库静态、单元、集成、回归、E2E | 每个程序的 PASS/FAIL/SKIP 与具体原因 | 46项真实核心回归通过；2501项程序矩阵完成，5个ABC变体和独立Gate1均已重建并实际验证；17个import失败原因见TEST_COVERAGE.md。后续梯度性能修复另行回归，最终E2E/全量未完成 |
| c1355 与所有可运行 benchmark | 每输入状态，六类 baseline，真实指标与预算 | 修复后8设计pilot完成，905输入双分片继续；旧版本运行保留。c1355近期两模式均未改善；div/hyp等设计超时，已定位部分开销并核验等语义性能修复；尚未完整验收 |
| 复现、异常处理、持续集成、文档 | 固定种子、可重跑命令、环境、日志与使用说明 | 完整本地CI已执行；重复c1355选择/数值/精确图结构一致；原生ABC时间注释使文件字节不同；最终版本复现仍需核对 |

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
