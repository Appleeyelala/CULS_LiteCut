# CULS LiteCut research workspace

这里集中管理梯度下降与 LiteCut 研究的源码、证据和项目历史，研究对象是 AIG rewriting、strict maximal MFFC、候选局部 delay 与映射后 timing。长期目标是形成论文。此前完整优化框架的实现与全量验证已完成；[逐项验收](docs/global_timing/REQUIREMENTS_AUDIT.md)记录完成证据和明确遗漏，[运行记录](docs/global_timing/RUN_LOG.md)区分历史证据与新实验。工程完成不等于论文结论或算法优势已经确立。

联合梯度选择与单电路 GPU 改进已完成正式432方法点、补充48方法点与独立审计，见[最终结果与边界](docs/gpu_joint/FINAL_RESULTS.md)。对旧版QoR非劣门槛通过，但总体中位加速仅1.0355倍，未达到预定1.5倍；9设计27对实际CUDA工作量对照中位加速1.3262倍。梯度相对随机/关闭内层更新的优势未成立；四设计控制未支持局部分数或显式pair的总体收益，保留修复有质量作用。来源/规模/阶段汇总和四组PNG/PDF已完成，见[交付审查](docs/gpu_joint/GOAL_DELIVERY_AUDIT.md)与[复现索引](docs/gpu_joint/REPRODUCIBILITY.md)。原905结果保持原样；本轮交付不等于论文性能目标已达成。

源码工作区是 `/home/wushangge/CULS_LiteCut`，运行与数据根目录是 `/data/wushangge/CULS_LiteCut`。

当前新增MFFC级GPU改造位于 `work/mffc-gpu-905`：完整候选聚合/Region DAG设备队列前反传播、全局梯度联合块和候选复用预检已通过32项数值/回归及6开发输入的18个真实A/B/Ours方法点；这些不替代完整905。当前协议仍为冻结前草案，输入核验和正式准备见[MFFC运行记录](docs/mffc_gpu/RUN_LOG.md)、[实际实现](docs/mffc_gpu/IMPLEMENTATION.md)与[905协议](BENCHMARK905_PROTOCOL.md)。旧905和432/48结果保持原范围，不能作为新版全量性能结论。

新增框架的构建与运行方法见 [使用说明](docs/global_timing/USAGE.md)，数学与真实执行边界见 [模型契约](docs/global_timing/MODEL_AND_EXECUTION.md)，源码与报告的对应关系见 [证据索引](docs/global_timing/EVIDENCE_INDEX.md)。统一版本905输入主实验为901 PASS、4项native对照缺失的PARTIAL；全部905设计的有效测量完成路径采集和原始证据复核，13981份映射、16696个成功引用、最终审计错误0。[全量结果](docs/global_timing/FULL905_SEARCH_RESULTS.md)包含各baseline与original/no-op对照，尚未建立稳定优于同预算随机搜索的优势。

完整 AIG 数据集现位于本仓库的 [open-sourceDataset](open-sourceDataset/)，包含 938 个文件（905 个 AIG/AAG），不进入 Git。后续统一从此目录选择输入；默认设计为 `OpenABC/c1355_orig.aig`，路径配置在 `config/workspace.json`。原 `/home/wushangge/cutLibOrigin/open-sourceDataset` 保留指向新目录的兼容链接，迁移记录见 [dataset_migration.json](provenance/dataset_migration.json)。实际运行前，准备工具会把选定设计复制到 `/data` 的运行目录并记录 SHA256。

建议先读 [已有工作整理](docs/WORK_HISTORY.md)，再读 [代码与数据流](docs/PROJECT_MAP.md) 和 [结论边界与已知问题](docs/KNOWN_LIMITATIONS.md)。后续运行和版本管理遵循 [Git 与 Slurm 工作规范](docs/OPERATIONS.md)。

| 目录 | 内容 |
|---|---|
| `projects/culs/` | CULS 当前源码、Gate3A 至 Gate3F 工具，以及后续 path delay 实验脚本和补丁 |
| `projects/litecut/` | LiteCut 特征、Teacher/Student 源码、推理 bridge、Gate2 工具 |
| `legacy/gate1_aigrewrite/` | 实际完成 Gate1 的工作副本源码及脚本 |
| `third_party/abc/` | 已核对 commit 的干净 ABC 源码，保留上游版权文件 |
| `history/abc_variants/` | 历史 ABC 实验副本相对干净基线的实际修改文件 |
| `evidence/` | 小型历史报告、JSON/CSV 和图；完整原始产物保存在数据区 |
| `provenance/` | 文件来源及 SHA256、原仓库 commit/状态/差异、资产及历史运行索引 |
| `scripts/` | 导入核验、未来运行目录准备工具，不自动执行实验 |
| `config/` | 数据根目录、工程位置和资源上限 |
| `global_timing/` | 新增分区、候选、批量推理、时序代理、梯度搜索与真实验证闭环 |
| `tests/` | 结构参照、数值梯度、候选可行性及真实 CULS/ABC 回归 |
| `framework/abc/` | 独立 ABC 构建的只读 Liberty 端点报告插桩 |

历史证据覆盖 Gate1、Gate2、Gate3A 至 Gate3F，以及后续 timing 标签更正、source-restricted timing、R9 机制分析和局部 mapper arrival 验证。Gate3F 的四组结果、R9 单案例与局部 arrival 验证具有不同测量目标，不能合并为一个已验证的全局优化算法。

可直接进行轻量文件核验：

```bash
python -B scripts/verify_import.py
```

本仓库的历史脚本保留原样，其中仍有旧绝对路径。它们是有来源的源码快照，不表示所有历史命令已经迁移为可直接执行的统一入口。构建、推理、实验和输出必须在 `/data` 的独立 Slurm 工作目录进行。

上游 CULS、Gate1、ABC 的许可证随源码保留。LiteCut 源目录未找到独立许可证文件；这里记录本地研究来源，未执行对外发布。模型、ASAP7 库和二进制不进入 Git，位置与哈希见 [资产清单](provenance/assets.json)。

生成产物的清理范围、保留数据及删除清单见[运行产物清理记录](docs/GENERATED_ARTIFACT_CLEANUP.md)。历史缓存路径可能已清理，完整候选分数与实验原始证据仍保留。
