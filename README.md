# CULS LiteCut research workspace

这里集中管理梯度下降与 LiteCut 研究的已有源码、证据和项目历史，研究对象是 AIG rewriting、strict maximal MFFC、候选局部 delay 与映射后 timing。长期目标是形成论文；本次建立可追溯的工作基线，后续研究方向留待讨论。

源码工作区是 `/home/wushangge/CULS_LiteCut`，运行与数据根目录是 `/data/wushangge/CULS_LiteCut`。

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

历史证据覆盖 Gate1、Gate2、Gate3A 至 Gate3F，以及后续 timing 标签更正、source-restricted timing、R9 机制分析和局部 mapper arrival 验证。Gate3F 的四组结果、R9 单案例与局部 arrival 验证具有不同测量目标，不能合并为一个已验证的全局优化算法。

可直接进行轻量文件核验：

```bash
python -B scripts/verify_import.py
```

本仓库的历史脚本保留原样，其中仍有旧绝对路径。它们是有来源的源码快照，不表示所有历史命令已经迁移为可直接执行的统一入口。构建、推理、实验和输出必须在 `/data` 的独立 Slurm 工作目录进行。

上游 CULS、Gate1、ABC 的许可证随源码保留。LiteCut 源目录未找到独立许可证文件；这里记录本地研究来源，未执行对外发布。模型、ASAP7 库和二进制不进入 Git，位置与哈希见 [资产清单](provenance/assets.json)。
