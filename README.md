# CULS LiteCut 研究工作区

当前工作分支为 `work/abc-candidate-export`，聚焦于 **AIG 层 level 加权 rewrite 候选选择**。

- **新对话交接与研究现状**：先读 [交接文档：AIG 层 level 加权 rewrite 选择](docs/交接文档：AIG%20层%20level%20加权%20rewrite%20选择（交给%20Gemini）.md)。
- **当前方向实验产物**：[analysis/level_weighted/](analysis/level_weighted/)
  - [Step 1：原图 unit-AND level 与 root slack](analysis/level_weighted/step1/README.md)（另见原生 3 叶子反例说明 [analysis/aig_level_weighted_rewrite/README.md](analysis/aig_level_weighted_rewrite/README.md)）
  - [Step 2：插桩 ABC 导出 4 叶子 rewrite 候选](analysis/level_weighted/step2/README.md)
  - [Step 3 v2：平移软权重与按 slack 分层对比（4 试点）](analysis/level_weighted/step3_v2/README.md)
  - [Step 3 v3：近临界路径 TNS 指标与 44 电路扩展](analysis/level_weighted/step3_v3/README.md)
  - [ABC 原生命令全局基线（44 电路）](analysis/level_weighted/abc_native/README.md)
- **旧方向（LiteCut + 全局可微时序代理 + 联合梯度选择）最终结题归档**：[analysis/paper_joint_evidence/README.md](analysis/paper_joint_evidence/README.md)（旧方向完整过程代码与阶段文档保留在本地分支 `work/paper-joint-evidence` 及初始基线 `main`）。

## 目录说明

| 目录 / 文件 | 内容 |
|---|---|
| `analysis/level_weighted/` | 当前 AIG 层加权 rewrite 选择各步报告、统计表与小体积 CSV |
| `analysis/paper_joint_evidence/` | 旧方向 14 开发电路与 24 外部验证电路结题汇总表及图表 |
| `global_timing/` | AIG 结构解析、MFFC 分区、level/slack 传播、候选冲突与验证基础模块 |
| `scripts/` | 运行目录准备（`prepare_run.py`）、基线校验（`verify_import.py`）与当前实验脚本 |
| `tests/` | 核心结构、层数加权、候选可行性与进程管理单元测试 |
| `projects/`、`third_party/abc/`、`legacy/`、`history/`、`framework/` | 导入的 CULS、LiteCut、干净 ABC 基线及历史参考源码 |
| `evidence/`、`provenance/` | 初始 Gate1–Gate3F 历史小证据、文件哈希溯源与 `rwr_dump.patch` |
| `config/`、`benchmark905_manifest.json` | 工作区路径/资源配置与 905 电路清单 |
| `open-sourceDataset/` | 905 个 AIG/AAG 电路数据集（Git 忽略） |

## 运行与管理规范

- 源码工作区：`/home/wushangge/CULS_LiteCut`
- 数据与运行根目录：`/data/wushangge/CULS_LiteCut`（编译、大规模导出、实验和输出均须在 `/data` 下独立目录通过 Slurm 执行，详见 [docs/OPERATIONS.md](docs/OPERATIONS.md) 与 [AGENTS.md](AGENTS.md)）。
- **盲测集红线**：外部 24 个验证电路仅用于方法与参数完全冻结后的最终一次性验证，开发与调参阶段严禁读取或统计。
- 基础背景文档：[已有科研工作整理](docs/WORK_HISTORY.md) · [代码职责与数据流](docs/PROJECT_MAP.md) · [结论边界与已知问题](docs/KNOWN_LIMITATIONS.md)
