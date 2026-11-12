# 生成文件清理记录

2026-11-13（北京时间）按用户“删除不需要的中间过程文件和测试”的要求清理生成产物。程序源码、测试源码、冻结输入和研究结果保持原样；本次没有执行新的研究实验或训练。

## 已清理范围

| 类型 | 文件数 | 删除条件 |
|---|---:|---|
| LiteCut 重复推理缓存 | 1,984 | `scores` 数组与已保存的 `litecut.json` 逐字节一致；保留完整分数及缓存来源元数据 |
| 旧编译中间文件 | 5,489 | 已结束运行的 `.o` / `.o.d`；排除当前工具构建、native 变体编译证据及所有 `rewrite.cu.o` |
| Matplotlib 字体缓存 | 38 | 可自动生成的 `fontlist-v*.json` |

共删除 **7,511 个文件**，回收分配块估算 **32.71 GiB**。保留来源元数据及删除记录占约 1.66 GiB，净节省估算 **31.05 GiB**。这是按逐文件分配块计算的估算值，不以共享磁盘空闲量变化作为依据。

## 保留内容与后续读取

核心回归测试、不可变源码快照、905 设计主结果、原始候选和顺序 trace、失败日志、CEC、mapping/timing、路径报告、模型、数据集、库和当前二进制均保留。原始工程及 `history/`、`evidence/`、`provenance/` 未清理。

旧 `litecut.json` 中的 `cache.path` 是历史来源字段，清理后可能不存在。完整候选特征和分数仍在该结果文件的 `scores` 中；每个被删缓存的原路径、SHA256、分数数组哈希和 identity 元数据均已归档。读取既有结果无需重新推理；以后需要运行缓存时，在新的运行目录生成。旧构建被清掉的对象文件需重新编译，当前工具与 native writer 所需对象仍保留。

## 核验与记录

清理前逐项核对目标文件哈希及保留副本，再按明确清单删除。清理后 174,062 条受保护路径仍存在，文件标识、大小和修改时间一致；90 份交付根文件及 1,984 份分数文件重新计算 SHA256，均与清理前一致。未重跑算法测试或完整重算全量原始数据哈希。

完整管理目录：`/data/wushangge/CULS_LiteCut/maintenance/20261112T162453Z-cleanup-9620f7d2`。

- `output/deletion_plan.jsonl`：逐文件原路径、删除原因、哈希及保留分数位置。
- `output/deletion_journal.jsonl`：逐项删除记录。
- `output/preserved_cache_identities/`：缓存来源元数据。
- `output/cleanup_summary.json` 与 `cleanup_delivery.json`：核验结果、空间估算及 Slurm 终态。

Slurm 64909（盘点）、64910（核对清单）、64911（删除与复核）均为 `COMPLETED / 0:0`，不使用 GPU。源码清理前基线为 `ab883b3`，本次文档记录位于 `work/cleanup-generated-artifacts` 分支。
