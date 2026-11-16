# 最新正式 905 电路数据：GitHub 阅读入口

本目录来自已核验的 905 分析资料包；正式算法冻结在 `ecc85e6ac82571caf9f1a6004098dbe976536256`。
此次只改变文件组织，没有运行 CULS、ABC、CEC、LiteCut、优化器或新实验。
原资料包说明保留在 [README.md](README.md)，完整项目入口见 [../../docs/GITHUB_DATA_INDEX.md](../../docs/GITHUB_DATA_INDEX.md)。

| 目录/文件 | 内容 |
|---|---|
| `benchmark/BENCHMARK905_RESULTS.csv` | 905×A/B/Ours=2715 条终态，含全部失败；901 电路三方法均成功 |
| `benchmark/BENCHMARK905_SUMMARY.json` | 正式汇总、统计口径及质量/耗时结论 |
| `optimization/circuit_index.csv` | 电路 ID、候选和各文件路径索引 |
| `optimization/cases/0000/work/summary.json` | 每个电路完整搜索摘要；0000 至 0904 |
| `optimization/cases/0000/work/optimization/` | 实际请求 observation、预测、梯度记录、校准及真实测量 |
| `optimization/decisions/0000/part-0000.csv` | 按 circuit_id 分片的全部 4,296,273 decisions |
| `prediction/global_request_pairs.csv` | 3041 请求，3039 成功/2 失败；2134 有效预测—真实 Global Delay 配对 |
| `prediction/litecut_scores/0000/part-0000.csv` | 按 circuit_id 分片的全部 12,698,529 条局部分数记录 |
| `prediction/feature_columns/` | 全部 4,998,334 组精确 19 维特征的列分片 |
| `runtime/` | 端到端、阶段、工具调用和已有 GPU profile |
| `config/` | 实验协议、环境、实现说明及原有读取程序 |
| `audit/` | 原始核验、失败、缺失、来源及校验信息 |
| `github/` | 本次分片映射、读取程序及 GitHub 文件 SHA256 |

**计量边界**：LiteCut local abc_delay、优化代理预测与真实 `ABC_SCL_LIBERTY` delay 分开保存。
局部候选的真实 abc_delay ground truth 未记录，有效局部准确率配对为 0；不能用全局 delay 冒充局部标签。
GPU 记录不能单独证明整套算法相对 CPU 的因果加速；详见原始说明和 runtime/timing_semantics.json。

## CSV 分片与完整性

大于 4 MiB 的 CSV 拆为可直接阅读的带表头分片；不使用 Git LFS 指针或压缩文件。
`github/sharding_manifest.json` 给出每个逻辑原文件的 SHA256、全部分片顺序、全局行号范围和 circuit_id。
全部分片按顺序合并并去掉重复表头，已逐文件核验与原始 SHA256 完全一致。
feature_id 为每列拼接后的 0 基数据行号；分片 first_row 保持这一编号，不能在每片重新编号。
summary/observation 中 `$bundle_ref.file` 仍相对于本目录；JSON 引用目标均原样保留。
原 `audit/CHECKSUMS.sha256`、`FILE_MANIFEST.jsonl` 描述整理前的逻辑文件；本次物理文件清单见 `github/CHECKSUMS.sha256`。
原 `config/read_feature_columns.py` 适用于重建后的逻辑 CSV；直接读本仓库请用新增读取程序：

```python
import sys
sys.path.insert(0, 'analysis/benchmark905/github')
from read_data import iter_csv, iter_feature_vectors
root = 'analysis/benchmark905'
for row in iter_csv(root, 'prediction/litecut_scores.csv', circuit_id=0):
    print(row)
```

源码、汇总表与单个请求可以按文件读取。数百万条记录的全量统计仍需流式程序处理，不能假设一次聊天已读完全部数据。
服务器原数据和 ZIP 均保留，既有结果未覆盖或删除。
