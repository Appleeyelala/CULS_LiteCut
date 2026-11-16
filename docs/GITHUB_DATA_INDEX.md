# GitHub 项目与实验数据索引

目标仓库：`Appleeyelala/CULS_LiteCut`。当前公开；源码及正式 905 结果保留全部成功、失败、回退和 no-op 记录。
实验算法冻结 commit：`ecc85e6ac82571caf9f1a6004098dbe976536256`。GitHub 整理提交不代表新算法版本或新实验。

## 最新 STE100（先读）

先读 [交接报告](../HANDOFF.md)、[STE100结果](mffc_gpu/STE100_RESULTS.md)、[实现](mffc_gpu/STE_ADAM_IMPLEMENTATION.md)和[数据](../analysis/ste100/README.md)。冻结算法`9a199bf`，100电路/300方法点；本轮为40步STE＋Adam、至多2次Mapping，无中间反馈。以下是保留的旧905版本，不能与STE100混算。

## 旧905阅读顺序

1. [项目说明](../README.md)、[梯度实现与实验状态](../GRADIENT_OPTIMIZATION_STATUS.md)。
2. [正式实验协议](../BENCHMARK905_PROTOCOL.md)、[MFFC GPU 实现](mffc_gpu/IMPLEMENTATION.md)。
3. [正式结果报告](../BENCHMARK905_RESULTS.md)、[全部 A/B/Ours CSV](../BENCHMARK905_RESULTS.csv)、[统计 JSON](../BENCHMARK905_SUMMARY.json)。
4. [GitHub 数据说明](../analysis/benchmark905/README_GITHUB.md)、[完整电路索引](../analysis/benchmark905/optimization/circuit_index.csv)。

## 文件目录

| 路径 | 用途 |
|---|---|
| `global_timing/` | 候选、LiteCut 推理接口、时序代理、梯度搜索和真实验证闭环 |
| `projects/culs/`、`projects/litecut/` | CULS 与 LiteCut 源码 |
| `framework/`、`tests/` | ABC 插桩及既有测试源码 |
| `docs/mffc_gpu/` | 当前正式 905 实现、审计与交付说明 |
| `analysis/benchmark905/benchmark/` | 全部 905 电路 A/B/Ours 对照和真实测量 |
| `analysis/benchmark905/optimization/` | 每电路摘要、候选、请求、梯度记录与代理校准 |
| `analysis/benchmark905/prediction/` | 请求预测—真实配对、全量 LiteCut 分数与 19 维特征 |
| `analysis/benchmark905/runtime/` | 时间口径、阶段耗时、工具调用和已有 GPU 信息 |
| `analysis/benchmark905/config/` | 正式协议、冻结实现片段及运行环境 |
| `analysis/benchmark905/audit/` | 失败、缺失、正确性审计和原始来源 |
| `analysis/benchmark905/github/` | 大 CSV 分片映射、流式读取程序和上传文件校验 |
| `evidence/`、`history/`、`provenance/` | 历史证据与来源，不与正式 905 数值混算 |

## 口径与读取边界

905 电路均有 A/B/Ours 终态；901 电路三方法均成功，B 有 4 个失败。Ours 有 3041 个请求，3039 成功、2 失败；有效全局预测—实测配对 2134 条。
LiteCut 局部分数不是真实全局 Delay。缺少候选局部 abc_delay 实测标签，不能据此声称局部预测精度已经验证。
不混用 Gate3F MIO arrival、Liberty stime、旧 PI→PO 标签和 mapper mode0 arrival。

大于 4 MiB 的 CSV 按行无损拆分，候选表按 circuit_id 分目录；每片都有表头。特征列保留全局 feature_id 行序。
分片全部还原核验原始 SHA256；未分片的资料包文件保持原始字节。无需下载 ZIP 或读取 Git LFS 对象。
原数据和压缩包继续保存在服务器，未上传完整数据集、模型、Liberty、二进制或大型日志。

对 ChatGPT 指定仓库及上述入口文件，并按问题读取对应数据；数百万行完整统计需流式程序处理。当前仓库公开；使用连接器时是否可读取还取决于应用连接范围。
