# 单输入、多配置 replay 的枚举复用

本项针对真实计时中反复启动原生 replay、重复原图 cut 枚举的成本。它属于执行工程改进，不是新的梯度方法，也不声称提高 QoR。

原 `gate3e_execution_replay` 已允许同一输入包含多个配置，但每个配置均重新做 GPU 枚举/预评估。新增 **opt-in** `CULS_REPLAY_PREPARED=1` 在命令内先枚举一次，再供各配置复用；未设置时继续原路径。记录在输出 `prepared_replay.json`，不修改旧 trace schema。

- 缓存只在同一命令内生存，不读取磁盘中未核对的 cuts。保存原图精确的 reorder fanins、refcounts、接口、全部 cuts 和可执行 candidate mask。
- 每次 replay 仍重新 Reset 原图，并逐项比对缓存身份；不匹配即失败。只允许显式正增益 replay，不能给普通 rewrite/zero-gain 流程套用缓存。
- replacement builder 重新上传原图、拆分 literal 并建立原结构哈希；只跳过重复 cut 枚举与预评估。它仍在 GPU 构建替换 DAG，CPU 仍按指定顺序重查 leaf、saved/added、依赖环和完整安装。
- 不并行安装同一 AIG 内冲突的 rewrites；不同配置彼此独立，分别输出完整 trace/AIG。Python 上层仍必须核对“请求=preselected，全部 ACCEPTED”，然后 CEC 和统一 ABC timing；部分输出不能给请求贴成功标签。
- 新增 `global_timing/batch_execution.py` 的可关闭批量接口，默认 pipeline 保持原顺序执行。批量接口尚需集成回归和真实搜索对照，不能据此声称其端到端时间已经减少。

源码在 `projects/culs/src/algorithms/rewrite.{h,cu}`。`tests/test_prepared_replay.py` 比较 no-op 和 GG/GL/LG/LL 的全部 trace/AIG，与默认路径及冻结旧二进制逐项相等，并对新产物做 CEC/mapping；另检查非法候选拒绝和真实动态拒绝 fixture。`scripts/build_replay_tools.py` 只重建 CULS，继续使用冻结 ABC；默认路径与开启复用的已有真实工具回归分别执行。构建/运行结果必须以实际 Slurm 文件为准，本文件不把尚未执行的检查写成 PASS。

注意历史 CULS script dispatcher 在部分命令失败时仍可能以进程状态 0 结束。本项没有改变该既有 CLI 语义。非法候选的验证必须同时检查明确错误及成功 AIG/trace 缺失；上层原有的文件、精确请求集合与全部 ACCEPTED 检查继续保留，不能只用进程 return code 判成功。

## Python 批量执行边界

`replay_many` 将同一反馈批次的独立配置交给一次原生命令。要求 manifest 完整、prepared provenance 匹配、每个请求精确 preselected 且全部 ACCEPTED；命令整体异常/不完整时所有计划请求失败，动态单配置拒绝则单独保留。各请求目录保存可单独回放的 choices、replay 摘要及指向共享原始 trace/AIG 的链接，共享命令只计一次，不复制为多次调用。

`BatchedExperiment.evaluate_many` 在 replay 前保存预测，最多使用分配内的 CPU workers 对完整成功的 AIG 各自执行 CEC 和 mapping。失败组合不送入 evaluator。返回顺序固定为请求顺序，修复仍是新预算请求；各请求 wall 含共享/并发部分，不能相加当总时间。profile 入口 `feedback_batch_cpu/feedback_batch_cuda` 的时间轴在每个请求实际验证完成时记录，CPU/CUDA 使用相同的批量引擎和 worker 数。并行 ABC/枚举复用的工程贡献必须与新增 GPU 提案贡献分开比较。

Slurm 64939 的 7 项集成/原有 pipeline 回归已通过，包括真实混合成功/拒绝批次、逐成功项 CEC/mapping、独立 replay 的 AIG 字节一致与预算/来源/预测保存；自由梯度搜索中的选择等价与 runtime 收益仍待端到端对照。
