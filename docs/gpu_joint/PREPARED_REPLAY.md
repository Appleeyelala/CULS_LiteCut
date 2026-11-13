# 单输入、多配置 replay 的枚举复用

本项针对真实计时中反复启动原生 replay、重复原图 cut 枚举的成本。它属于执行工程改进，不是新的梯度方法，也不声称提高 QoR。

原 `gate3e_execution_replay` 已允许同一输入包含多个配置，但每个配置均重新做 GPU 枚举/预评估。新增 **opt-in** `CULS_REPLAY_PREPARED=1` 在命令内先枚举一次，再供各配置复用；未设置时继续原路径。记录在输出 `prepared_replay.json`，不修改旧 trace schema。

- 缓存只在同一命令内生存，不读取磁盘中未核对的 cuts。保存原图精确的 reorder fanins、refcounts、接口、全部 cuts 和可执行 candidate mask。
- 每次 replay 仍重新 Reset 原图，并逐项比对缓存身份；不匹配即失败。只允许显式正增益 replay，不能给普通 rewrite/zero-gain 流程套用缓存。
- replacement builder 重新上传原图、拆分 literal 并建立原结构哈希；只跳过重复 cut 枚举与预评估。它仍在 GPU 构建替换 DAG，CPU 仍按指定顺序重查 leaf、saved/added、依赖环和完整安装。
- 不并行安装同一 AIG 内冲突的 rewrites；不同配置彼此独立，分别输出完整 trace/AIG。Python 上层仍必须核对“请求=preselected，全部 ACCEPTED”，然后 CEC 和统一 ABC timing；部分输出不能给请求贴成功标签。
- 此阶段只增加原生命令能力。梯度 pipeline 尚未批量调用该功能，不能据此声称其端到端时间已经减少。

源码在 `projects/culs/src/algorithms/rewrite.{h,cu}`。`tests/test_prepared_replay.py` 比较 no-op 和 GG/GL/LG/LL 的全部 trace/AIG，与默认路径及冻结旧二进制逐项相等，并对新产物做 CEC/mapping；另检查非法候选拒绝和真实动态拒绝 fixture。`scripts/build_replay_tools.py` 只重建 CULS，继续使用冻结 ABC；默认路径与开启复用的已有真实工具回归分别执行。构建/运行结果必须以实际 Slurm 文件为准，本文件不把尚未执行的检查写成 PASS。
