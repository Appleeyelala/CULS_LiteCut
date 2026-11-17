# full905 后续执行修复（2026-11-17）

先读 [根因、修复及研究主线](ROOT_CAUSES_AND_REPAIRS.md)。
本目录是新的诊断/回归结果，**不替换历史full905或STE100成绩**。

- [MEMORY_REGRESSION.json](MEMORY_REGRESSION.json)：原版16/24独占40步PASS、827独占OOM；分块伴随后三例40步全部PASS。含源码commit、原输入SHA256、原始证据路径与hash、显存峰值和选择一致性。
- [REPLAY_REGRESSION.json](REPLAY_REGRESSION.json)：原4失败组合的旧/新binary对比；4失败例和2成功对照在两种Replay路径下全部完整安装且CEC PASS。
- [LITECUT_BOUNDED_PROFILE.json](LITECUT_BOUNDED_PROFILE.json)：超时552保存候选中的64个Decision、393个选项的函数剖析。不是完整case耗时测量。

修复源码入口：`global_timing/gpu_slot.py`（同GPU调度）、`global_timing/fixed_incidence.py`/`mffc_model.py`（等价分块伴随）、`projects/culs/src/algorithms/rewrite.cu`（联合安装边界生命周期）。
对应回归工具：`scripts/probe_nonpositive_memory.py`、`check_saved_replay.py`、`profile_saved_litecut.py`。

新原始产物、二进制和日志仅在各JSON指向的`/data/wushangge/CULS_LiteCut/runs/`目录中。
所有计算经Slurm。没有新Mapping，没有重新运行A/B或100/905全量，没有改变STE/Adam、no-op、优化目标或候选域。
40步探针使用已保存候选/LiteCut分数，不能算作已实测的一次Mapping完整部署性能。

后续研究目标已经由用户确认：CEC PASS；真实Mapping Delay整体优于B为首要目标，Area改善次之；最终只做一次原图Mapping；GPU加速覆盖主要耗时。
下一步应检查梯度优化目标和联合组合选择，再安排保持语义的主要阶段GPU化；不预设必须改预测模型，不自动启动新一轮实验。
