# 实现与证据

- `projects/culs/src/algorithms/rewrite.cu::EvaluateNode`：显式 allowNonPositiveCandidates 仅绕过 gain 正负过滤；根自复用、cut及library语义不变。mask 1/2/3记录正/零/负，非零仍表示成员资格；默认正 gain 调用仍只产生1。
- `GPUSolver::EnumerateAndPreEvaluate` 接收显式策略和可选7项GPU计数；Gate3C输出 candidate_rule=allow_nonpositive_gain、候选 gain 符号及拒绝统计。use_zeros=false 是旧参数，不代表新策略只允许正 gain。
- Gate3E普通/prepared枚举与导出用同一策略；prepared缓存核验策略身份。安装仍须 `CULS_REPLAY_ALLOW_NONPOSITIVE_GAIN=1`，保持cut-leaf/环检查、完整 ACCEPTED 和最终CEC。
- `global_timing/candidates.py` 显式设置并核验 `CULS_ALLOW_NONPOSITIVE_CANDIDATES`，将gain符号随catalogue保留；`execution.py::execute_one`传递两项独立许可。
- `NonpositiveSteConfig`版本化固定协议，旧 `SteSearchConfig` 默认不变。新协议第二次 Mapping 只记录质量，不按 Delay 回退；完整安装/CEC失败无替代指标，真实Area超1.05报告AREA_LIMIT_EXCEEDED并保留观察值。
- STE搜索、可行性规则、LiteCut、12参数代理、GPU DAG源码未改。40步、lr=.12、1restart、T1→.3、no-op+2、max16均冻结。

候选检查源码 `61ec4fcef316f20ff5fed1d8aadbe4932748a48b`；运行路径 `/data/wushangge/CULS_LiteCut/runs/20261116T163957Z-nonpositive-coverage100-2f699c58`。编译命令、独立gpuls及所有冻结工具/模型/库哈希见 `RUN_PROVENANCE.json` 和该run的 `input/commands.json`/`run.json`。全量运行另列其真实源码commit，不把后续发布提交冒充运行提交。

原17例各一个可行替换还独立通过 `verify_library_truth` 穷举库真值检查，见 `LEGAL_REPLACEMENT_WITNESSES_17.json`，Slurm 65899；此检查只读取已有catalogue，不重跑枚举/Mapping。各层计数中的 Boolean 指既有cut真值/原始上下文证明，随后完整Ours的LiteCut准备仍逐候选检查库真值。

最小验证：12项新增零/负gain Replay/CEC探针（含普通与prepared模式），状态 {'PASS': 12}。输入3默认正gain导出的 2104 个候选身份与旧正式记录完全一致。Slurm 65898 的5项政策单元检查验证退化候选保留、失败无替代指标、面积上限、旧回退和固定参数；实际输出见证据索引。

完整文件保留在/data；公开目录只提交源码、表格和小型证据索引，不上传AIG、模型、Liberty、二进制或大型日志。
