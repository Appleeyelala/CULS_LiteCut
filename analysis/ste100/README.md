# STE100：正式结果与梯度记录

来源：冻结源码 `9a199bf0ea0dcf20bf55e40da8caa8b8bbff5fac`，Slurm65534。
固定从905抽取100电路（seed20261116）；优化seed20261117，40步STE＋Adam。
A/B/Ours共300终态全部PASS；67个最终候选完整ACCEPTED及CEC PASS，31个真实改善。

- `STE100_RESULTS.csv`：100行三方法真实global Delay与完整冷启动Runtime。
- `STE100_SUMMARY.md`：真实配对比较；`STE100_STATUS.json`：300终态和Mapping次数。
- `STE100_SAMPLE_MANIFEST.json`：样本ID、输入路径、SHA256。
- `cases/NNNN/optimization/trajectory.json`：83个有候选电路的40步原始梯度/更新/loss轨迹（逐字节复制）。
- `cases/NNNN/optimization/proxy_best.json`：原始代理最优组合；此处预测不是真实Mapping标签。
- `cases/NNNN/summary.json`：100个Ours摘要，含阶段时间、后端、原图/最终真实测量与最终来源。
- `records/NNNN_{A,B,Ours}.json`：300个方法终态和完整Runtime。
- `execution/`：实际命令、节点/GPU/线程记录；`FILE_MANIFEST.json`：每个来源、hash及提取规则。

精简规则：cases摘要及records递归省略`po_arrivals_ps`、`po_rise_fall`和`abc_native_trace_lines`；choices向量改成可无损恢复的length/default/nonzero编码。其余字段保留。顶层五份交付文件及optimization文件保持原始字节。不存在未提交候选的真实Timing，不能把预测当标签。

未上传AIG、netlist、模型、库、大型候选目录和重复日志。完整原件仍在服务器：
`/data/wushangge/CULS_LiteCut/runs/20261116T070435Z-ste-adam-random100-d532863a/output/`。
本数据仅属STE100，禁止与`analysis/benchmark905/`旧方向导数/多Mapping反馈实验混算。
