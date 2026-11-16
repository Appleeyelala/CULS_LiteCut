# STE + Adam 固定随机 100 电路结果

正式终态：300/300；电路清单：100/905；源码 `9a199bf0ea0dcf20bf55e40da8caa8b8bbff5fac`。
采样种子 20261116；优化种子 20261117；40 Adam steps、1 restart、最多16编辑。
Delay 为同一 ABC/ASAP7 Liberty 的真实 global timing，单位 ps；Runtime 为独立冷进程完整端到端秒数。

| 方法 | 成功/100 | 真实 Delay 中位数 ps | Runtime 中位数 s（所有已完成尝试） |
|---|---:|---:|---:|
| A | 100 | 3123.414675 | 2.172834 |
| B | 100 | 3076.719850 | 2.702031 |
| Ours | 100 | 3090.977665 | 13.175791 |

不同电路的原始 Delay 中位数不能作为配对性能收益；性能比较以下表为准。

| 配对 | 成功配对数 | Delay 比几何均值（正值分母数） | Ours 胜/平/负 | Runtime 比中位数 |
|---|---:|---:|---:|---:|
| Ours/A | 100 | 0.959272 (100) | 31/69/0 | 5.281315 |
| Ours/B | 100 | 1.001070 (100) | 35/27/38 | 4.045126 |

方法成功/失败均保留；Ours Mapping 调用数分布：{2: 67, 1: 33}；最终来源：{'original_fallback': 36, 'ste_adam_verified_improvement': 31, 'original_noop': 33}。
最终候选验证状态：{'PASS': 67, 'NOT_REQUESTED': 33}。失败/面积不合格/无真实改善时返回 Ours 自己已付费测量的原图。
Ours 原图 Mapping 成本计入自身 Runtime；内循环无 ABC/CEC/Mapping、无在线校准或修复重试。
有效候选电路执行40步，零候选电路不加载代理；每步真实梯度与 Adam 更新范数见各 work/optimization/trajectory.json。
未用代理预测替代真实结果；本对照不能单独证明 GPU 相对同算法 CPU 的加速。
胜/平/负容差：max(2e-5 ps, 2e-6 × 两者较大 Delay)；所有配对排除失败，几何平均另排除零 Delay。

路径：STE100_RESULTS.csv、STE100_SAMPLE_MANIFEST.json、STE100_STATUS.json；逐点原始结果见 case_XXXX_{A,B,Ours}/。

## 梯度与真实结果的核对

83个有候选电路全部使用CUDA，均完成40次Adam step及原生DAG反向传播，合计3320次 backward。17个无候选电路跳过模型和优化，返回自身测量的原图；本样本没有触发CPU代理fallback。

80/83有候选电路出现可测logits更新；67/83改变hard组合，代理最优目标及末步loss低于初始值。这67个组合各提交一次真实验证，全部精确ACCEPTED、CEC PASS：31个真实Delay降低、2个相等、34个变差，均未超过1.05倍面积限制。真实Delay变差或不变的36个返回原图；另16个有候选电路仍选择no-op，连同17个无候选电路形成33个仅一次Mapping的结果。不能把代理下降等同于真实Mapping改善。

没有可测logits更新的三个原始ID：242、637的梯度范数为0；836的梯度范数约8.59e-43至1.97e-41，Adam更新未在float64 logits中产生可测变化。没有针对这些电路更换参数、追加搜索或重跑。

结论：按固定参数完成了真实STE＋Adam闭环。相对A，配对Delay几何平均比0.959272（约降低4.07%）；相对B为1.001070（约增加0.107%），没有显示整体优势。配对Runtime中位数仍为A/B的5.28/4.05倍，未实现端到端加速。三方法原始Delay中位数仅作分布描述，不能当作配对收益；该实验也未进行同算法CPU/GPU消融。

## 文件与版本

- 正式源码：`9a199bf0ea0dcf20bf55e40da8caa8b8bbff5fac`；分支 `work/mffc-ste-adam-100`。
- Slurm `65534`：COMPLETED，退出0:0，39分9秒；eda节点，4CPU、64GiB、1×RTX4090。完整300方法耗时之和约2328.068秒，作业时长还包含清单核验和汇总。
- 小检查：Slurm `65533`，1项合成检查＋指定3个smoke；未计入正式统计，见 [实现说明](STE_ADAM_IMPLEMENTATION.md)。没有运行全仓测试或追加对照实验。
- 算法入口：[STE Adam搜索](../../global_timing/mffc_ste_search.py)、[独立Ours流程](../../global_timing/mffc_ste_experiment.py)、[冷进程runner](../../scripts/run_ste100.py)、[协议](../../config/ste100_protocol.json)。旧方向导数入口和历史结果未改动。

正式结果目录：
`/data/wushangge/CULS_LiteCut/runs/20261116T070435Z-ste-adam-random100-d532863a/output/`

- [100电路逐电路CSV](/data/wushangge/CULS_LiteCut/runs/20261116T070435Z-ste-adam-random100-d532863a/output/STE100_RESULTS.csv)
- [固定抽样与输入哈希](/data/wushangge/CULS_LiteCut/runs/20261116T070435Z-ste-adam-random100-d532863a/output/STE100_SAMPLE_MANIFEST.json)
- [300终态、工具、参数及Mapping次数](/data/wushangge/CULS_LiteCut/runs/20261116T070435Z-ste-adam-random100-d532863a/output/STE100_STATUS.json)
- [交付一致性核验](/data/wushangge/CULS_LiteCut/runs/20261116T070435Z-ste-adam-random100-d532863a/output/DELIVERY_CHECK.json)

复现实际命令见 `/data/wushangge/CULS_LiteCut/runs/20261116T070435Z-ste-adam-random100-d532863a/input/command.json`；环境及GPU信息见 `input/session_65534.json`，资源与进程环境见 `run.json`。所有候选原始报告和轨迹留在该run中。
