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
