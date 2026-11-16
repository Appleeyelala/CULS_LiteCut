# MFFC GPU 三方法评估：完整905结果

证据范围：905输入、2715终态方法点；原始测量审计5750份，成功replay观察3039份。

| 方法 | 尝试/成功/失败/超时 | 中位冷E2E s | P90 s | 全部终态总记录时间 s | 面积可行成功点 |
|---|---|---:|---:|---:|---:|
| A | 905/905/0/0 | 2.3225 | 5.8847 | 3063.3171 | 898 |
| B | 905/901/4/0 | 2.8215 | 7.6835 | 4646.0542 | 828 |
| Ours | 905/905/0/0 | 18.7669 | 49.6327 | 25640.7139 | 905 |

| 比较 | 成功配对 | Delay胜/平/负 | 改善中位 % | Delay ratio几何均值（分母） | 配对baseline/Ours时间比中位 |
|---|---:|---|---:|---|---:|
| 相对A | 905 | 544/360/1 | 0.3551 | 0.944392 (905) | 0.1350 |
| 相对B | 901 | 354/184/363 | 0.0000 | 0.998084 (901) | 0.1661 |

Delay ratio=Ours/baseline，越低越好；时间比=baseline/Ours，越高越好。全部单方法runtime含失败/超时；配对比只含成功方法。面积单位为固定Liberty的库单位。CEC通过仅指成功测量；失败不给替代delay。

缺少完整冷时钟的方法点数：{'A': 0, 'B': 0, 'Ours': 0}；这类中断保持N/A，不补零，总记录时间仅求和可用时钟。

GPU并行的是Candidate→Decision→MFFC聚合与保留依赖的完整Region DAG前/反传播，最终真实改图仍顺序受控。仅A/B对照不能独立证明GPU或梯度的因果收益；阶段与设备是实现证据。候选预检仍非完整可执行性预测。

原始目录：`/data/wushangge/CULS_LiteCut/runs/20261113T191340Z-mffc-full905-frozen-6a45b8db`；算法源码`ecc85e6ac82571caf9f1a6004098dbe976536256`。所有预测、Student分数与真实ABC_SCL_LIBERTY分列。无候选设计数：94，未知候选数设计：0。

## 完整来源与规模分组

| 分组 | 输入数 | Ours成功/尝试 | 对A Delay比 | 对B Delay比 | A/Ours时间比 | B/Ours时间比 |
|---|---:|---|---:|---:|---:|---:|
| by_source/ACE | 40 | 40/40 | 0.962246 | 1.036525 | 0.1340 | 0.1747 |
| by_source/OpenABC | 47 | 47/47 | 0.915158 | 1.020876 | 0.1280 | 0.1502 |
| by_source/forgeEDA | 805 | 805/805 | 0.945281 | 0.990899 | 0.1369 | 0.1672 |
| by_source/iwls2022 | 5 | 5/5 | 0.965776 | 1.258332 | 0.1103 | 0.1561 |
| by_source/iwls2024 | 8 | 8/8 | 0.927930 | 1.290442 | 0.1015 | 0.1454 |
| by_and_size/0_128 | 0 | 0/0 | N/A | N/A | N/A | N/A |
| by_and_size/128_4096 | 15 | 15/15 | 0.941276 | 0.947970 | 0.1345 | 0.1511 |
| by_and_size/4096_8192 | 241 | 241/241 | 0.945160 | 1.014591 | 0.1287 | 0.1530 |
| by_and_size/8192_16384 | 300 | 300/300 | 0.933537 | 0.976032 | 0.1367 | 0.1670 |
| by_and_size/16384_65536 | 291 | 291/291 | 0.958743 | 1.010853 | 0.1417 | 0.1819 |
| by_and_size/65536_plus | 58 | 58/58 | 0.927176 | 0.994828 | 0.1101 | 0.1652 |

## 面积条件下的质量—时间关系

仅统计A/B/Ours共同成功且均满足本方法付费no-op面积阈值的输入。better/tie/worse分别按冻结delay/runtime容差判断；下表九格关系互斥，未按结果挑子集。

对A，共同有效可行分母826：

```json
{
  "better_delay__worse_time": 479,
  "tie_delay__worse_time": 347
}
```

对B，共同有效可行分母826：

```json
{
  "better_delay__worse_time": 328,
  "worse_delay__worse_time": 313,
  "tie_delay__worse_time": 184,
  "better_delay__better_time": 1
}
```

## 实际设备、调用与主要成本

| 方法 | CEC调用 | Mapping调用 | 原生replay命令 | 失败候选请求 | repair请求 | 梯度设备/最终来源 |
|---|---:|---:|---:|---:|---:|---|
| A | 905 | 905 | 0 | 0 | 0 | {'cpu': 905} / {'original': 905} |
| B | 901 | 901 | 0 | 0 | 0 | {'cpu': 901} / {'abc_single_native_rewrite': 901} |
| Ours | 4284 | 3944 | 2312 | 2 | 0 | {'cuda': 806, 'none': 94, 'cpu': 5} / {'mffc_global_gradient': 415, 'original_fallback': 131, 'replay_noop': 359} |

Ours各inclusive阶段：

| 阶段 | 有记录方法点 | 中位 s | P90 s | 总inclusive s |
|---|---:|---:|---:|---:|
| original_cec_mapping | 905 | 1.8467 | 5.2453 | 2642.9243 |
| mffc_partition | 905 | 0.0661 | 0.2912 | 139.7359 |
| candidate_enumeration | 905 | 1.3981 | 4.6086 | 2088.2577 |
| boolean_validation | 905 | 0.0727 | 1.3311 | 498.9865 |
| real_replay_cec_mapping | 905 | 5.0654 | 14.3137 | 7184.5525 |
| candidate_reuse_preflight | 811 | 0.0974 | 0.5052 | 199.2836 |
| litecut_preparation_inference | 811 | 1.4705 | 12.1419 | 4388.5347 |
| global_model_preparation | 811 | 1.1659 | 1.7647 | 1081.6363 |
| global_gradient_and_joint_blocks | 811 | 0.0522 | 0.1965 | 79.0822 |
| real_batch_replay_cec_mapping | 778 | 6.5294 | 18.2070 | 8351.2348 |
| gpu_or_cpu_proxy_calibration | 777 | 0.0671 | 0.2824 | 105.8436 |

阶段和工具调用可嵌套/并发，不能相加当完整wall或直接换算独占百分比。整批本次driver elapsed为33434.4075s；与单电路cold latency分开。断点多session的总elapsed需结合各Slurm记录。

已中断且无完整观察的内部请求：{'A': 0, 'B': 0, 'Ours': 0}；损坏/不完整的失败命令文件：{'A': 0, 'B': 0, 'Ours': 0}。这些保持显式缺失，不补造成成功或零成本。

## 图与完整数据

![delay_relative_A_B](figures/delay_relative_A_B.png)

[导出PDF](figures/delay_relative_A_B.pdf)

![runtime_relative_A_B](figures/runtime_relative_A_B.png)

[导出PDF](figures/runtime_relative_A_B.pdf)

![quality_runtime_by_size](figures/quality_runtime_by_size.png)

[导出PDF](figures/quality_runtime_by_size.pdf)

逐电路长表：`BENCHMARK905_RESULTS.csv`；完整统计与分母：`BENCHMARK905_SUMMARY.json`；逐点核验索引：`audited_records.json`；原始证据hash：`evidence_hashes.json`。

## 解释边界

本轮是固定小预算、一个预定seed和既有905集合的真实结果，不是未见集或最优性证明。可微梯度只属于固定原图代理，不经过Student或真实mapper；保留final best也可能来自repair/no-op/付费original。A/B与Ours的成本不同，不能把整体相对速度当作纯GPU加速。没有布局布线后signoff，也没有在本轮加入random/greedy/旧版或joint扩展。

## 完整安装、修复和实际DAG计算补充

完整安装PASS/FAIL/UNKNOWN为{'PASS': 3039, 'FAIL': 2, 'UNKNOWN': 0}；其中非空完整组合2134个。失败部分安装中确实接受0个rewrite，仅计实际工作量，不给这些部分电路成功timing标签。拒绝状态：{}。

失败请求已有2个时钟，inclusive合计6.5457s；repair已有0个，inclusive合计0.0000s。缺终态观察0个。共享replay/并发验证会重叠，这些不是独占wall占比。

| 比较 | 共同可行分母 | delay更好且不更慢 | 更快且delay不差 | 更快但质量下降 | 更慢但质量改善 |
|---|---:|---:|---:|---:|---:|
| A | 826 | 0 | 0 | 0 | 479 |
| B | 826 | 1 | 1 | 0 | 328 |

前两列可能重叠；原九格关系保持不变。

原始正式作业65245为COMPLETED/0:0，Slurm elapsed=34427s，资源billing=4,cpu=4,gres/gpu=1,mem=64G,node=1；它含套件准备/分析，不是单方法latency，也不等于所有并发阶段之和。

有实际CUDA原生传播的Ours方法点806个。下表仅描述该子集的结构与事件成本；不构成同算法CPU/GPU因果对照。

| 结构字段 / 区间 | 方法点 | MFFC中位数 | DAG层数中位 | 最大同层宽度中位 | 原生GPU内核合计中位ms | 完整E2E中位s |
|---|---:|---:|---:|---:|---:|---:|
| region_dag_layers/0_1 | 0 | N/A | N/A | N/A | N/A | N/A |
| region_dag_layers/1_32 | 449 | 2900.0000 | 8.0000 | 1504.0000 | 51.8769 | 17.0149 |
| region_dag_layers/32_512 | 315 | 3465.0000 | 72.0000 | 509.0000 | 83.2184 | 22.6444 |
| region_dag_layers/512_4096 | 36 | 8356.5000 | 1088.5000 | 255.0000 | 384.6441 | 43.6351 |
| region_dag_layers/4096_plus | 6 | 37383.0000 | 4573.0000 | 2378.0000 | 1583.4394 | 120.2535 |
| max_layer_width/0_32 | 5 | 1437.0000 | 263.0000 | 25.0000 | 87.9073 | 19.6046 |
| max_layer_width/32_512 | 282 | 2174.5000 | 65.5000 | 232.5000 | 53.2347 | 18.2817 |
| max_layer_width/512_8192 | 465 | 4144.0000 | 12.0000 | 1311.0000 | 77.7400 | 20.2567 |
| max_layer_width/8192_plus | 54 | 13585.5000 | 6.5000 | 8951.0000 | 275.0167 | 41.5869 |

原生事件不包括其他Torch聚合、上下文初始化、CPU特征/Student和EDA；不同调用的batch可不同。结构分组为描述性补充，来源/AND主分组与所有胜负/时间容差仍使用冻结协议。完整逐点安装明细见`installation_details.json`，完整设备/结构分组见SUMMARY的`delivery_supplement`。


## CEC、Mapping与Replay的实际记录成本

包含成功、失败及超时方法点内已保存的命令。每点时间为该类命令的inclusive和；有并发验证时会重叠，不能相加当独占wall占比。时间n只统计有记录的点，未出现该工具或时钟缺失不补零。调用数为已保存命令数，损坏命令文件另列。

| 方法 / 工具 | 已记录调用数 | 有时间方法点 | 每点inclusive中位s | P90 s | 总inclusive s | 有调用但无时钟点 |
|---|---:|---:|---:|---:|---:|---:|
| A/cec | 905.0000 | 905 | 0.0641 | 0.1642 | 89.4409 | 0 |
| A/mapping | 905.0000 | 905 | 1.8172 | 5.2026 | 2569.3783 | 0 |
| A/rewrite | 0.0000 | 0 | N/A | N/A | N/A | 0 |
| A/native_rewrite | 0.0000 | 0 | N/A | N/A | N/A | 0 |
| B/cec | 901.0000 | 901 | 0.4647 | 2.2177 | 1694.6541 | 0 |
| B/mapping | 901.0000 | 901 | 1.7165 | 4.5203 | 2320.2538 | 0 |
| B/rewrite | 0.0000 | 0 | N/A | N/A | N/A | 0 |
| B/native_rewrite | 905.0000 | 905 | 0.1642 | 0.4145 | 215.8255 | 0 |
| Ours/cec | 4284.0000 | 905 | 1.0865 | 4.2685 | 2735.9113 | 0 |
| Ours/mapping | 3944.0000 | 905 | 7.9428 | 24.0457 | 11307.9938 | 0 |
| Ours/rewrite | 2312.0000 | 905 | 1.0468 | 2.9723 | 1431.6098 | 0 |
| Ours/native_rewrite | 0.0000 | 0 | N/A | N/A | N/A | 0 |

未读/损坏的命令文件数：{'A': 0, 'B': 0, 'Ours': 0}。完整冷wall使用主表，不以这些重叠分项代替。
