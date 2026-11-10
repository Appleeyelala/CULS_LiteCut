# Gate 2B 交接文档

## 当前状态

已完成并通过：Gate 1A、Gate 1B、Gate 1C-R、Gate 2A mapping-free LiteCut Student bridge、Gate 2B。

当前 Gate 2B 已停止。除非用户明确要求，不要修改 Gate 1C-R、candidate 语义或 `best.pt`。

主仓库：`/home/wushangge/cutLibOrigin`  
Gate 1C-R：`/data/wushangge/gate1a_AIGRewrite/gate1c_r_out/`  
Gate 2B 产物：`/home/wushangge/cutLibOrigin/gate2b_out/`  
Slurm execution workspace：`/data/wushangge/gate2b_runs/gate2b_20260917/`

成功 Slurm job：`44300`。

## Gate 1 decisions

```text
d1 = (root=146, cut_slot=0)
d2 = (root=226, cut_slot=0)
d1 boundary = [2, 42, 82]
d2 boundary = [3, 4, 146, 178]
146 ∈ boundary(d2)
d1 → d2
```

Candidate 语义必须保持：

```text
A = d1 subgraph 0 = NO_OP_FULL_CORRESPONDENCE
B = d1 subgraph 1 = PHYSICAL_REPLACEMENT
C = d2 subgraph 0 = PHYSICAL_REPLACEMENT
D = d2 subgraph 1 = PHYSICAL_REPLACEMENT
```

四个联合配置 `AC/AD/BC/BD` 都从同一 normalized original snapshot 独立生成，且已有 Gate 1C-R CEC PASS。不要把 A 改称 physical candidate。

## LiteCut Student

Repo：`/home/wushangge/cutLibOrigin`  
Commit：`077aa1e49b8b97ed6d76013a7272c06bfacabf23`  
Checkpoint：`/home/wushangge/cutLibOrigin/best.pt`  
SHA256：`b3f499d2973bdb36fb42690794e9838d4c8359f05d6e141aa8cbe2b9fe3ae64d`

Model family：`student_group_fusion_soft_moe`，continuous input dimension 19。LiteCut 数值只能叫 `LiteCut-predicted local abc_delay`，不能叫 whole-design/global delay。

```text
A = 29.53646804117542
B = 24.08421517952560
C = 20.574257743484107
D = 35.493827015539594
```

因此 local greedy：`d1→B, d2→C, local_greedy=BC`。禁止把 local scores 相加称为 global timing model。

## 固定 ABC mapping/STA flow

ABC：`/data/public/AIGRewrite/abc/abc`  
版本：`UC Berkeley ABC 1.01 (compiled Aug 6 2025 14:07:48)`  
Library source：`/home/wushangge/cutLibOrigin/lib/asap7.lib`  
Execution copy：`/data/wushangge/gate2b_runs/gate2b_20260917/input/asap7.lib`  
Library SHA256：`819ecfc2609a1f53b9d298bf0cba75af0f1a8c737f33d4bbc5c9fb1156e8fa55`

精确 command：

```text
read "<technology_library>";
read "<input_aig>";
map;
topo;
stime
```

whole-design delay 取 canonical ABC 输出的 `Area = ... Delay = ... ps` 行；不要混入额外 `print_stats` 输出。完整配置见 `flow.abc`、`environment.json`、`library_manifest.json`。

## 实测结果

| configuration | whole-design delay | area | mapped gates |
|---|---:|---:|---:|
| original | 263.32 ps | 557.07 | 432 |
| AC | 263.60 ps | 557.54 | 435 |
| AD | 263.60 ps | 554.74 | 435 |
| BC | 263.60 ps | 556.84 | 436 |
| BD | 263.60 ps | 554.74 | 435 |

四个 configuration 均重复运行两次，delay、area、gate count 完全一致。`repeatability noise = 0.00 ps`；ABC 输出 precision 为 `0.01 ps`，比较 tolerance 使用 `0.01 ps`。

## Interaction 结果

```text
Delta1(C) = D_BC - D_AC = 0.00 ps
Delta1(D) = D_BD - D_AD = 0.00 ps
Delta2(A) = D_AD - D_AC = 0.00 ps
Delta2(B) = D_BD - D_BC = 0.00 ps
I = D_AC + D_BD - D_AD - D_BC = 0.00 ps
```

两个 identity 也都成立：`I = Delta1(D)-Delta1(C)`，`I = Delta2(B)-Delta2(A)`。

没有 d1 或 d2 context-dependent ranking reversal。四个配置在 0.01 ps tolerance 内并列最低，因此 BC 是 lowest measured set 中的一员，但不是 unique winner。当前 pair 没有可测的 global timing interaction；不能据此声称一般意义上的 local greedy global optimality，也不能排除低于 0.01 ps 的差异。

## 五个问题的答案

1. LiteCut local greedy 是否仍是 BC？**是。**
2. 真实 delay：`D_original=263.32 ps`，`D_AC=D_AD=D_BC=D_BD=263.60 ps`。
3. BC 是否最低？**与 AC/AD/BD 并列最低 measured delay。**
4. 是否 ranking reversal？**d1=false，d2=false。**
5. interaction 是否超过 noise/precision？**I=0.00 ps，不超过。**

## 关键产物

```text
gate2b_report.md
gate2b_summary.json
interaction_analysis.json
gate2b_results.csv
original_sta.json
AC_sta.json
AD_sta.json
BC_sta.json
BD_sta.json
AC_sta_run2.json
AD_sta_run2.json
BC_sta_run2.json
BD_sta_run2.json
original.log
AC.log
AD.log
BC.log
BD.log
repeat_BC_run1.log
repeat_BC_run2.log
AC_cec.log
AD_cec.log
BC_cec.log
BD_cec.log
flow.abc
environment.json
library_manifest.json
gate2b.slurm
gate2b_runner.py
```

## 下一次对话的限制

不要训练 LiteCut、恢复 Teacher、重建 dataset、修改 checkpoint、实现 gradient/softmax/global predictor/timing aggregator、扩展新 region、替换 A/B/C/D、切换 library/ABC binary/flow，除非用户明确开启新的 Gate。

新对话建议先读取本文件、`gate2b_report.md` 和 `gate2b_summary.json`。如果没有新的任务，Gate 2B 应保持结束状态。
