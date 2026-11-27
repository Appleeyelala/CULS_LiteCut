# Stage B (`rw4b-grad`) 代码审查与交接文档（给 Claude）

> **审查范围**：严格只读审计（不改代码、不跑命令）。请对照设计蓝图 [stage_b_gpu.md](./stage_b_gpu.md) 与以下 3 个核心源码文件，检查是否存在数学/拓扑/并行 Bug、与蓝图思路的偏差，以及全量 905 电路扩展前的潜在风险或性能瓶颈。
> **当前分支与基线**：`work/4b-review-opt` @ `b9a3df05`；Stage B-1 已通过双精度中心差分梯度检验（`c1355`/`c5315`/`intra_pred_PE` 绝对误差 `6.8e-9 ~ 5.4e-8`），并在 16 电路面板上取得 Delay `-47.00%`、Area `-3.13%`、ADP `-48.66%`、`16/16 cec = PASS`（对比 Stage A：Delay `-44.22%`、Area `+1.64%`、ADP `-43.30%`）。

---

## 一、精准源码导航（节省阅读额度）

| 文件 | 关键行号区间 | 功能说明 |
|---|---|---|
| `projects/culs/src/algorithms/rw4b_grad.h` | `L1–L80` | `SubGate`、`GradProblem`、`GradSolveStats` 与 `solveJointGradGPU` 接口 |
| `projects/culs/src/algorithms/rw4b_grad.cu` | `L23–L112` | 光滑 `softplusF`/`sigmoidF`/`lse2W`、PI 相位读写 `readLitSmooth`/`readLitHard`/`addLitGrad` |
| `projects/culs/src/algorithms/rw4b_grad.cu` | `L113–L345` | `k_init_exp_fo`、`k_softmax_and_exp_fo`、`k_discretize_and_exp_fo`、`k_fwd_level`、`k_fwd_level_hard` |
| `projects/culs/src/algorithms/rw4b_grad.cu` | `L346–L598` | `k_extract_po_arr`、`k_po_loss_stats`、`k_po_grad_init`、`k_aux_loss_terms`、`k_bwd_level`、`k_fanout_grad`、`k_region_grad_and_adam` |
| `projects/culs/src/algorithms/rw4b_grad.cu` | `L600–L1140` | `solveJointGradGPU`：显存管理、Warm-Start、`--grad-check`（`L811–L995`）、3 阶段退火主循环、8 档多阈值离散寻优（`L1060–L1115`） |
| `projects/culs/src/algorithms/rw4b.cu` | `L1154–L1560` | `evaluateNode4BMenuKernel`：分层预筛、4 槽位帕累托菜单、`slack_other` 守卫、`level(hit) < level(root)` 匹配 |
| `projects/culs/src/algorithms/rw4b.cu` | `L2272–L2825` | `if (params.jointGrad)` 主控：Host 菜单验证与微子图编码、`L2562` 线性 GWMIN 与 4 条不变式、`L2699` 置信度降序实时扇出堆叠守卫、`StrashRebuild (--fo-cap 0)` |

---

## 二、当前实现与蓝图 `stage_b_gpu.md` 的主动调整（已验证动机）

在 16 电路调试过程中，有 4 处相对初始蓝图做了针对性调整，请重点评估其数学合理性与副作用：

1. **关闭二次超额扇出正则（`lamFo = 0.0f`，`rw4b_grad.cu:L718`）**：
   - 原因：在 `sram_64`/`dmem` 等内存电路上存在 1,000~4,000 扇出的控制网线，二次项 `(fo - 6)^2 / |V0|` 爆炸至 `115.0`，彻底淹没 WNS/TNS（`~2.0`）。当前完全依赖通路 2 的精确物理扇出/Slew 偏导 `g_fo(u)` + 菜单/后置 `slack_other` 护栏。
2. **贴近原图初始化（`z_{m,0} = 2.5f`，`rw4b_grad.cu:L693`）**：
   - 原因：原 `0.25` 均匀初始化会让 4~5 万个活跃区在 `t=1` 同时开启 ~75%，共享网线扇出瞬间叠加导致 `initLoss` 暴涨至 `114.85`（57× $D_0$）。改为 `z_{m,0} = 2.5f`（$p(\text{no-op}) \approx 0.75\sim0.92$）配合 `lr = 0.20` 后消除 `t=1` 爆炸。
3. **增加噪声候选预筛（`rw4b.cu:L1547`）**：
   - `if (adpScore <= 1e-9f && !(inVision && deltaJ > 0.5f)) continue;`，避免 `sram_64` 等大电路将数万个微弱负分区域塞入 Host 打包与 GWMIN。
4. **增加离散化后“置信度降序实时扇出堆叠守卫”（`rw4b.cu:L2699–L2756`）**：
   - 在 `solveJointGradGPU` 选出离散方案 `chosenCand` 后，按 `chosenProb` 降序用 `accFo + querySlackOtherHost` 逐个核验并剔除会造成共享网线扇出堆叠超限的区域（此项将 `reservation_station` 从 `64,895 ps` 压降至 `3,773 ps`）。

---

## 三、请重点帮我们核查的 5 个潜在疑点

1. **疑点 A（后置扇出守卫与 GPU 离散寻优解耦的隐患，`rw4b.cu:L2699–L2756`）**：
   - GPU 多阈值离散寻优（`rw4b_grad.cu:L1060`）选出 `bestTheta` 时并未考虑后置守卫的拦截；随后 Host 守卫踢掉部分区域后，**会不会反而把原本靠多区域抵消扇出的“正负配对”拆散**（例如 A 区域减了节点 `u` 的扇出、B 区域加了 `u` 的扇出，若按 `chosenProb` 排序 B 先于 A 接受而被误杀，或 A 被踢后 B 反而变差）？且守卫过滤后没有再重验离散代理 `dHard <= depth0`。
2. **疑点 B（`sirv_sim_ram` 与 `branch_prediction_globle` 在 `66415` 中回退）**：
   - 在 `66414`（无后置守卫）中，`sirv_sim_ram` 为 `15,240 ps`、`branch_prediction_globle` 为 `11,137 ps`；加上后置守卫后（`66415`），这两张反而退化为 `60,041 ps` 与 `27,549 ps`（而 `reservation_station` 大幅好转 `64,895 → 3,773 ps`）。怀疑是否与疑点 A 的顺序敏感/误杀有关（例如高扇出减载区排在加载区后面，或 `budgetTol = 1e-4` 过严拦截了本可整体降延迟的协同改写）？
3. **疑点 C（前向/反向与离散重建之间是否还存在微小的语义偏差）**：
   - 核查 `rw4b.cu:L2460–L2525` 的 `SubGate` 编码、`foDeltas` 记账（特别是 `match[rt] != -1` 整棵子图直接命中已有门的情况、`rt < 4` 导线折叠的 `flip` 与负相 PI 扇出）是否与 `StrashRebuild`（`L2770–L2800`）100% 一致；
   - 核查 `k_fwd_level` 中普通节点 `remStrict` 保持 `foBase` 不变时，对 `k=0` 和 `k>=1` 的梯度及离散评估 `k_fwd_level_hard` 是否存在任何遗漏。
4. **疑点 D（3 阶段退火与 `z_{m,0} = 2.5` 的动力学边界）**：
   - 当 `z_{m,0} = 2.5`、`tauP: 1.0 -> 0.45` 时，在 `steps = 25` 内，弱梯度但有益的候选（或非关键区纯省面积 `da < 0` 候选）是否来得及翻过 `2.5` 门槛？或者 `Gini` 磁铁在早期/中期是否会反而把还没翻过 `2.5` 的好候选往 `no-op` 吸回去？
5. **疑点 E（全量 905 电路的 Host 端性能瓶颈，`rw4b.cu:L2313–L2760`）**：
   - 在 8.3 万门的 `sram_64` 上，GPU 25 步联合梯度仅耗时 `29.5 ms`，而 `cpu_sel_rebuild` 高达 `~900 ms`（3 轮合计 `1,151 ms`）。请指出最值得优先优化的 2~3 处 Host 热点（如 `origStrash`、`collectBoundedMffcHost`、`std::vector` 重复分配、或不必要的 D2H 拷贝），以便全量 905 电路保持 `100×` 以上加速比。

---

## 四、给 Claude 的精简只读审查提示词（可直接复制）

```markdown
请以只读审查模式（不要改代码、不要运行命令、不要寒暄）审查 Stage B 实现。
先阅读 `docs/StageB代码审查与交接文档_给Claude.md`，再按其中的精确行号对照检查 `projects/culs/src/algorithms/rw4b_grad.h`、`projects/culs/src/algorithms/rw4b_grad.cu` 与 `projects/culs/src/algorithms/rw4b.cu` 的 Stage B 代码段。
请按 P0（正确性/严重效果回退）/ P1（动力学或建模缺陷）/ P2（Host 加速与细节）分级输出：
1. 逐项回应交接文档中的疑点 A–E，并指出我们未注意到的真实 Bug 或与 `docs/stage_b_gpu.md` 思路的偏差；
2. 针对每个问题给出最小改动方案与预期影响（Delay / Area / ADP / Runtime）。
```
