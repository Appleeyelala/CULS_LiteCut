# Stage B 开发蓝图：GPU 原生多区域端到端可微梯度联合优化（`rw4b-grad`）

> **文档定位**：本蓝图面向 `CULS-LiteCut (4B)` 项目的核心范式跃迁阶段（Stage B），在已封存验证的 Stage A GPU 底座（全量 905 电路 ASAP7 Delay `-10.22%`、Area `+3.73%`、ADP `-6.88%`、`$313.9\times$` 核心加速、`905/905` 零崩溃、`904/904` 完成电路 `cec` 通过）之上，给出**多区域、多路径端到端可微梯度联合优化器**的总体架构、精确数学建模、CUDA 工程布局与分阶段验收标准。

---

## 一、大方向与核心架构定位

### 1. 我们要解决的核心问题（范式跃迁）

* **Stage A 已达成的底座**：纯 GPU `rw4b`（12 槽位混合 4-cut + 完整内部锥 `coneIds` 时序评估 + 规模无关 ADP 一阶打分 + `--fo-cap 0` 对齐 ABC 语义）已在全量 905 电路上全面超越 CPU 版与 ABC 单命令/脚本基线。
* **Stage A 的算法天花板**：目前每轮本质仍是**“一阶单步敏感度 + 孤立贪心排队”**——评估某个局部改写时，默认全图其他节点静止不动。当成百上千个改写同时回装时，上下游路径之间的高阶非线性竞争、多区域同时修改同一共享叶子的扇出/Slew 叠加效应，以及“单步略亏但双区联动大赚”的协同改写，无法在当轮决策中联合求解。
* **Stage B 的破局方向**：将后端物理设计中验证成功的**端到端可微时序梯度优化（Guo & Lin, DAC 2022）**、**离散选择连续松弛与离散化磁铁（Du et al., ICCAD 2024）**以及**冲突图独立集调度（Riener et al., DATE 2019）**系统性迁移到大规模 AIG 逻辑重写中。用**“外层 3 轮圈地备菜单 + 内层 GPU 连跑 20 至 30 步双通路精确梯度协调 + 多阈值离散寻优”**取代单步贪心，让全图活跃区域在多条 PI→PO 路径（WNS + TNS）、扇出/Slew 负载与面积约束之间同步协商出全局最优配合。

### 2. 不可动摇的工程底线

* **零破坏兼容**：在 `rw4b.cu` 中以独立 CLI 开关（如 `--joint-grad --steps 25`）接入 Stage B，**完整保留 Stage A 默认入口与行为**，确保严格同口径消融对比。
* **沿用已验证的 Stage A 物理与重建结论**：
  * 严格保持 **ADP 物理无量纲归一化**（延迟按当轮原图基准 $D_0$ 归一、面积与离散化磁铁按当轮门数 $|V_0|$ 归一），杜绝超参随电路规模失灵或引发面积暴涨。
  * 保持主输入正负相位区分（`pi_phase_aware`）与结构哈希复用感知（`strash_aware`）。
  * 离散回装阶段沿用 **`--fo-cap 0`** 的干净 `StrashRebuild`（严格对齐 ABC `read_aiger` 自动 strash 语义，不引入无效重复门复制）。
* **等价性与盲测红线**：所有电路必须 100% 通过 ABC `cec -n` 等价验证；**24 电路外部盲测集（`panel.json`）在全部算法与超参最终锁死前严格封存**。

### 3. 总体执行架构（快慢双循环）

借鉴 DAC 2022 的“外层离散结构更新 + 内层冻结区域拓扑连跑梯度”思想，总体流程如下：

```text
[Input AIG Circuit]
         |
         v
======================================================================================
                 Outer Slow Loop (3 Rounds: Outer Round 1 -> 2 -> 3)

  [Step 1] Stratified Pre-filter, Conflict-Free Region Partition & Pareto Menu
    * Reuse Stage A 12-slot hybrid 4-cut engine & full-cone timing
    * Stratified eligibility filter (near-critical/TNS cone keeps speed+area cands;
      deep off-critical keeps only da <= 0 area-saving cands; empty menus dropped)
    * Select conflict-free regions {R_1, ..., R_M} via GWMIN (strict remStrict /
      leaves / hitIds protection, level(hit) < level(r_m), wire-fold chain guard)
    * Keep K_m <= 5 Pareto-diverse candidates per region (including k=0 no-op)
    * Build levelized DAG & sparse external fanout-delta CSR table once per round
                                         |
                                         v  (freeze regions & compact micro-subgraphs)
  +--------------------------------------------------------------------------------+
  | [Step 2] GPU Inner Joint Gradient Loop (T = 20~30 steps, ~50~150 ms total)     |
  |                                                                                |
  |   for t = 1 ... T:                                                             |
  |     1. Prob & Smooth FO : Softmax(z / tau_p) -> p; CSR sum -> E[fo], SoftPlus  |
  |     2. Exact Micro-Fwd  : levelized fwd; k=0 live orig gate, k>=1 live NPN DAG |
  |     3. Multi-Obj Loss   : ADP-scaled TNS + WNS + Area + ExcessFO + Gini Magnet |
  |     4. Exact Dual Bwd   : [Path 1: Arrival & Slew] + [Path 2: Weighted FO/PI]  |
  |     5. Safe Adam Step   : Softmax Jacobian -> step-clipped Adam + annealing    |
  +--------------------------------------------------------------------------------+
                                         |
                                         v  (multi-threshold discrete verification)
  [Step 3] GPU Multi-Threshold Discrete Line Search & StrashRebuild
    * Sweep confidence thresholds theta in {0.50, 0.65, 0.80, 0.90, 0.95} with
      exact hard-max discrete forward (~3 ms); pick best discrete ADP (or early-stop)
    * Run verified StrashRebuild (--fo-cap 0)
======================================================================================
         |
         v
[Output Optimized AIG + ABC cec -n Equivalence Verification]
```

---

## 二、核心算法与精确数学设计（5 大模块）

### 模块 1：外层分层预筛、无冲突活跃区域划分与帕累托候选池

* **工程直觉**：先按物理可达性筛掉“既不能省门、又远离关键路径”的无效候选，再圈出一批互不破坏删除集与复用门的活跃区域，并给每个区域配备最多 5 种互补候选方案，让内层联合梯度既有协同空间又不白占独立集名额。

#### 1. 分层物理可达性预筛（防止非关键纯加门节点抢占独立集）
设当轮初始硬传播最大延迟为 $D_0$，TNS 目标水位为 $T_{\text{target}} = \rho D_0$（$\rho \in [0.80, 0.90]$）：
* **近临界 / TNS 视野内节点**（满足 $\text{slack}(r) = \text{req}(r) - a(r) \le \max(\kappa_{\text{pool}}\tau,\; (1 - \rho)D_0)$，如 $\kappa_{\text{pool}} = 3.0$）：**放宽准入**，保留所有局部平滑提速（$\Delta J > 0$）或省面积（$\Delta\text{AND} \le 0$）的合法候选——即使某个候选在 Stage A 单步静态评估下因保守 `side` 略呈负分，只要有提速潜力也允许入池，交由内层多区域联合梯度裁决是否能够协同翻盘。
* **视野外纯非关键节点**：内层时序偏导接近 $0$，增面积候选必不可能胜出；**直接剔除所有 $\Delta\text{AND} > 0$ 候选**，仅保留 $\Delta\text{AND} \le 0$ 的省面积候选。
* 若某节点过滤后仅剩 `no-op`（$K_m = 1$），直接不进入 GWMIN 冲突图，绝不阻塞周边的省面积改写。

#### 2. 严格区域无冲突与并行无竞态规则（对齐 Stage A P0-1 教训）
对候选区域 $R_m$（根节点为 $r_m$），严格区分两个节点集合：
* **严格删除内部集** $\text{remStrict}(R_m) = \bigcup_{k=1}^{K_m-1} \text{remStrict}_{m,k}$（改写后将被移除的 MFFC 内部门，**不含根节点 $r_m$**）；
* **全改写集** $\text{remAll}(R_m) = \text{remStrict}(R_m) \cup \{r_m\}$；
* **外部引用集** $\text{Refs}(R_m) = \bigcup_{k=1}^{K_m-1} \big(\text{Leaves}_{m,k} \cup \text{Hits}_{m,k}\big)$（所有非平凡候选用到的 4-cut 叶子与结构哈希复用门 `hitIds`）。

任意两个同时入选的活跃区域 $R_i \neq R_j$ 必须满足以下 **4 条硬性不变式**：
1. **改写区互不重叠**：$\text{remAll}(R_i) \cap \text{remAll}(R_j) = \emptyset$（特别是 $r_i \neq r_j$）；
2. **叶子与复用门双向保护**：$\text{remStrict}(R_i) \cap \text{Refs}(R_j) = \emptyset$ 且 $\text{remStrict}(R_j) \cap \text{Refs}(R_i) = \emptyset$（保证没有任何候选的输入叶子或 strash 命中门被其他区域删除）；
3. **同拓扑层竞态消除**：候选内部允许复用的结构哈希命中节点 $h \in \text{Hits}_{m,k}$ 必须严格满足单位拓扑层约束 $\text{level}(h) < \text{level}(r_m)$（由于 cut 叶子天然满足 $\text{level}(l) < \text{level}(r_m)$，此约束彻底消除 GPU 按层并行核函数的同层读写竞态与组合环路）；
4. **上下游串联与导线折叠守卫**：允许上游活跃根 $r_1$ 直接作为下游活跃区域 $R_2$ 的输入叶子或复用门（$r_1 \in \text{Refs}(R_2)$），实现多级路径内层实时联动；但**若 $r_1$ 被任何下游活跃区域引用（$r_1 \in \bigcup_{j} \text{Refs}(R_j)$），则从 $R_1$ 的菜单中剔除“0 门导线折叠（$rt < 4$）”候选**（导线折叠占比 $<1\%$，剔除后彻底消除“转移扇出数随下游选择变化”的 $p_1 p_2$ 双线性交叉项，使静态线性 CSR 扇出表 100% 精确）。

#### 3. GWMIN 选区与帕累托多样化候选菜单（$K_m \le 5$）
* **GWMIN 贪心独立集选区（DATE 2019）**：对预筛通过的候选节点统计冲突度 $d(m)$，按修正潜力分 $\frac{W(m)}{1 + d(m)}$ 降序选出无冲突活跃区域集 $\{R_1, \dots, R_M\}$。
* **帕累托菜单（去重后每区域 $2 \le K_m \le 5$ 个候选，允许来自该节点 12-slot 混合 cut 中的不同 4-cut）**：
  * **$k = 0$（保底项）**：`no-op`（保持原结构不变，保证下界）；
  * **$k = 1$（纯速项）**：局部平滑降层 $\Delta J$ 最大的候选；
  * **$k = 2$（均衡项）**：Stage A 规模无关 ADP 综合分最高的候选；
  * **$k = 3$（省门项）**：净门数变化 $\Delta\text{AND}_{m,k}$ 最小且不显著恶化局部延迟的候选；
  * **$k = 4$（低扇出项）**：对外部引用节点扇出正增量 $\sum_u \max(0, \Delta\text{fo}_{m,k}(u))$ 最小的提速候选。

---

### 模块 2：连续概率松弛、光滑扇出负载与实时微子图按层前向传播

#### 1. 概率松弛映射
为每个活跃区域 $R_m$ 的候选 $k \in \{0, \dots, K_m - 1\}$ 维护连续 logits $z_{m,k} \in \mathbb{R}$，经温度 $\tau_p$ 的 Softmax 得到选择概率：

$$p_{m,k} = \frac{\exp(z_{m,k} / \tau_p)}{\sum_{j=0}^{K_m - 1} \exp(z_{m,j} / \tau_p)}, \qquad \sum_{k=0}^{K_m - 1} p_{m,k} = 1$$

#### 2. 全图期望扇出汇总（对齐 `pi_phase_aware`、无内部自污染、光滑可导）
* **相位感知统一存储**：长度为 `nObjs` 的数组 `d_expFo[u]` 按节点类型赋予精确物理含义：
  * 对**内部 AND 门**（$u > N_{\text{PI}}$）：维护总期望扇出 $\mathbb{E}[\text{fo}(u)]$（延迟系数 $\alpha$）；
  * 对**主输入 PI**（$1 \le u \le N_{\text{PI}}$）：直接维护反相器驱动的**期望负相位扇出 $\mathbb{E}[\text{negFo}(u)]$**（延迟系数 $\alpha_{\text{pi}}$；正相 PI 引用边延迟与 Slew 恒为 $0$），零额外数组 100% 对齐 Stage A 的 `pi_phase_aware`。
* **防自污染记账边界**：稀疏 CSR 表中的 $\Delta\text{fo}_{m,k}(u)$（$k \ge 1$，$\Delta\text{fo}_{m,0}(u) \equiv 0$）**仅统计对外部存活节点 $u \notin \text{remStrict}(R_m)$（即 `leaves` 与 `hitIds`）的扇出净增减**，不对将被删除的内部节点 $v \in \text{remStrict}(R_m)$ 扣减扇出。这样当 $p_{m,k>0} \in (0,1)$ 增大时，`no-op`（$k=0$）所读取的原图内部门不会因扇出缩水而产生“虚假加速”，彻底杜绝阻碍改写的自污染负反馈。
* **光滑可微有效负载（消除 $\text{fo}=1$ 处硬折点）**：AIG 中约 65% 的门处于 $\text{fo}=1$，硬截断 $\max(0, x-1)$ 在 $x=1$ 不可导。内层采用低温 $\text{SoftPlus}_{\tau_{\text{fo}}}$（默认 $\tau_{\text{fo}} = 0.1$）：

$$\mathbb{E}[\text{fo}(u)] = \text{fo}_{\text{base}}(u) + \sum_{m,k} p_{m,k} \cdot \Delta\text{fo}_{m,k}(u)$$

$$\text{load}_{\tau_{\text{fo}}}(u) = \text{SoftPlus}_{\tau_{\text{fo}}}\!\big(\mathbb{E}[\text{fo}(u)] - 1\big) = \tau_{\text{fo}} \log\!\Big(1 + \exp\!\big((\mathbb{E}[\text{fo}(u)] - 1) / \tau_{\text{fo}}\big)\Big)$$

其一阶导数 $\sigma(u) = \text{sigmoid}\!\big((\mathbb{E}[\text{fo}(u)] - 1)/\tau_{\text{fo}}\big) \in (0,1)$ 全程 $C^\infty$ 光滑有界。

#### 3. 实时微子图精确按层前向传播（保证任意 one-hot 顶点与离散图 100% 逐位一致）
定义字面量 $\text{lit} = (u \ll 1) \mid c$ 的边到达时间与边 Slew 辅助映射：
* 若 $u = 0$（常量）：$\tilde{a}(\text{lit}) = 0,\; \tilde{s}(\text{lit}) = 0$；
* 若 $1 \le u \le N_{\text{PI}}$（主输入）：当 $c = 1$（负相边）时 $\tilde{a}(\text{lit}) = \alpha_{\text{pi}} \cdot \text{load}_{\tau_{\text{fo}}}(u),\; \tilde{s}(\text{lit}) = \text{load}_{\tau_{\text{fo}}}(u)$；当 $c = 0$（正相边）时 $\tilde{a}(\text{lit}) = 0,\; \tilde{s}(\text{lit}) = 0$；
* 若 $u > N_{\text{PI}}$（内部门）：$\tilde{a}(\text{lit}) = a(u),\; \tilde{s}(\text{lit}) = s(u)$。

按预建拓扑层 `Level 1 .. L` 顺序执行：
* **普通非活跃节点 $v \notin \mathcal{R}$（含 `remStrict` 内部节点，供 $k=0$ 读取）**：设左右扇入字面量为 $l_0, l_1$：

$$a_{\text{in}}(v) = \text{LSE}_\gamma\!\big(\tilde{a}(l_0), \tilde{a}(l_1)\big), \qquad s_{\text{in}}(v) = \text{LSE}_{\gamma_s}\!\big(\tilde{s}(l_0), \tilde{s}(l_1)\big)$$

$$d_g(v) = 1 + \alpha \cdot \text{load}_{\tau_{\text{fo}}}(v) \cdot \big(1 + \beta \cdot s_{\text{in}}(v)\big), \qquad a(v) = a_{\text{in}}(v) + d_g(v), \qquad s(v) = \text{load}_{\tau_{\text{fo}}}(v)$$

* **活跃区域根节点 $r_m \in \mathcal{R}$（实时微子图评估，无静态路径表失真）**：
  * **$k = 0$（`no-op`）**：直接用原图扇入 $(f_0[r_m], f_1[r_m])$ 计算 $a^{\text{in}}_{m,0}, s^{\text{in}}_{m,0}$，结合外部负载 $\text{load}_{\tau_{\text{fo}}}(r_m)$ 得：
    $$a_{m,0} = a^{\text{in}}_{m,0} + 1 + \alpha \cdot \text{load}_{\tau_{\text{fo}}}(r_m)\big(1 + \beta \cdot s^{\text{in}}_{m,0}\big), \qquad s^{\text{out}}_{m,0} = \text{load}_{\tau_{\text{fo}}}(r_m)$$
  * **$k \ge 1$（替换候选）**：
    * 若为**导线折叠**（直连字面量 $l^*$）：$a_{m,k} = \tilde{a}(l^*),\; s^{\text{out}}_{m,k} = \tilde{s}(l^*)$；
    * 若为 **NPN 微子图**（仅含 1~8 个新建内部门 $j \in \mathcal{V}_{m,k}$，输出门为 $rt$）：子图中复用的 strash 节点 $h$（已保证 $\text{level}(h) < \text{level}(r_m)$）直接读取实时 $(a(h), s(h))$；新建内部非根门 $j < rt$ 按候选内部确定扇出 $e\text{Fo}_j$ 的负载 $\text{load}(e\text{Fo}_j) = \max(0, e\text{Fo}_j - 1)$ 逐门实时递推 $\text{LSE}_\gamma$ 与 $\text{LSE}_{\gamma_s}$；根门 $rt$ 挂载实时外部负载 $\text{load}_{\tau_{\text{fo}}}(r_m)$：
      $$a_{m,k} = a^{\text{in}}_{m,k}(rt) + 1 + \alpha \cdot \text{load}_{\tau_{\text{fo}}}(r_m)\big(1 + \beta \cdot s^{\text{in}}_{m,k}(rt)\big), \qquad s^{\text{out}}_{m,k} = \text{load}_{\tau_{\text{fo}}}(r_m)$$
  * **根节点概率混合输出**：

$$a(r_m) = \sum_{k=0}^{K_m-1} p_{m,k} \cdot a_{m,k}, \qquad s(r_m) = \sum_{k=0}^{K_m-1} p_{m,k} \cdot s^{\text{out}}_{m,k}$$

* **前向权重缓存**：缓存普通门与各候选微子图内每个 2 输入门的到达时间 LSE 权重与 Slew LSE 权重，供反向传播直接读取，避免重算指数。

---

### 模块 3：物理 ADP 量纲对齐的多目标可微损失函数

所有分项严格对齐 $\log(\text{ADP}) = \log D + \log A$ 的物理一阶无量纲微分尺度：

$$\mathcal{L}(\mathbf{z}) = \underbrace{t_1 \cdot \frac{\text{TNS}_\gamma(\mathbf{z})}{D_0} + t_2 \cdot \frac{\text{WNS}_\gamma(\mathbf{z})}{D_0}}_{\text{多路径平滑时序收益}} + \underbrace{\lambda_{\text{area}} \cdot \frac{\sum_{m,k} p_{m,k}\,\Delta\text{AND}_{m,k}}{|V_0|}}_{\text{规模无关面积代价}} + \underbrace{\lambda_{\text{fo}} \cdot \frac{\sum_u \big(\text{SoftPlus}_{\tau_{\text{fo}}}(\mathbb{E}[\text{fo}(u)] - F_{\text{th}})\big)^2}{|V_0|}}_{\text{光滑超额扇出软正则}} + \underbrace{\frac{\lambda_{\text{disc}}}{|V_0|}\sum_{m=1}^{M}\left(1 - \sum_{k=0}^{K_m-1} p_{m,k}^2\right)}_{\text{与单门面积同量纲的 Gini 离散化磁铁}}$$

关键定义说明：
* **$\text{WNS}_\gamma$ 与近临界归一化 $\text{TNS}_\gamma$**：设 $a_{\text{po}} = \tilde{a}(\text{outs}[\text{po}])$：

$$\text{WNS}_\gamma = \text{LSE}_{\gamma_{\text{po}}}\!\big(\{a_{\text{po}}\}\big), \qquad \text{TNS}_\gamma = \frac{1}{|\text{PO}_{\text{norm}}|}\sum_{\text{po}} \text{SoftPlus}_{\gamma_{\text{po}}}\!\big(a_{\text{po}} - T_{\text{target}}\big)$$

其中 $T_{\text{target}} = \rho \cdot D_0$（$\rho \in [0.80, 0.90]$），**分母固定为当轮 $t=0$ 时的初始近临界 PO 数量**：
$$|\text{PO}_{\text{norm}}| = \max\!\Big(1,\; \big|\big\{\text{po} : a_{\text{po}}^{(0)} \ge T_{\text{target}} - \gamma_{\text{po}}\big\}\big|\Big)$$
从而保证即使电路存在数千个快速状态输出引脚，$\text{TNS}_\gamma / D_0$ 始终严格代表“每条近临界路径的平均相对超限量”，绝不被无关快引脚稀释。
* **Gini 磁铁按 $|V_0|$ 归一**：单区域磁铁梯度为 $-\frac{2\lambda_{\text{disc}}}{|V_0|} p_{m,k}$，与单门面积代价 $\frac{\lambda_{\text{area}}}{|V_0|}$ 严格同量纲（例如 $\lambda_{\text{area}}=1.0, \lambda_{\text{disc}}=0.5$ 即代表后期站队吸附力恒等于 1 个 AND 门的面积权重，跨 500 门至 30 万门电路无需重调）。

#### 迭代退火调度表（以单轮 $T = 25$ 步为例，默认总时序权重 $t_1 + t_2 \approx w = 2.0, \lambda_{\text{area}} = 1.0$）

| 迭代阶段 | 步数区间 | $(t_1, t_2)$ 权重分配 | LSE 温度 $\gamma, \gamma_s$ | Softmax $\tau_p$ & 磁铁 $\lambda_{\text{disc}}$ | 核心目的 |
| --- | --- | --- | --- | --- | --- |
| **阶段 1：全局协商期** | $t = 1 \dots 10$ | $(1.7,\; 0.3)$（侧重 TNS） | $1.5\,\tau_0$ | $\tau_p = 1.0,\; \lambda_{\text{disc}} = 0.0$ | 全图自由协商，同步压平多条近临界路径水位 |
| **阶段 2：瓶颈聚焦期** | $t = 11 \dots 18$ | 线性过渡至 $(0.8,\; 1.2)$ | 退火至 $1.0\,\tau_0$ | $\tau_p \to 0.6,\; \lambda_{\text{disc}}$ 线性升至 $0.25$ | 聚焦拔除最慢路径刺头，引导优势候选分化 |
| **阶段 3：离散收网期** | $t = 19 \dots 25$ | $(0.5,\; 1.5)$（侧重 WNS） | 收紧至 $0.5\,\tau_0$ | $\tau_p \to 0.3,\; \lambda_{\text{disc}}$ 升至 $0.60$ | 将摇摆概率平稳吸至近 one-hot，压低离散化落差 |

---

### 模块 4：双通路完整精确链式法则反向传播（确保有限差分误差 $< 10^{-4}$）

#### 1. 通路 1：逆拓扑按层反传到达时间与 Slew 伴随梯度 $\big(g_a(v), g_s(v)\big)$
同时维护主输入负相边的伴随累加器 $\big(g^{\text{neg}}_a(u), g^{\text{neg}}_s(u)\big)_{1 \le u \le N_{\text{PI}}}$：
* **PO 起点**：对每个 PO 计算 $\frac{\partial \mathcal{L}}{\partial a_{\text{po}}}$（由 $\text{WNS}_\gamma$ Softmax 权重与 $\text{TNS}_\gamma$ Sigmoid 权重相加），累加到驱动门 $g_a$（若 PO 为反相直连 PI，则累加到 $g^{\text{neg}}_a$）。
* **逆层扫描 `Level L .. 1`**：
  * **普通节点 $v \notin \mathcal{R}$**：读取前向缓存的到达时间权重 $(w^a_0, w^a_1)$ 与 Slew 权重 $(w^s_0, w^s_1)$。令门内 Slew 耦合敏感度 $c_{\text{slew}}(v) = g_a(v) \cdot \alpha \beta \cdot \text{load}_{\tau_{\text{fo}}}(v)$，向两个输入字面量 $l_b$（$b \in \{0,1\}$）分别分发 $\Delta g_a = g_a(v) w^a_b$ 与 $\Delta g_s = c_{\text{slew}}(v) w^s_b$（内部门累加至 $g_a, g_s$；负相 PI 边累加至 $g^{\text{neg}}_a, g^{\text{neg}}_s$；正相 PI 边为 $0$）。
  * **活跃根节点 $r_m \in \mathcal{R}$**：对每个候选 $k \in \{0, \dots, K_m - 1\}$：
    * 若 $k$ 为**导线折叠**（直连 $l^*$）：向字面量 $l^*$ 分发 $\Delta g_a = g_a(r_m) p_{m,k}$ 与 $\Delta g_s = g_s(r_m) p_{m,k}$；
    * 若 $k$ 为 **`no-op` 或 NPN 微子图**：以根门局部伴随初值 $\delta_a(rt) = g_a(r_m) p_{m,k},\; \delta_s(rt) = 0$ 出发（根门输出 Slew 为 $\text{load}_{\tau_{\text{fo}}}(r_m)$，对内部门 Slew 无直接通路，仅通过根门延迟项产生 $\delta_{s,\text{in}}(rt) = \delta_a(rt)\cdot \alpha\beta \cdot \text{load}_{\tau_{\text{fo}}}(r_m)$），沿微子图的 1~8 个内部门逆序精确反传 $\delta_a, \delta_s$ 至各输入叶子字面量、strash 命中节点 $h$ 及负相 PI 边。

#### 2. 通路 2：汇总全图节点期望扇出总梯度 $g_{\text{fo}}(u) = \frac{\partial \mathcal{L}}{\partial \mathbb{E}[\text{fo}(u)]}$
记负载导数 $\sigma(u) = \text{sigmoid}\!\big((\mathbb{E}[\text{fo}(u)] - 1)/\tau_{\text{fo}}\big)$，超额扇出软正则导数：
$$r_{\text{ex}}(u) = \frac{2\lambda_{\text{fo}}}{|V_0|} \cdot \text{SoftPlus}_{\tau_{\text{fo}}}\!\big(\mathbb{E}[\text{fo}(u)] - F_{\text{th}}\big) \cdot \text{sigmoid}\!\big((\mathbb{E}[\text{fo}(u)] - F_{\text{th}})/\tau_{\text{fo}}\big)$$

分三类节点一次性并行写出精确扇出偏导：
* **普通内部 AND 门（$u > N_{\text{PI}},\, u \notin \mathcal{R}$）**：
  $$g_{\text{fo}}(u) = \sigma(u) \cdot \Big[g_a(u)\cdot \alpha\big(1 + \beta \cdot s_{\text{in}}(u)\big) + g_s(u)\Big] + r_{\text{ex}}(u)$$
* **活跃区域根节点（$u = r_m \in \mathcal{R}$，按非导线折叠候选概率精确加权）**：
  $$g_{\text{fo}}(r_m) = \sigma(r_m) \cdot \sum_{k \notin \text{wire}} p_{m,k}\Big[g_a(r_m)\cdot \alpha\big(1 + \beta \cdot s^{\text{in}}_{m,k}(rt)\big) + g_s(r_m)\Big] + r_{\text{ex}}(r_m)$$
* **主输入 PI（$1 \le u \le N_{\text{PI}}$，对应负相扇出 $\mathbb{E}[\text{negFo}(u)]$）**：
  $$g_{\text{fo}}(u) = \sigma(u) \cdot \Big[g^{\text{neg}}_a(u) \cdot \alpha_{\text{pi}} + g^{\text{neg}}_s(u)\Big] + r_{\text{ex}}(u)$$

#### 3. 合成候选概率梯度与 Logits 梯度
每个活跃区域 $R_m$ 的候选 $k$ 的概率偏导由 **完整 4 项链式法则** 精确闭合：

$$g_{p,m,k} = \frac{\partial \mathcal{L}}{\partial p_{m,k}} = \underbrace{g_a(r_m) \cdot a_{m,k} + g_s(r_m) \cdot s^{\text{out}}_{m,k}}_{\text{1. 到达时间与输出 Slew 直接混合贡献}} + \underbrace{\sum_{u} g_{\text{fo}}(u) \cdot \Delta\text{fo}_{m,k}(u)}_{\text{2. 外部叶子/命中门扇出与 Slew 投诉反传}} + \underbrace{\frac{\lambda_{\text{area}}}{|V_0|}\,\Delta\text{AND}_{m,k}}_{\text{3. 规模无关面积代价}} - \underbrace{\frac{2\lambda_{\text{disc}}}{|V_0|}\,p_{m,k}}_{\text{4. 离散化磁铁吸附}}$$

再经 Softmax Jacobian 得到 $z_{m,k}$ 的解析梯度：

$$\frac{\partial \mathcal{L}}{\partial z_{m,k}} = \frac{p_{m,k}}{\tau_p}\left(g_{p,m,k} - \sum_{j=0}^{K_m - 1} p_{m,j}\,g_{p,m,j}\right)$$

---

### 模块 5：物理量纲自稳态更新与 GPU 多置信度阈值离散寻优

1. **Warm-Start 初始化**：将 `no-op`（$k=0$）设为温和正偏置（如 $z_{m,0} = 0.25$，其余 $z_{m,k} = 0.0$），保证零收益或非关键摇摆区域在磁铁吸附期自然收敛回 `no-op`。
2. **严禁跨目标平均梯度强行拉平（吸取 Stage A P0-3 教训）**：**不做**“在 $t=1$ 用全图平均时序梯度幅值去缩放面积权重”的预条件操作——因为大电路中多数区域不在最慢路径上，按均值拉平会人为放大时序权重数十倍、重新引发严重面积膨胀。时序与面积的平衡直接由已对齐物理意义的 ADP 无量纲系数（$t_1+t_2 = w = 2.0,\; \lambda_{\text{area}} = 1.0$）决定。
3. **Adam 自适应步长 + 单步硬限幅**：采用每个参数独立的轻量 Adam 更新 $z_{m,k}$，并对单步更新量施加硬裁剪 $|\Delta z_{m,k}| \le \delta_{\max}$（如 $\delta_{\max} = 0.35$），防止个别极端高敏感度区域单步跳变锁死。
4. **GPU 多置信度阈值离散寻优（Multi-Threshold Discrete Line Search）**：
   * 内层 $T$ 步结束后，令各区域首选候选为 $k^*_m = \arg\max_k p_{m,k}$。
   * 在 GPU 上对一组置信度阈值 $\theta \in \{0.50,\; 0.65,\; 0.80,\; 0.90,\; 0.95\}$ 并行/快速顺序执行 **5 次精确硬 $\max$ 离散前向评估**（当 $p_{m,k^*_m} < \theta$ 时该区域置回 $k=0$ `no-op`，5 次合计耗时 $< 3\text{ ms}$）。
   * 计算每档真实离散代理净收益 $\Delta\text{ADP}_{\text{disc}}(\theta) = w \cdot \frac{D_{\text{disc}}(\theta) - D_0}{D_0} + \frac{\Delta\text{AND}(\theta)}{|V_0|}$，直接选取 $\Delta\text{ADP}_{\text{disc}}(\theta^*)$ **最小（最负）** 的最优档位 $\theta^*$。
   * **严格单调兜底**：若所有档位均满足 $\Delta\text{ADP}_{\text{disc}}(\theta) \ge -10^{-7}$（无净正收益），当轮安全早停不改图；否则调用已验证零崩溃的 `StrashRebuild (--fo-cap 0)` 生成新 AIG 进入下一外层轮次。

---

## 三、CUDA GPU 工程数据结构与核函数布局

所有内层数组在外层每轮开始时一次性分配常驻显存，内层 $T$ 步循环内**零动态显存分配**：

### 1. 核心扁平化显存数组（SOA 布局，`K = 5`, `MAX_SUB_GATES = 8`）
* **活跃区域与紧凑微子图元数据**：
  * `d_isRegionRoot[N]`（标记节点是否为活跃根 $r_m$ 及其区域下标 $m$）；
  * `d_regionRoot[M]`、`d_candCount[M]`、`d_candDeltaAnd[M * K]`、`d_candWireLit[M * K]`（$\ge 0$ 表示导线折叠目标字面量，$-1$ 表示门级微子图）；
  * `d_candSubGates[M * K * MAX_SUB_GATES]`（紧凑编码每个候选的 1~8 个内部门的左右输入源——指向叶子/已有 `hit` 节点或前序内部门、极性反相标志及内部静态扇出 $e\text{Fo}_j$）。
* **稀疏外部扇出变化 CSR 表**：`d_foDeltaOffset[M * K + 1]`、`d_foDeltaNode[NNZ]`、`d_foDeltaVal[NNZ]`（每个候选仅 4~8 个非零项）。
* **优化变量与前向/反向状态缓存**：
  * `d_logits[M * K]`、`d_prob[M * K]`、`d_adamM[M * K]`、`d_adamV[M * K]`；
  * `d_expFo[N]`、`d_arr[N]`、`d_slew[N]`、`d_candArr[M * K]`、`d_candOutSlew[M * K]`、`d_candRootInSlew[M * K]`；
  * `d_fwdW[N * 4]`（普通节点到达时间与 Slew LSE 权重）与 `d_candFwdW[M * K * MAX_SUB_GATES * 2]`（微子图内部 LSE 权重）；
  * `d_gArr[N]`、`d_gSlew[N]`、`d_gPiNegArr[N_PI + 1]`、`d_gPiNegSlew[N_PI + 1]`、`d_gFo[N]`。

### 2. 内层单步 6 个轻量 CUDA 核函数流水线
1. `k_softmax_and_exp_fo`：计算 $p_{m,k}$，初始化 `d_expFo` 为 `fo_base`（PI 为 `negFo_base`），并按稀疏 CSR `atomicAdd` 累加 $p_{m,k}\Delta\text{fo}_{m,k}(u)$；
2. `k_fwd_level`（按预建 `Level 1 .. L` 启动）：对普通门计算光滑负载与实时到达时间/Slew；对活跃根 $r_m$ 实时评估 `no-op` 与各候选微子图，写回混合 $(a(r_m), s(r_m))$ 及前向 LSE 权重缓存（因已保证 $\text{level}(h) < \text{level}(r_m)$，单层单次发射完全无竞态）；
3. `k_loss_and_po_grad`：归约计算 $\text{WNS}_\gamma$、$\text{TNS}_\gamma$ 与总 Loss，初始化 PO 端伴随梯度；
4. `k_bwd_level`（逆序 `Level L .. 1` 启动）：普通门与活跃根微子图沿缓存权重逆向精确分发 $\Delta g_a, \Delta g_s$ 至上游节点与 `d_gPiNeg*`；
5. `k_fanout_grad`：按三类节点公式一次性并行算出全图精确扇出偏导 `d_gFo[N]`；
6. `k_region_grad_and_adam`：每个线程负责 1 个活跃区域 $m$，读 CSR 收集扇出投诉，合成完整 4 项 $g_{p,m,k}$，经 Softmax Jacobian 与 $|\Delta z| \le \delta_{\max}$ 裁剪后执行 Adam 更新。

---

## 四、分阶段开发路线与验收门槛

### Stage B-1：MVP 闭环、双精度/单精度有限差分梯度检验与 16 电路面板验证
* **开发内容**：在 `rw4b.cu` 落地上述 5 大模块与 `--joint-grad` 开关。
* **第一道硬门槛（强制优先通过）**：编写 **有限差分数值梯度检验（`--grad-check`）**——在 `c1355`、`c5315`、`intra_pred_PE`（覆盖普通门、PI 负相边、strash 命中门与非串联导线折叠）上随机抽检 $\ge 30$ 个活跃区域的全部 $z_{m,k}$，对比解析反向传播梯度与数值中心差分 $\frac{\mathcal{L}(z + \epsilon) - \mathcal{L}(z - \epsilon)}{2\epsilon}$，**必须保证最大相对误差 $< 10^{-4}$**，彻底排除任何求导或索引漏项后再做调参。
* **16 电路目标面板验收标准**：
  * 全部 16 电路 `cec -n` 100% PASS；
  * 内层 Loss 平稳下降，多阈值离散寻优稳定命中正收益档位，离散化前后代理落差可控；
  * 真实 ASAP7 映射 Delay 与 ADP 超越 Stage A 同面板基线。

### Stage B-2：全量 905 电路大考与顶会消融实验矩阵
* **全量 905 电路验收**：在 RTX 4090 上跑通全量 905 电路，目标：
  * 905/905 零崩溃，可完成 `cec` 电路 100% PASS；
  * 全集 894 可比电路真实 ASAP7 Delay 与 ADP 全面突破 Stage A（Delay 优于 `-10.22%`、ADP 优于 `-6.88%`）；
  * 保持 $100\times$ 以上核心加速比（全集核心总耗时控制在 5 至 10 分钟内）。
* **论文核心消融与强基线补全**：
  * **消融 1（核心范式验证）**：Stage A 单步一阶贪心 vs. Stage B 多步联合梯度下降（同 12-cut 引擎、同 3 轮外循环）。
  * **消融 2（双通路反传验证）**：关闭通路 2（令 $g_{\text{fo}} = 0$）vs. 完整双通路反传。
  * **消融 3（离散化机制验证）**：无磁铁单阈值 vs. Gini 磁铁 + 多置信度阈值离散寻优。
  * **补齐强延迟基线**：在 905 电路上补跑 ABC 强延迟综合流（如 `if -g` SOP Balancing、`dch` 结构选择），夯实顶会对比说服力。

### Stage B-3：参数锁死与 24 电路外部盲测收官
* 待 905 电路全部消融完成并彻底锁死代码与默认超参后，一次性解封 **24 电路外部盲测集（`panel.json`）** 完成最终验收，杜绝任何数据泄露与过拟合风险。

---

## 五、工程性能与边界细节备忘

1. **超高扇出节点的原子累加开销**：若全图数百个活跃区域同时向同一高扇出 PI 或共享门 `atomicAdd` 反向梯度，RTX 4090（Ada `sm_89`）原生 `float atomicAdd` 通常仅耗时亚毫秒；若在大电路上成为热点，可在 warp 内做寄存器归约后再发射原子加。
2. **小电路的逐层 Kernel 发射延迟**：对几百门的小电路，内层 25 步逐层发射核函数的 Host 调度延迟可能高于计算本身。Stage B-1 跑通后，可在 Stage B-2 引入 **CUDA Graph** 将单步层循环录制为静态计算图（仅更新步间标量温度/权重参数），一次 `cudaGraphLaunch` 跑完单步前反向。
