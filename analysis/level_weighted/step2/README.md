# Step 2：用插桩的 ABC 导出 rewrite 候选（c1355）

c1355 的全部 rewrite 候选：ABC 每评估一个替换子图就写一行，子图结构与层数一并导出。
本轮只做 c1355，其余三个电路等检查通过后再跑。

## 做了什么

在 `third_party/abc` 的 `/data` 副本上打补丁（原仓库文件未改），补丁见
[`provenance/rwr_dump.patch`](../../../provenance/rwr_dump.patch)，生成器是
[`scripts/patch_abc_rwr_dump.py`](../../../scripts/patch_abc_rwr_dump.py)。环境变量
`CULS_RWR_DUMP=<dir>` 打开导出，**不设变量时行为与原版完全一致**。

补丁共四处：

1. **候选导出**（`rwrEva.c` 的 `Rwr_CutEvaluate`）：`Dec_GraphToNetworkCount` 改以
   `NodeMax = LevelMax = ABC_INFINITY` 调用，因此"新增节点超过 `nNodesSaved`"和"超过
   `Required` 层数"两类候选不会被 `continue` 丢掉。绕开了对应的 `assert`。
2. **不替换节点**（`abcRewrite.c`）：导出模式下 `Rwr_NodeRewrite` 返回后直接 `continue`，
   跳过 `Dec_GraphUpdateNetwork`。所有候选都在原图快照上枚举，而不是在不断被改写的图上。
3. **`accepted_by_default`**：用无界调用拿到的真实 `nodes_added` 与 ABC 自己的新 root 到达层数，
   分别与 `nNodesSaved` 和 `Required` 比较，复现默认判据。判据等价性：`Counter` 与 `LevelNew`
   沿 DAG 单调不减，因此"任一节点越界"等价于"最终计数/root 层数越界"。
4. **计数**：三处评估前剪枝、两处节点级跳过、每个节点一行汇总（含无 cut 的节点）。

## 计数

| 项 | 值 |
|---|---:|
| 被评估节点（`Abc_NtkForEachNode` 遍历到） | 512 |
| 节点级跳过：persistant / 高扇出（>1000） | 0 / 0 |
| 平凡 cut（节点自身，每节点各跳过 1 个） | 512 |
| 非平凡 cut 总数 | **5,942** |
| ├ 导出（4 输入 cut 且单扇出叶子 ≤ 2） | 3,300 |
| ├ 叶子不足 4 个被跳过 | 2,220 |
| ├ 单扇出叶子超过 2 个被跳过 | 422 |
| └ 叶子查不到对象被跳过 | 0 |
| 候选总数 | 11,114 |
| ├ 默认规则会接受 | 1,392 |
| └ 自替换（新 root 就是原 root，`Dec_GraphToNetworkCount` 返回 -1） | 768 |

`counters.json` 里的 `cuts_total` 只在进入导出路径时递增，因此等于 `cuts_exported = 3300`；
非平凡 cut 总数必须按 3300 + 2220 + 422 + 0 计算。`stats.json` 的 `counters.derived` 已给出换算。

512 个节点里只有 **376 个**有导出 cut，其余 136 个的所有 cut 都被上述三类剪枝排除。
`c1355.nodes.jsonl` 逐节点记录了原因计数，覆盖率可审计。

## 编号对齐

ABC 的对象编号顺序是 PI → PO → AND，AIGER 是 PI → AND。因此 AND 编号整体偏移一个常量，
**偏移量等于 PO 数**：

```
abc_and_id = aiger_and_id + po_count        # c1355: +32
abc_lit    = aiger_lit                      # PI 与常量 0 不变
abc_lit    = aiger_lit - 2 * po_count       # AND 字面量
```

这条规则是**逐节点验证**出来的，不是假设：512/512 个 AND 的扇入字面量在偏移后与
`parse_aig` 完全一致（其中 512 个的扇入顺序被 ABC 交换，属正常）。验证函数
`identifier_mapping` 在偏移不唯一或任一节点对不上时直接中止，不会产出错位的分析。

同一份数据里还验证了 PO 侧：AIGER `po_literals` 与 ABC 的 PO 扇入一致。

## 六项验收

| 验收 | 结果 |
|---|---|
| 1 默认模式回归 | 4/4 电路：打补丁与不打补丁的 `read_aiger; rewrite` 输出**节点数、level、结构摘要全等**，CEC 全部 equivalent |
| 2 快照不被修改 | 导出后 `print_stats` 仍为 `and = 512  lev = 29`，与导出前逐字段相同 |
| 3 对账 | 376 个有 cut 的节点，`gain_best` 与"导出集 + `accepted_by_default` 过滤后重算的最大增益"**0 处不一致** |
| 4 cut 有效 | 3,300 个 cut 全部满足：从 root 到任何 PI 的路径都经过叶子；**0 个非法** |
| 5 层数核对 | 随机 20 个候选按 `edges` 重算 `leaf_depths` **0 处不一致**；全量 11,114 个候选的 `root_level` 也 **0 处不一致** |
| 6 原结构层数重建 | c1355 上 **0/3300 不可重建** |

**验收 6 的说明**：第 1 步 README 记录的不可重建案例在 **c5315 的 root 460**，c1355 本来就没有
这类 cut，所以本轮 0 是预期结果，不构成对该现象的验证。要统计它必须跑 c5315 的导出。

**验收 1 的对照基准**：主对照是同源未打补丁构建（`7ab8dca8…`），不是冻结二进制。冻结二进制
（`5327dacd…`）只作次级记录——它的构建链没有源码快照和校验和，且带 CULS 映射补丁，无法把差异
归因到本次补丁。本轮它的 rewrite 输出与同源构建**完全一致**，作为附加信息记录。

## 体量

| 文件 | 大小 |
|---|---:|
| `cuts.jsonl`（3,300 行，11,114 候选） | 3.5 MB |
| `nodes.jsonl`（512 行） | 128 KB |
| `nodes.tsv` | 12 KB |

远低于计划设的 5 GB 阈值，无需 gzip 或分文件。按节点数线性外推，cabac_ucontext（37,490 AND）
约 250 MB，仍可直接落地。完整导出留在 `/data`：

```
/data/wushangge/CULS_LiteCut/runs/20261123T042529Z-rwr-dump-c1355b-41071bcc/output/dump/c1355/cuts.jsonl
bytes = 3575552
sha256 = 451b1abe360f063970ae07201723587443ca8b67bbbbbfddec141a229c5f8b75
```

## 溯源

| 项 | 值 |
|---|---|
| 构建 run | `20261123T041932Z-rwr-dump-build2-3c2894f0`，Slurm 66305，COMPLETED，8 CPU / 32 GiB / 0 GPU / 45 min |
| 实验 run | `20261123T042529Z-rwr-dump-c1355b-41071bcc`，Slurm 66307，COMPLETED，4 CPU / 16 GiB / 0 GPU / 60 min |
| 构建源码 commit | `f5fd14ecfc9df18f4d0ea3c6e1ac4bf34597a878` |
| 实验源码 commit | `37a15bde134604bd99bf212b87b5c884cf69709e` |
| ABC 源码 | `third_party/abc` @ `f5fd14ec`（原仓库文件未改，补丁只作用于 `/data` 副本） |
| abc 未打补丁 | `7ab8dca842988ffff6b8f14559c3ba7718018a85317b84be7eab4cfbc81311f0` |
| abc 打补丁 | `2b5119455ac4970dd706c0118c71bb36dcd98ed716bb5512f07509ec855a1ad8` |
| 冻结 abc（次级） | `5327dacd8df3b98caf6cea3ab4f01134413c0a8dc8dcda6a181a59e2c83f2bbb` |
| 输入 c1355 | `2602b10f131a6ba5996521036ba130aab128a4aa15e3ab49ce44f2f02047f0b0` |

其余三个电路的输入 SHA-256 与第 1 步一致，已逐个核对：c5315 `d1c0cea5…`、intra_pred_PE
`889b40b4…`、cabac_ucontext `77fdf62a…`。

构建与实验都以 `--export=NONE` 提交，PATH 与所有缓存目录手动指回 run 目录；未指定 nodelist
（纯 CPU 任务，交给 Slurm 放置，避免占用 GPU 节点）。

## 文件

| 文件 | 内容 |
|---|---|
| `stats.json` | 汇总：六项验收、计数（含 derived 换算）、编号映射、覆盖率、完整导出路径与 SHA256、每个打包文件的 SHA256 |
| `c1355.sample.jsonl` | 完整导出的前 1,000 行 |
| `c1355.nodes.jsonl` | 512 个节点的逐节点汇总（含 reason、三类剪枝计数、增益） |
| `c1355.nodes.tsv` | ABC 节点表：编号、类型、两个扇入字面量、level、扇出 |
| `c1355.counters.json` | 二进制自己写出的原始计数，原样保留 |
| `c1355.export_check.json` | 验收 2 的原始记录（导出前后 `print_stats`） |

## 与计划的差异

1. **未在本轮跑其余三个电路**，按计划"先完成 c1355 并上传、等我检查后再跑"。因此验收 6 的
   c5315 反例统计缺失，见上。
2. **额外导出了 `edges`、`n_leaves`、`n_size`、`root_node`。** 计划只要求 `leaf_depths`，
   但没有图结构就无法独立复算层数，验收 5 只能变成自证。代价是行变长，c1355 上仍只有 3.5 MB。
3. **额外记录了 `abc_level`。** `root_level` 是按 `max(叶子 level + 深度)` 结构算的（第 3 步要用），
   `abc_level` 是 ABC 内部经过 AIG 节点复用与常量折叠后的到达层数，两者可能不同；验收 3 的
   `accepted_by_default` 用后者，第 3 步的目标函数用前者。分开记录避免混淆。
4. **`cuts_total` 命名有歧义**，语义见上文换算说明；未改名以免重跑一轮构建。
5. 计划要求"记录剪枝的位置和条件"：三处位置为 `rwrEva.c` 的 `pCut->nLeaves < 4`、
   `Abc_NtkObj(...) == NULL`、单扇出叶子 `Counter > 2`；节点级为 `Abc_NodeIsPersistant` 与
   `Abc_ObjFanoutNum(pNode) > 1000`。

## 失败与负例

无电路失败。768 个自替换候选与 8,954 个默认规则不接受但已导出的候选全部保留，未过滤。

下一步：c5315 的导出（用于验收 6 的反例统计），以及另两个电路的回归与导出。
