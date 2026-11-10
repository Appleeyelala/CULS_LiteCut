# Gate 3B report — c1355_orig

## 结果

Gate 3B = **PASS**。
本轮直接复用 Gate 3A 的 `STRICT_MAXIMAL_MFFC_PARTITION`，没有重新计算 partition。

## 固定输入

- AIG：`/home/wushangge/cutLibOrigin/open-sourceDataset/OpenABC/c1355_orig.aig`
- SHA256：`2602b10f131a6ba5996521036ba130aab128a4aa15e3ab49ce44f2f02047f0b0`
- PI / PO / AND：`41 / 32 / 512`
- strict maximal MFFC regions：`62`

## Region DAG

- topological sort 长度：`62` / `62`
- cycle count：`0`
- region DAG 无环：`True`
- 所有跨 region AND→AND edge 已进入 quotient DAG：`True`
- PI→region boundary 完整记录：`True`
- region→PO boundary 完整记录：`True`
- distinct region edges：`128`
- cross-region edge occurrences：`164`

每个 region 的 predecessor/successor、PI/PO 边界以及真实跨 region node edge 保存在 `region_metadata.jsonl`；PI→region 与 region→PO 分别保存在 `pi_region_edges.csv` 和 `region_po_edges.csv`。

## Path projection

投影规则是：对 node-level path 中的每个 AND node 使用 `owner[node]`，忽略 PI/constant 节点，然后压缩相邻重复 region ID。由于所有原始跨 region AND→AND edge 都有对应 DAG edge，这一规则是 **edge-complete path projection**，不是通过枚举所有 PI→PO path 得出的抽样结论。

- edge-complete path projection：`True`
- 每个 PO 的 longest PI→PO example：`32` 条
- example path 全部有 PI：`True`
- node path 全部合法：`True`
- region sequence 全部合法：`True`
- reentered region path count：`0`
- root 不是映射依据：`True`
- 不经过该 region root 的 sampled region segments：`0`

平均 node path length（含起始 PI）：`30.000000`；最大：`30`。

平均 compressed region path length：`6.000000`；最大：`6`。

## Determinism

run1 与 run2 对全部 canonical analysis artifacts 的 SHA256 比较结果在 `gate3b_summary.json` 中；`deterministic = **True**`。

## 约束检查

本 Gate 未执行 LiteCut、Gate1 candidate attach、rewrite/replacement、ABC mapping、STA、critical-path optimization 或 gradient。
