# Gate 3A report — c1355_orig

## 固定输入与规模

- 输入：`/home/wushangge/cutLibOrigin/open-sourceDataset/OpenABC/c1355_orig.aig`
- SHA256：`2602b10f131a6ba5996521036ba130aab128a4aa15e3ab49ce44f2f02047f0b0`
- PI / PO / AND：`41 / 32 / 512`
- CULS cut limit K：`12`

## 7 个问题的结论

1. Q1：选择 `c1355_orig`；PI=41，PO=32，AND=512。
2. Q2：CULS PO→PI discovery 得到 `62` 个 regions。
3. Q3：CULS 覆盖率 `1.000000`；100% = `True`。
4. Q4：disjoint=`True`，unique owner=`True`，multiply-owned=0，unowned=0。
5. Q5：完整 Standard MFFC `62` 个；cut-limited FFC `0` 个；mismatch `0` 个。
6. Q6：inclusion-maximal standard MFFC 数量 `62`；100% coverage=`True`，disjoint=`True`。
7. Q7：基于实测结果建议 `STRICT_MAXIMAL_MFFC_PARTITION`。

## 语义与 determinism

CULS raw discovery 保留 singleton cone，没有人为补 singleton；后续原始 refactor 的 `cone_size < 2` filter 未执行。

run1 canonical SHA256=`a2a7e60f8619c202f5ec8056ede0ea31f639834bdb9ab2a150f685d84ef8e287`；run2 canonical SHA256=`a2a7e60f8619c202f5ec8056ede0ea31f639834bdb9ab2a150f685d84ef8e287`；deterministic=`True`。

若出现 `MISMATCH`，说明 CULS internal-node extraction 不是 strict standard MFFC 的子集，Gate 应阻断；本报告会保留所有 counterexample。
