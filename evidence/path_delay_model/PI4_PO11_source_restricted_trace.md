# PI4 → PO11 source-restricted MIO traceback

Input: original `c1355_orig.aig` (SHA256 `2602b10f131a6ba5996521036ba130aab128a4aa15e3ab49ce44f2f02047f0b0`), ASAP7 library, `read; map; topo; stime; print_pair_delay -tv po11 pi03`. AIG PI node 4 maps to ABC `pi03`; the mapped PO retains `po11`. This trace is on the **mapped** network, not the original Gate3B AIG node path. Units: ps, MIO fixed block delay. Arrival is a float32 value rendered with `%.9g`.

| Step | Mapped cell | Input pin / transition | Arc delay | Arrival |
|---|---|---|---:|---:|
| `pi03` | PI | rise | 0 | 0 |
| `n200` | XOR2x2_ASAP7_75t_L | 0 / rise→rise | 27.8099995 | 27.8099995 |
| `n212` | NOR2xp33_ASAP7_75t_L | 0 / rise→rise | 14.5699997 | 42.3799973 |
| `n217` | OAI211xp5_ASAP7_75t_L | 0 / rise→rise | 19.0499992 | 61.4299965 |
| `n225` | AOI22xp33_ASAP7_75t_L | 2 / rise→rise | 17.2999992 | 78.7299957 |
| `n227` | OAI21xp33_ASAP7_75t_L | 0 / rise→rise | 16.3600006 | 95.0899963 |
| `n228` | NAND2xp33_ASAP7_75t_L | 1 / rise→rise | 13.2399998 | 108.329994 |
| `n333` | NAND4xp25_ASAP7_75t_L | 3 / rise→rise | 18.1800003 | 126.509995 |
| `n334` | NAND4xp25_ASAP7_75t_L | 3 / rise→rise | 18.1800003 | 144.690002 |
| `n370` | AND4x1_ASAP7_75t_L | 1 / rise→rise | 20.6499996 | 165.339996 |
| `n372` | NAND4xp25_ASAP7_75t_L | 0 / rise→rise | 15.5900002 | 180.929993 |
| `n373` | NAND2xp33_ASAP7_75t_L | 1 / rise→rise | 13.2399998 | 194.169998 |
| `po11` | PO, driver `n373` | rise | 0 | **194.169998** |

The original `print_delay po11 pi03` displays **194.4 ps**, but that is the all-PI PO arrival (`194.429977` before formatting) restricted only at traceback. Source restriction yields **194.169998 ps**, a difference of −0.259979 ps. The mapped trace starts at the requested source and every gate arrival agrees with predecessor arrival plus selected MIO arc delay to float32 precision. `PAIR_CHECK` reports upper bound PASS and independent single-source forward-engine consistency PASS.

Raw records: `/data/wushangge/runs/CULS-source-pair-20260930-qU1wxi/output/baseline.log`.
