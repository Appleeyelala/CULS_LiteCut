# Step 1：原图 unit-AND level 与 root slack

四个 pilot 电路（c1355、c5315、intra_pred_PE、cabac_ucontext）的原图 unit-AND level、required
层数和每个 strict MFFC root 的 slack。本目录不含候选、改写或映射，只回答两个问题：原图层数
算得对不对，slack 怎么分布。

按用户 2026-11-23 的指示，本步骤复用 2026-11-21 已验收的 run，不重新计算；这里只做打包与上传。

## 结果

| 电路 | AND | strict MFFC roots | 零 slack root | Python D | ABC D | 相等 |
|---|---:|---:|---:|---:|---:|:--:|
| c1355 | 512 | 62 | 56 | 29 | 29 | ✓ |
| c5315 | 1,613 | 264 | 8 | 38 | 38 | ✓ |
| intra_pred_PE | 5,407 | 1,163 | 48 | 71 | 71 | ✓ |
| cabac_ucontext | 37,490 | 5,249 | 220 | 77 | 77 | ✓ |
| **合计** | 45,022 | 6,738 | 332 | — | — | 4/4 |

零 slack root 占 332 / 6,738 = **4.93%**。四个电路共 6 个不可达节点（不以任何 PO 为终点），
对应 0 个不可达 root。

**ABC 校验全部相等。** Python 的 `propagate_levels` 给出的 `D` 与冻结 ABC 对同一份 AIG 执行
`read_aiger; print_stats` 得到的 `lev` 在四个电路上逐一相等，不存在差一。

## slack 的定义（第 3 步必须区分）

本目录 CSV 里的 `slack` 是 **root 级** slack：

```
slack(root) = required[root] - level[root]
```

由 `global_timing.level_weights.Levels.slack()` 计算。对一个 root 来说，它与"路径 slack"
`required[root] - (level[root] + d)` 在 `d = 0` 时完全一致，所以本步骤的定义与第 3 步不冲突。

**但第 3 步的叶子权重不能用 `Levels.slack(leaf)`。** 一个 cut 叶子可能同时喂给多条下游路径，
`required[leaf]` 取的是所有路径里最紧的那条，因此节点 slack 会**低估**经过当前 root 的路径
slack。叶子的路径 slack 必须按

```
slack_path(leaf) = required[root] - (level[leaf] + original_leaf_depths(...)[leaf])
```

计算，即 `global_timing.level_weights.original_leaf_depths` 返回的 `depths`。这个函数在遇到
cut 叶子时停止，只统计 cut 边界内的路径，正好对应"经这个 root"的语义。

另注：`original_leaf_depths` 同时返回第二个值 `outside`，是边界外节点的最大绝对到达时间。
c5315 的 root 460 正是靠它定位的——support 最小化后的三个叶子漏掉了一条 7 层路径，
函数验证通过但层数不可重建。第 2 步导出候选时会再次遇到同一现象，不要把它当作例外跳过。

## 执行命令

```
/home/share/process/anaconda3/bin/python -B \
  /data/wushangge/CULS_LiteCut/runs/20261121T053338Z-level-four-baselines-9506eb5c/src/scripts/aig_level_experiment.py \
  --run /data/wushangge/CULS_LiteCut/runs/20261121T053338Z-level-four-baselines-9506eb5c \
  --cohort pilot
```

## 溯源

| 项 | 值 |
|---|---|
| run | `20261121T053338Z-level-four-baselines-9506eb5c` |
| Slurm | 66256，COMPLETED |
| 资源 | 2 CPU / 8 GiB / 0 GPU / 20 min / 未指定 nodelist |
| 源码 commit | `b43292dfb3caebf3c8caac1d2b522864b1b40728` |
| 生产脚本 | `scripts/aig_level_experiment.py`，SHA-256 `a01407e2b71d2dbdde5b4d856ddf359bbde99fcad6b96832af6f9489f37918e5`（与 run 快照逐字节一致） |
| ABC 二进制 | `/data/.../runs/20261111T173355Z-acyclic-replay-tool-build-64cf764e/build/framework_tools/abc/abc`，SHA-256 `5327dacd8df3b98caf6cea3ab4f01134413c0a8dc8dcda6a181a59e2c83f2bbb` |
| Liberty | ASAP7，SHA-256 `819ecfc2609a1f53b9d298bf0cba75af0f1a8c737f33d4bbc5c9fb1156e8fa55`（本步骤未使用） |

输入 AIG（`stage_inputs` 在运行前逐个核对 SHA-256，不一致会直接失败）：

| 电路 | SHA-256 | 来源 |
|---|---|---|
| c1355 | `2602b10f131a6ba5996521036ba130aab128a4aa15e3ab49ce44f2f02047f0b0` | 本次 run 的 `input/design.aig`（默认设计，取自 `open-sourceDataset/OpenABC/c1355_orig.aig`） |
| c5315 | `d1c0cea55baf5a407495d82eb4385a9ddea8528cde21a762f46bb0774fc93afd` | `runs/20261119T020347Z-litecut-ours14-af7ccd61/input/circuits/c5315_orig.aig` |
| intra_pred_PE | `889b40b408ff0e3333a81f73e99887ee379fb60c6e8c1b5f23d1056ce7c088a4` | 同上目录 |
| cabac_ucontext | `77fdf62aa46fa9a5bdb67a49ede156bff75bc8ea121faff67b74fa89a2a78707` | 同上目录 |

c1355 不在 paper14 的 `circuits.csv` 里，所以它走默认设计路径；其余三个来自 paper14 冻结名单。

## 文件

| 文件 | 内容 |
|---|---|
| `<circuit>.csv` | 每行一个 strict MFFC root：`region, root, level, req, slack` |
| `summary.json` | 四电路汇总：`D`/`abc_D`/`ands`/`strict_roots`/`zero_slack_roots`/`root_slack_histogram`/`po_level_histogram`/`unreachable_nodes`，加溯源字段、slack 定义说明与每个 CSV 的 SHA-256 |

`level_check.json` 的字段已并入 `summary.json` 的 `circuits` 段，不再单列文件。

## 与计划的差异

1. **未新写 `scripts/level_slack_report.py`。** 计划要求新脚本，但相同逻辑已作为
   `scripts/aig_level_experiment.py` 的 `original_levels` 阶段在 11-21 执行并通过验收。
   按用户指示复用，不重写，避免两套实现对同一物理量给出不同数字。
2. **"slack=0 的路径占比"给的是 root 级占比（4.93%）。** 真正的路径级 slack 分布需要 cut 叶子，
   要等第 2 步导出候选后才能算。这里不提前给出会影响第 3 步的可比性。
3. 计划的验收项"`tests/test_level_weights.py` 通过"属于同一 run 的 `unit_fixtures` 阶段，
   先于 `original_levels` 执行，失败即中止；本次 run 为 COMPLETED，因此该检查已通过。

没有电路失败。
