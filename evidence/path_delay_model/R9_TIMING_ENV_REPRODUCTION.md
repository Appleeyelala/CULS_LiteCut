# R9 timing environment reproduction — PASS (2026-10-12)

## Fixed inputs and provenance

Only c1355, candidate `root197_cut2_subgraph2`, region R9 (root213), cut `[17,33,151,188] → 197`, source `pi03` (PI node4), target `po11` were used. The candidate function point is serialized287; region root213 corresponds to serialized322. The existing independent single-replay snapshot `S002.aig` was reused, not regenerated or cumulatively rewritten. Its manifest has exactly one ACCEPTED decision, root197/cut_slot2/selected_subgraph2. Fresh whole-network CEC: PASS.

Historical run: `/data/wushangge/runs/CULS-mapper-internal-20261003-34U6OZ` (abbreviated PRIOR below).

| Artifact | SHA256 |
|---|---|
| PRIOR/input/c1355_orig.aig | `2602b10f131a6ba5996521036ba130aab128a4aa15e3ab49ce44f2f02047f0b0` |
| PRIOR/input/S002.aig | `59dbe54c03d28c336775785e42d3e4781713e66e96769ac688fd38977672b5e1` |
| PRIOR/input/asap7.lib | `819ecfc2609a1f53b9d298bf0cba75af0f1a8c737f33d4bbc5c9fb1156e8fa55` |
| Historical independent ABC binary | `f255acd802f928fb011737843419d1b39454f060bbd74968a355d3e476822784` |
| New read-only trace binary | `be15b3f932a691e104f555101801ff24359c6a953a13c338500227fc6421910e` |

Source commit: `c738ed6e86e6ca10bcef0d4c649cc07d925b4be8`, plus prior pair-delay/mapper instrumentation patches. Current source checkout commit was rechecked. Historical executable: PRIOR/bin/abc_pair_timing. New source/binary: RUN/abc and RUN/abc/abc, where RUN is `/data/wushangge/runs/CULS-R9-mechanism-20261012-9CY36o`. Frozen binaries were not overwritten.

## Why BLIF reuse returned11 ps

This is a library-state problem, not a measured rewrite effect:

1. `read asap7.lib` loads Liberty/SCL and installs a provisional MIO library. In `src/map/scl/scl.c`, `Scl_CommandReadLib` defaults Gain to0 and calls `Abc_SclInstallGenlib`.
2. `src/map/scl/sclLibUtil.c:Abc_SclInstallGenlib` takes the Gain==0 branch. `Abc_SclProduceGenlibStrSimple` explicitly writes `UNKNOWN 1 999 1.00 0.00 1.00 0.00` for every input pin. These are unit-delay placeholders, not the mapped fixed-delay library used historically.
3. `map` invokes `src/base/abci/abcMap.c:Abc_NtkMap`. When SCL has timing data, it calls `Abc_SclDeriveGenlib`, then `Mio_LibraryTransferDelays(existing_MIO,derived_MIO)`.
4. The map command defaults Slew to0 (library average), Gain to250, nGatesMin to0. `Abc_SclProduceGenlibStr` computes class-pin fixed delays through `Abc_SclComputeDelayClassPin`, prints them into GENLIB with `%7.2f`, and reads this library. `Mio_LibraryTransferDelays` transfers block rise/fall/max delays by truth-class matching and positional pin traversal; it is not simply a Liberty cell-name lookup.
5. `read <mapped BLIF>` resolves cells in the currently installed MIO library but does not run that mapping-time derivation/transfer. The previous reuse flow therefore retained1 ps pins; its11-gate selected trace produced11 ps and even a different predecessor sequence.

The restored `print_pair_delay` model is MIO fixed-delay timing, derived from SCL during `map`. It is **not** per-network load/slew-aware Liberty `stime`. The model retains MIO UNKNOWN phase and separate rise/fall propagation; the selected arcs in this case have equal rise/fall delays. Delay derivation contains a0.01 ps GENLIB text formatting step; subsequent arrival arithmetic is float32, not rounded to0.1 ps.

## Reproduction commands and first responsible operation

Historical logs record:

```text
read PRIOR/input/asap7.lib;
read PRIOR/input/{c1355_orig.aig,S002.aig};
map; write_blif ...; topo; stime; print_stats;
print_pair_delay po11 pi03
```

Fresh unmodified historical binary runs used the same library and default `map`:

```text
read PRIOR/input/asap7.lib; read PRIOR/input/<variant AIG>;
map; write_blif RUN/<variant>.blif; topo;
print_pair_delay -tv po11 pi03;
stime; print_pair_delay -tv po11 pi03
```

Both pre-stime and post-stime results reproduce the history. Thus **map**, not stime, restores the MIO delay state. `topo` changes object IDs/order, not arc values or pair delay. No new library or map options were introduced.

| Variant | Expected ps | Reproduced ps (before/after stime) | Printed-value error | Threshold |
|---|---:|---:|---:|---:|
| Baseline | 194.169998 | 194.169998 / 194.169998 | 0 | 0.00002 ps |
| Candidate | 193.519989 | 193.519989 / 193.519989 | 0 | 0.00002 ps |

Delta of printed values:−0.650009 ps. Representative non-unit MIO pins: XOR A27.8099995, NAND4 C17.5300007, NAND4 D18.1800003 ps. Both upper-bound and single-source consistency checks pass. Float32 accumulation of each selected trace matches its output within0.00002 ps.

## Read-only instrumentation and noninterference

Only independent RUN sources were changed:

- `src/base/io/ioReadAiger.c:Io_ReadAiger`: print actual serialized variable→ABC object/polarity table. Structural hashing makes fixed-offset ID conversions invalid.
- `src/base/abci/abcTiming.c:Abc_NtkPairPrintTrace_rec`: print selected-node provenance and competing fanin arrivals/MIO pins.
- `src/base/abci/abcTiming.c:Abc_NtkPairDelayTrace`: print already-computed mapped state for provenance/cone checks.

All additions are guarded by `AIGREWRITE_PAIR_CASE`; they only read/print. Build: `make -C RUN/abc -j4` under Slurm. Instrumented pre-topo and post-topo queries match the historical binary's delays and selected cell/arc/arrival sequences. Each mapped BLIF body is identical to the fresh historical-binary output **and** the previous historical mapped BLIF. The only excluded BLIF line is its timestamp comment.35 cross-network function proofs pass CEC. No mapping decision or arrival computation changed.

Jobs:49362 (fresh CEC/reproduction),49363/49367/49369 (incremental read-only builds/observations),49365/49368/49370/49371 (trace analysis/proofs), all COMPLETED/exit0. Final build6 seconds, final analysis3 seconds; largest allocation4 CPUs/8G. Final reconstruction parses each `%.9g` arc back into float32 **before** addition, matching ABC arithmetic exactly; adding the decimal double directly can alter a half-ULP rounding tie. Small final CSV/JSON files are copied into the experiment results; binaries, full mapped state and raw logs remain under RUN. Reproduction scripts are RUN/job.slurm, RUN/instrument.slurm and project `run_r9_mechanism.py` (staged with `tools/gate3b_analyze.py`).
