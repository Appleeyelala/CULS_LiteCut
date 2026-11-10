# ABC timing source audit

Audit date: 2026-09-29  
Scope: read-only audit of the ABC binary used by Gate 3F and the closest local source checkout. No timing algorithm, mapping flow, or experiment was changed or rerun.

## Audited build and source provenance

The frozen Gate 3F manifest identifies:

- ABC binary: `/data/wushangge/runs/CULS-gate3f-20260922-gate2babc/bin/abc`
- Binary SHA256: `9b64d35ba403df4b1a2474c4591abdfd298e9a2fe1411f6ba16ae9096b6e2b2f`
- Version string: ABC 1.01, compiled 2025-08-06 14:07:48
- Flow: `read <ASAP7 lib>; read <AIG>; map; topo; stime`
- ASAP7 library declares `time_unit : "1ps"` and `capacitive_load_unit (1,ff)`.

The CULS `abc` submodule is not initialized. The closest complete local source tree by date is `/home/wushangge/AIGRewrite/abc`, commit `c738ed6e86e6ca10bcef0d4c649cc07d925b4be8` (2025-08-05). The frozen binary contains the `Abc_CommandPrintDelay` and `Scl_CommandStime` symbols and ABC source-path strings. The binary manifest does not record its source commit/build recipe, so the source below is a strongly date-matched audit basis, but cannot be cryptographically proven to be the exact source used for that binary.

## Command and call chains

### `print_delay <CO_name> <CI_name>`

Registration and implementation are in `src/base/abci/abc.c` (`Abc_CommandPrintDelay`, around lines 924 and 3209). The command resolves the named output and input objects, then calls:

```text
Abc_CommandPrintDelay
  -> Abc_NtkDelayTrace(pNtk, pObjOut, pObjIn, fPrint=1)
     -> Abc_NtkTimePrepare
     -> Abc_NtkDfs + Abc_NodeDelayTraceArrival for mapped logic nodes
     -> Abc_NtkDelayTraceCritPath_rec / Collect_rec
     -> print path text
```

The timing and trace routines are in `src/base/abci/abcTiming.c` (roughly lines 424–451, 939–1018, and 1031–1157). `Abc_NtkDfs` explicitly returns internal logic nodes, not CI/CO objects (`src/base/abc/abcDfs.c`, around line 82).

### `stime`

Registration is in `src/map/scl/scl.c` (around line 112); `Scl_CommandStime` is around line 840. It checks that the network is mapped and topologically ordered and a Liberty library is loaded, then calls:

```text
Scl_CommandStime
  -> Abc_SclTimePerform
     -> Abc_SclTimePerformInt
        -> Abc_SclManStart
           -> Abc_SclMioGates2SclGates
           -> Abc_SclManReadSlewAndLoad
           -> Abc_SclTimeNtkRecompute
              -> Abc_SclComputeLoad
              -> forward CI -> node -> CO arrival/slew pass
              -> reverse node pass for downstream departure/depth
        -> Abc_SclTimeNtkPrint
        -> Abc_SclManFree
```

The implementation is in `src/map/scl/scl.c`, `sclSize.c` (notably `Abc_SclTimeNode`, `Abc_SclTimeNtkRecompute`, `Abc_SclManStart`, around lines 273–445 and 648–712), `sclSize.h`, `sclTime.h`, and `sclLib.h`.

With bare `stime`, `-c`, `-a`, `-p`, and `-d` are all off, and the tree-load ratio is zero. Thus this command does not enable the optional wire-load model or the optional fanout/tree estimate. It calculates timing in a temporary `SC_Man`, prints the report, and frees that manager; its full per-node state is not retained for a later command.

## Two different mapped timing models

The two commands do not read the same per-arc delay representation:

| Command | Active timing data | Arc computation |
|---|---|---|
| `print_delay` | mapped network's MIO gate/pin data (`pNode->pData`, `Mio_Pin_t`) | fixed `Mio_PinReadDelayBlockRise/Fall` values; transition polarity is applied; no slew/load table lookup in this pass |
| `stime` | mapped gate IDs translated to Liberty `SC_Cell`/`SC_Timing` data | pin-to-output rise/fall delay surfaces interpolated using input slew and output load; output slew is propagated too |

`print_delay`'s arrival state is attached to `Abc_Ntk_t::pManTime`: `vArrs` and `vReqs` are indexed by object ID, and each `Abc_Time_t` contains `float Rise, Fall` (`src/base/abci/abcTiming.c`; public readers are declared in `src/base/abc/abc.h`). The mapped pin delay accessor returns `double`, but `Abc_NodeDelayTraceArrival` casts it to `float` before adding it to the float arrival.

In this `read Liberty; map` flow, `Abc_NtkMap` derives the mapper's MIO/GENLIB library from the SCL library when timing tables are available (`src/base/abci/abcMap.c`, `Abc_NtkMap`; `src/map/scl/sclLibUtil.c`, `Abc_SclDeriveGenlib`). The mapper-facing pin delay is a fixed GENLIB approximation derived from the Liberty surface at a reference slew/load model (`Abc_SclComputeDelayCellPin`, `PD + 0.01 * LD * Gain`). After mapping, `print_delay` consumes the resulting fixed MIO block-delay fields; `stime` independently evaluates the retained Liberty tables against mapped-network loads and propagated slews.

For `stime`, the active `SC_Man` stores `SC_Pair` float arrays: `pTimes` (rise/fall arrivals), `pSlews`, `pLoads`, and `pDepts`; the object-to-cell assignment is `vGates`. `SC_Timing` stores rise/fall delay and transition surfaces per related input pin and timing sense (`src/map/scl/sclLib.h`). `Scl_LibLookup` performs bilinear interpolation in float arithmetic (`sclLib.h`, around line 490), and `Scl_LibPinArrival` takes maxima across the applicable positive/negative/non-unate transition cases (around line 526).

The Liberty reader normalizes timing to ps and capacitance to fF (`Abc_SclLibNormalize`, `src/map/scl/sclLibUtil.c`, around lines 750–783). It scales time by `10^(12 - unit_time)`; for this ASAP7 library, whose declared unit is 1 ps, the time scale is 1. `stime`'s global delay is printed with `%.2f ps`; per-node/path arrivals are generally printed with one decimal place. `print_delay` path arrivals use `%.1f`.

## Arrival, required, departure, and slack

- **`print_delay` arrival:** `Abc_Time_t` rise/fall floats in `pManTime->vArrs`. `Abc_NtkTimePrepare` seeds CI arrivals from the timing manager/default constraints, clears internal-node arrivals, and then `Abc_NodeDelayTraceArrival` propagates mapped-gate arrivals.
- **`print_delay` required:** `Abc_Time_t` rise/fall values in `pManTime->vReqs`. `Abc_NtkDelayTrace` does not perform a backward required-time propagation. The path footer reads the output object's existing required value; the edge ranking used to select the displayed path is separately stored as float bit patterns in a `Vec_Int_t`.
- **`stime` arrival/slew:** `SC_Man::pTimes` and `pSlews`.
- **`stime` reverse quantity:** `SC_Man::pDepts` is propagated backward through mapped arcs by the `fDept` pass. It represents the downstream departure/depth used by this implementation, not the `Abc_Time_t` required array.
- **`stime` slack:** the SCL helper computes it from a delay target and arrival plus departure, e.g. `D - max(arrival_rise + departure_rise, arrival_fall + departure_fall)` (`sclSize.h`, `Abc_SclObjGetSlack`). Bare `stime` does not persist a full classic per-node required-time array. Its main report is global max delay and its optional path/all-node text.

## Precision: display versus internal representation

The `0.1 ps` seen in `print_delay` path text is **display precision**, not an internal 0.1-ps timing quantum. The command's arrival storage and arithmetic are binary32 `float`; mapped pin block delay is cast to float, and output uses `%.1f`. Around 194 ps, one binary32 ULP is about `1.5e-5 ps`; actual model/table accuracy is a separate and much larger concern. The output text itself only preserves tenths of a ps.

`stime` also uses binary32 floats for its `SC_Pair`, delay surfaces, interpolation, and max-delay state. Its summary uses two decimal places, but that formatting does not mean the timing calculation is quantized to 0.01 ps. The 0.01-ps precision stated in the old Gate 3F manifest is not established as an internal precision by the source; it is only a report claim. The `print_delay` extraction documented by the calibration scripts has one decimal place and therefore has 0.1-ps textual resolution.

## What `print_delay <PO> <PI>` actually establishes

The command's help describes a critical path between the optional endpoints. The implementation does two distinct things:

1. It propagates arrival at every mapped node from the network's CI arrival seeds, using **all** CIs. The arrival at a reconvergent node is the max across its input pins/transitions. The PO-wide maxima are also computed across all outputs.
2. When a CI is specified, recursive path tracing only accepts fanin paths that can reach that CI. At each gate it chooses the eligible edge with the smallest locally computed edge slack. That edge slack was formed from the already-computed **unrestricted, all-CI** node arrivals.

Consequently, the specified PI constrains which structural route is printed, but ABC does not rerun forward timing with other PIs excluded. The printed per-node `Arrival` values are the unrestricted global arrivals, not cumulative arrivals restricted to that PI. `Abc_NtkDelayTrace`'s return value is the maximum over all CO drivers, and `Abc_CommandPrintDelay` discards it. The output footer prints required time and path slack, not a dedicated pair-delay scalar.

Therefore:

- `print_delay po11 pi03` is a **PI-constrained critical-path trace**, not a reliable numeric `PI4 -> PO11 endpoint_pair_delay` calculation by itself.
- Parsing the final gate's `Arrival` from that trace produces the global arrival at that gate (formatted to 0.1 ps); it does not prove the delay from the specified PI.
- The existing 194.4-ps baseline and pilot values, where extracted as the last `Arrival` in `print_delay` output, must be treated as **legacy trace-arrival values**, not confirmed PI4→PO11 endpoint-pair delays. Do not use them as exact pair-delay ground truth without a corrected extraction.
- `stime` as invoked by the frozen flow reports a global critical delay; it has no PI/PO endpoint arguments. A mapped-network endpoint-pair delay with the same Liberty load/slew model requires a source-restricted propagation/export path (or another explicitly validated endpoint query), not a guess based on the traced AIG path.

This is the main semantic mismatch discovered in Step A. It should be resolved before fitting or evaluating an ABC-propagation prototype against the current single-candidate labels.

## State export/API implications for the next stage

The MIO-model arrivals can be read through ABC's public `Abc_NodeReadArrival` after the trace has populated them, but they are not the Liberty-table `stime` state. The SCL `SC_Man` timing manager is freed at the end of `stime`; `stime -a` can print nodes, but the normal per-node text is only one decimal place. To export the full-precision SCL state, a small read-only export hook must run while `SC_Man` is alive (before `Abc_SclManFree`) or the timing command must be wrapped in an API that retains that manager. No such hook was implemented in this audit.

ABC's SCL sizing implementation already contains incremental timing machinery (`Abc_SclTimeIncUpdate`, `Abc_SclTimeIncUpdateArrival`, and level/fanout dirty propagation in `sclSize.c`, around lines 455–575). This is used inside sizing/update flows; the source does not expose it as a standalone API for injecting an arbitrary rewrite candidate and querying a selected PI/PO pair. Its existence makes downstream-only propagation plausible, but does not establish that a scalar root-delay replacement is sufficient: the active SCL arc delay depends on mapped cell, input slew, output load, timing sense, and the changed fanout/load network.

## Audit conclusion

1. `print_delay` is a mapped-network fixed-pin-delay critical-path tracer; it is not the frozen flow's Liberty-table STA command and does not produce a source-restricted endpoint delay scalar.
2. `stime` is the Liberty-table static timing pass. It propagates rise/fall arrivals and slew forward, then downstream departure/depth backward, and reports global critical delay.
3. `print_delay`'s 0.1-ps display is formatting. Its active arrival arithmetic is `float`, not 0.1-ps quantized. `stime` is also float-based; its `%.2f` global output is formatting, not proof of 0.01-ps internal quantization.
4. Before a one-round prototype, the PI/PO measurement needs a corrected endpoint-specific baseline and matching candidate labels. A candidate should ultimately enter at the mapped candidate network's cell/arc/connectivity/load/slew timing state, not as a LiteCut scalar delta. The exact prototype boundary is left for the next confirmed stage.
