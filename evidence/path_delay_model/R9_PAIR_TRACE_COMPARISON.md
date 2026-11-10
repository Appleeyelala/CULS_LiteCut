# R9 source-restricted critical traceback comparison

Source PI4=`pi03`, target=`po11`. Pair delays:194.169998 vs193.519989 ps. Trace CSVs include input transition, output transition, pin index/name, arc delay, rise/fall arrivals, predecessor, pre-topo IDs, post-topo IDs and serialized-function provenance.

All IDs below are **within their own variant before topo**. Alignment is justified by actual reader provenance and35 CEC proofs, including all11 trace output functions and each gate's input frontier. It is not inferred from equal numeric IDs or equal path positions. Literal notation is ordinary serialized AIG literal; `vOrigNodeIds` is decoded separately because its low bit stores mapper fPhase, not ordinary complement polarity.

## The two traces

Each starts with PI4 arrival0 and ends with PO11's driver arrival. Cell names below omit the common `_ASAP7_75t_L` suffix. All selected transitions are rise→rise; full rise/fall fields are retained in CSV. Equal arrivals/UNKNOWN MIO phase in this case do not justify changing the general timing semantics.

| Position | Cell | Baseline gate | Candidate gate | Baseline/candidate pin | Baseline/candidate arc ps | Baseline arrival ps | Candidate arrival ps |
|---:|---|---:|---:|---|---|---:|---:|
| 1 | XOR2x2 | 148 | 140 | A / A | 27.8099995 / 27.8099995 | 27.8099995 | 27.8099995 |
| 2 | NOR2xp33 | 149 | 141 | A / A | 14.5699997 / 14.5699997 | 42.3799973 | 42.3799973 |
| 3 | OAI211xp5 | 153 | 145 | A1 / A1 | 19.0499992 / 19.0499992 | 61.4299965 | 61.4299965 |
| 4 | AOI22xp33 | 157 | 149 | B1 / B1 | 17.2999992 / 17.2999992 | 78.7299957 | 78.7299957 |
| 5 | OAI21xp33 | 165 | 157 | A1 / A1 | 16.3600006 / 16.3600006 | 95.0899963 | 95.0899963 |
| 6 | NAND2xp33 | 170 | 162 | B / B | 13.2399998 / 13.2399998 | 108.329994 | 108.329994 |
| 7 | NAND4xp25 | 334 | 330 | **D / C** | **18.1800003 / 17.5300007** | **126.509995** | **125.859993** |
| 8 | NAND4xp25 | 335 | 331 | D / D | 18.1800003 / 18.1800003 | 144.690002 | 144.039993 |
| 9 | AND4x1 | 369 | 368 | B / B | 20.6499996 / 20.6499996 | 165.339996 | 164.689987 |
| 10 | NAND4xp25 | 370 | 369 | A / A | 15.5900002 / 15.5900002 | 180.929993 | 180.279984 |
| 11 | NAND2xp33 | 373 | 372 | B / B | 13.2399998 / 13.2399998 | 194.169998 | 193.519989 |

## Where mapping differs

Position7 implements the same function, baseline AIG385 vs candidate AIG365. The predecessor implements baseline complement(AIG176) vs candidate complement(AIG339); CEC proves equivalence. Its arrival stays108.329994 ps. All four frontier functions of position7 are equivalent across variants, but their input pin assignments change:

| Baseline frontier function | Candidate equivalent | Baseline pin | Candidate pin |
|---|---|---|---|
| AIG139 | AIG338 | A | A |
| AIG213 (R9 region output) | AIG322 | B | D |
| AIG238 | AIG340 | C | B |
| complement(AIG176), selected critical input | complement(AIG339) | D | C |

Thus this is a downstream **pin-assignment mapping change**, not a new critical-path cell type or a forced AIG boundary. The first4, remaining3 intermediate output identities and all other selected predecessor/pin correspondences are CEC verified as well; no UNMATCHED critical gate is required for this conclusion.

## Reconvergence and switching

At position7, the source-restricted reachable competitors are:

| Input branch | Baseline input arrival | Baseline pin contribution | Candidate input arrival | Candidate pin contribution |
|---|---:|---:|---:|---:|
| R9 output213/322 | 109.679993 | B: +16.7299995 →126.409988 (float32) | 107.059998 | D: +18.1800003 →125.239998 (float32) |
| Selected176/339 branch | 108.329994 | D: +18.1800003 →126.509995 | 108.329994 | C: +17.5300007 →125.859993 |

The other two inputs are source-unreachable. The selected176/339 branch remains maximum in both variants: baseline gap about0.100006 ps, candidate gap about0.619995 ps. In all later CEC-aligned trace gates the selected predecessor function is unchanged. **No logical critical traceback switching is observed for this pair.** A physical input pin changes; that alone must not be called switching to a different logical branch. This does not assert that every other path/pair in the network is unchanged.

## Numerical explanation

Every chosen arc except position7 is identical. Its printed arc difference is17.5300007−18.1800003=−0.6499996 ps. Float32 accumulation along the two complete traces gives194.1699981689453125 and193.519989013671875 ps respectively, delta−0.6500091552734375 ps. The roughly0.0000096 ps difference from the isolated printed arc subtraction is ordinary accumulated float32 rounding, not unexplained cell delay. No per-AIG-node additive delay decomposition is assumed.

The gain is therefore numerically explained by a changed pin arc on a corresponding downstream selected cell, followed by propagation along the same logical critical chain. This is an observational/numerical explanation; no counterfactual remapping or intervention was performed to isolate why the mapper chose that permutation among all alternatives.

Raw alternative arcs are in `R9_TRACE_FANIN_ALTERNATIVES.csv`; `R9_mechanism_evidence.json` records proofs, correspondence, environment checks and region-cover membership. Full mapped state is retained under the reproduction run.
