# ABC `map`: subject graph, cuts, matching, timing, cover

Audited source: `/home/wushangge/AIGRewrite/abc`, actual clean HEAD `c738ed6e86e6ca10bcef0d4c649cc07d925b4be8` (verified 2026-10-03). Frozen binary SHA256 remains `9b64d35ba403df4b1a2474c4591abdfd298e9a2fe1411f6ba16ae9096b6e2b2f`. Experiments use the independently built, previously verified `abc_pair_timing` binary as the reference and that source plus read-only mapper instrumentation as the observed binary.

## Actual call chain

`Abc_CommandMap` → `Abc_NtkMap` → `Abc_NtkToMap` / `Map_ManCreate` / `Map_NodeAnd` → `Map_Mapping` → `Map_MappingCuts` → `Map_MappingTruths` → `Map_MappingMatches` → `Map_MatchNodePhase` → `Map_MatchNodeCut` → `Map_TimeCutComputeArrival` / `Map_MatchCompare` → `Map_MappingSetRefs` → area recovery passes → `Abc_NtkFromMap` / `Abc_NodeFromMap_rec` / `Abc_NodeFromMapPhase_rec` / `Abc_NodeFromMapSuper_rec` → `Map_ManFree`.

| Stage | Source relative to ABC | Functions | State |
|---|---|---|---|
| Command and defaults | `src/base/abci/abc.c` | `Abc_CommandMap` (line 19969 in original source) | `Abc_Ntk_t`; default area recovery=1, sweep=0, switching=0, delay target=-1 |
| Library / manager interface | `src/base/abci/abcMap.c` | `Abc_NtkMap`, `Abc_NtkToMap` | `Map_Man_t`, `Mio_Library_t`, `Map_SuperLib_t`; input arrivals copied to manager |
| Subject graph | `abcMap.c`, `src/map/mapper/mapperCreate.c` | `Map_NodeAnd`, `Map_NodeSetAigId`, `Map_ManCreateAigIds` | AIG object's `pCopy` → mapper node, manager `pAigNodeIDs[Num]` → original ABC object ID |
| Cut enumeration | `src/map/mapper/mapperCut.c` | `Map_MappingCuts`, `Map_CutCompute`, `Map_CutMergeLists`, `Map_CutMergeTwo`, `Map_CutSortCuts`, `Map_CutFilter` | `Map_Node_t.pCuts`; `Map_Cut_t.ppLeaves`, `nLeaves`, `nVolume`, `Phase`, `pOne/pTwo` |
| Truth / canonical match table | `src/map/mapper/mapperTruth.c`, `mapperCanon.c`, `mapperTable.c` | `Map_TruthsCut`, `Map_TruthsCutOne`, `Map_CanonComputeFast`, `Map_SuperTableLookupC` | Actual cut truth; `M[0/1].pSupers`, canonical input phase mask `uPhase` |
| Per-cut implementations | `src/map/mapper/mapperMatch.c` | `Map_MatchNodeCut` | Linked `Map_Super_t` candidates; input masks `uPhase XOR super.uPhases[i]`; best `Map_Match_t` |
| Rise/fall arrival | `src/map/mapper/mapperTime.c` | `Map_TimeCutComputeArrival` | `Map_Time_t` float Rise/Fall/Worst; `tDelaysR/F[leaf]`, leaf `tArrival[phase]`; optional node delay |
| Best cut / match | `mapperMatch.c`, `mapperRefs.c`, `mapperTime.c` | `Map_MatchCompare`, `Map_MatchNodePhase`, phase reconciliation, `Map_CutGetAreaFlow`, exact ref/deref area functions, required propagation | `pCutBest[2]`, `M[phase].pSuperBest`, `uPhaseBest`, `AreaFlow`, `tArrive`, node `tArrival[2]` |
| Selected cover | `src/map/mapper/mapperRefs.c` | `Map_MappingSetRefs`, `Map_MappingSetRefs_rec` | Reset `nRefAct[0,1,2]`; PO-driven recursion through **selected cut leaves** only |
| Final network | `src/base/abci/abcMap.c` | `Abc_NtkFromMap`, `Abc_NodeFromMap_rec`, `Abc_NodeFromMapPhase_rec`, `Abc_NodeFromMapSuper_rec` | Instantiate selected phases / supergate trees, cache `Map_NodeReadData`; only implemented outputs receive `vOrigNodeIds` |

Both input AIGs are already strashed when `map` is called. The command's alternate pre-strash/balance branch is not used. The observed mapper is the classic ABC `Map_Man_t` supergate mapper, and all requested state is live before `Map_ManFree`; no lifecycle extension is needed.

## Cut and match limits: actual code

This library derives a 5-input supergate table through `Map_SuperLibDeriveFromGenlib` → `Super_PrecomputeStr(pLib, 5, 1, ...)` (`mapperLib.c`). Cuts are merged from fanin cuts and deduplicated. `MAP_CUTS_MAX_COMPUTE=1000`, `MAP_CUTS_MAX_USE=250`; sorting's actual comparator compares leaf count (the nearby comment about average arrival is not its implementation). Up to 249 nontrivial cuts plus a trivial self cut are retained before dominance filtering. `Map_CutFilter` removes cuts whose support contains an earlier smaller support. Instrumentation records the target node's sorted-before-limit and dominated events.

`Map_TruthsCutOne` produces the truth using each cut's actual ordered leaves; both output polarities are canonicalized and looked up. The current derived supergate table does not have an implementation for every Boolean function. For the fixed four-leaf cut, **both variants retain the cut and both phase lookups have zero matches**. It was not lost to cut limits or dominance. ABC maps the function by choosing cuts that terminate at intermediate functions instead.

`Map_MatchNodeCut` considers at most the first 30 library-list entries, then their available input phase assignments. `uPhaseBest` bit i=0 selects leaf positive-phase timing; bit i=1 selects its negative phase. Pin permutation is represented by each supergate tree's terminal leaf index, exported as `implementation`, rather than an invented separate permutation field. The four transition-delay entries per pin (`rr,fr,rf,ff`) come directly from `Map_Super_t.tDelaysR/F`.

Timing starts from the manager's normal PI arrivals (all PIs); output rise/fall each maximize applicable pin contributions. During a limit-pruned attempt the original routine can return early and leave partial rise/fall fields, so those are recorded as `partial_or_saved_arrival`, never as a complete match arrival. Area-pruned attempts do not receive a new timing calculation. `AreaFlow` is area flow in delay/flow passes and an exact area cost in the corresponding ref/deref recovery passes; it is not simply the cell area. Supergate area is exported separately. Load/slew is not a stored per-match dynamic load quantity in these structs; `estimated_load` is `NOT_AVAILABLE`. Arrival values here are mapper all-PI estimates, not `print_pair_delay` source-restricted endpoint measurements and not Liberty `stime`.

## Why a timed node can be absent from final cover

Forward matching computes a best implementation for every subject function point, including nodes with zero final references. `Map_MappingSetRefs_rec` does not follow all subject fanins: it follows the selected best cut's leaves. Internal subject nodes between a selected output and its leaves can have `nRefAct=[0,0,0]` even though they have complete best timing state. Final construction instantiates the selected cover, not every independently optimized subject node. The R9 experiment supplies exact selected-cover witnesses for this behavior.

## Native choices support

**YES**, current `map` supports AIG choices. `Abc_NtkToMap` copies choice chains with `Map_NodeSetNextE` and representatives with `Map_NodeSetRepr`; `Map_MappingSetChoiceLevels` aligns levels (`mapperUtils.c`). `Map_CutCompute` unions equivalent-node cuts into the representative and corrects phase. `Map_MappingMatches` skips secondary nodes with `pRepr` and matches the representative. The observed experiment constructs no choices and uses the existing unmodified subject graphs.

See [experiment](MAPPER_INTERNAL_EXPERIMENT.md), [comparison](results/R9_mapper_internal_comparison.md), and [source patch](abc_mapper_debug.patch).
