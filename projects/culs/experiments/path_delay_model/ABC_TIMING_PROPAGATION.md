# ABC timing propagation: minimal source-grounded model

This note summarizes the active code paths audited in [`ABC_TIMING_AUDIT.md`](ABC_TIMING_AUDIT.md). It distinguishes the fixed-delay mapped-network tracer used by `print_delay` from Liberty-table STA used by `stime`.

## Minimal mapped DAG

```text
PI(s) -> n1 -> n2 -> PO
               ^
        other fanin(s)
```

ABC stores rise and fall separately. A scalar recurrence is only a shorthand; the real implementation also considers transition phase and mapped cell timing arcs.

## `print_delay`: fixed MIO pin-delay propagation

`Abc_NtkDelayTrace` seeds CI arrival times from the network timing manager, clears internal arrivals, obtains the mapped logic in DFS/topological order, and calls `Abc_NodeDelayTraceArrival` once per node. For a mapped pin with fixed rise/fall block delay and positive/non-inverting sense, the code includes terms of the form:

```text
Aout_rise = max(Aout_rise, Ain_rise + delay_pin_rise)
Aout_fall = max(Aout_fall, Ain_fall + delay_pin_fall)
```

For an inverting timing phase it also considers cross-transition terms:

```text
Aout_rise = max(Aout_rise, Ain_fall + delay_pin_rise)
Aout_fall = max(Aout_fall, Ain_rise + delay_pin_fall)
```

Non-unate pins admit both sets. The node's output arrival is therefore the max over all applicable fanin-pin and input-transition contributions. These fixed pin delays come from the mapped MIO gate representation and are cast to float in the propagation routine; no slew/load lookup is performed there.

For PO `o`, the driver node's worst transition is used in the network-wide maximum:

```text
D_global = max over CO drivers and rise/fall of A(driver)
```

The optional PI name to `print_delay` affects backward path selection/reachability, not the forward arrival seeds or recurrence. This distinction is why the final displayed node arrival cannot be interpreted as a PI-restricted endpoint delay.

## `stime`: Liberty-table arrival and slew propagation

The SCL timing pass computes output load, initializes CI timing, walks nodes in topological order, and copies each driver arrival to its CO. For each mapped gate input, it obtains the corresponding `SC_Timing` arc and evaluates the Liberty cell-rise/cell-fall tables at that input transition's slew and the driven gate's output load. Positive, negative, and non-unate senses determine which input transition feeds each output transition. In shorthand:

```text
Aout_rise = max over applicable arcs/transitions (Ain_transition + cell_rise(slew_in, load_out))
Aout_fall = max over applicable arcs/transitions (Ain_transition + cell_fall(slew_in, load_out))
Sout_rise/fall = max over applicable arcs/transitions transition_table(slew_in, load_out)
```

The actual implementation is `Abc_SclTimeFanin` → `Scl_LibPinArrival` → `Scl_LibLookup`. The lookup bilinearly interpolates/extrapolates the float surfaces. Thus the edge delay is not a fixed scalar independent of context: changing a cell, fanout/load, or upstream slew can change downstream arc delay and slew.

The report's global delay is the maximum output arrival maintained by the SCL output queue. For the normal `stime` call, ABC then walks the mapped nodes in reverse topological order and propagates `pDepts`, the downstream departure/depth quantity. SCL's slack helper derives slack from global delay minus arrival plus downstream depth; it does not use a separately exported `Abc_Time_t` required-time pass.

## Why the saved AIG path is not the mapped timing path

The Gate 3B path contains original AIG node identities. After `map`, ABC's timing graph is the mapped standard-cell network. `stime` propagates over that mapped graph; it does not retain a correspondence to the original AIG node sequence. `print_delay` also traverses mapped cells and prints mapped-gate names. Therefore an endpoint objective can be tied to stable PI/PO identities, but the original AIG path cannot be called the exact mapped path without an explicit mapping correspondence.

## Implication for downstream-only updates

The source-grounded state to update is the mapped network timing state. If only a mapped gate/arc changes while topology, cell loads, and input slews remain valid, the timing recurrence can be reevaluated in topological order over the affected fanout cone, including every fanin at reconvergent nodes. But a rewrite can alter the mapped topology/cell assignment and therefore loads and slew tables; those may require load recomputation and local re-mapping before downstream arrivals are meaningful. A single replacement scalar at an MFFC root is not equivalent to the SCL timing state.

ABC has incremental dirty-level/fanout update code used by sizing routines, but `stime` itself performs a full pass and frees its manager. Reusing incremental SCL state is a separate implementation stage; this note does not implement it.
