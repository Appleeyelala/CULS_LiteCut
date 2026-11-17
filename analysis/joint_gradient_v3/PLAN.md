# Joint gradient v3: approved execution contract

This is the implementation of the user's no-internal-pin-alignment plan. It does
not replace historical v2 evidence or change the production entry point.

1. Explain the 39/180 structural discrepancies against saved replay AIGs. Gate:
   every available hard combination must agree under the same structural model.
   No fitting may compensate for an unresolved installation discrepancy.
2. Freeze family/hash-separated data. The previous 12 circuits and their whole
   families are development-only. LiteCut is frozen. Learn conditional cut-port
   transfers from original/candidate PO and critical-delay differences, with a
   separate global residual ablation. No mapped-pin/AIG-node correspondence.
3. Shared port MLP: two 64-unit SiLU layers and positive output; global residual:
   two 32-unit layers. AdamW lr=1e-3, weight decay=1e-4, 200 epochs maximum,
   validation patience 20, three fixed seeds. Circuit-equal PO delta, critical
   delta and within-circuit ranking losses. Teacher timing is never an internal
   input. Only original PO timing/area are deployment reference labels.
4. If needed, freeze at most 8 nonempty combinations on 12 training and 12
   validation circuits and one serialized no-op each: <=216 new ABC Mapping
   calls. Reuse complete matching records. No replacement based on measured QoR.
5. Fixed-search ablations: current port model, learned ports, global-only,
   learned ports plus global. Check numeric gradients, discrete changes and real
   timing separately. Spearman median >=0.5 and improvement precision >=0.7;
   report coverage, ties, failures and area violations alongside these gates.
6. Only after development gates: freeze and evaluate 24 independent circuits,
   <=72 final-candidate mappings. No repeated A/B, no real-delay selection or
   rollback. Stop here; do not automatically launch 100/905.

All substantive computation runs through Slurm in /data with committed source
snapshots. Raw graphs, weights, libraries, logs remain there. Publish small
evidence, manifests and diagnostics here. Do not stage unrelated docs/task/.

Deployment target remains one original Mapping. Candidate Mapping is independent
research validation and is separately timed; excluding it from accounting does
not establish measured one-Mapping deployment performance.
