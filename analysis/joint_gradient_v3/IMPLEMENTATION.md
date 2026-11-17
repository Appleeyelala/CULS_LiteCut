# Joint-gradient v3 research implementation

This branch implements the approved no-internal-pin-alignment route. It does
not replace the production experiment entry point or historical v2 results.

## Representation and differentiation

`global_timing/phase_ports.py` retains positive/complement **logical literal**
channels at cut boundaries. These are not electrical rise/fall states. Candidate
output and consuming-edge complements compose by XOR. This resolves all 39
previous discrepancies: 180 saved installed combinations and 12 original graphs
match explicit literal composition exactly; compressed paths differ by at most
1.81e-10 structural units (tolerance 1e-8). This is structural correctness on the
tested domain, not evidence of mapped-timing prediction accuracy.

`conditional_arrival.py` batches edges by topological level. Each option uses an
LSE over its input ports; a root mixes its original and alternative transfers.
The custom first-order reverse tape stores edge derivatives and option arrivals.
It supports Torch CUDA without one autograd copy of the complete graph per
level. A differentiable reference checks both selection and parameter gradients.
Second derivatives are unsupported. This is not a new fused native CUDA kernel.

`learned_ports.py` provides four matched ablations: fixed structural ports,
learned ports, global residual only, and both. The port MLP has two 64-unit SiLU
layers; a Softplus multiplier gives nonnegative effective port cost, initialized
to the structural prior. Only replacement transfers receive learned multipliers;
untouched transfers retain additive structural costs so inserting unused cut
boundaries does not change their learned cost. Wires retain zero transfer cost. The input comprises
the frozen Student's 19 features (signed log1p / 10), local-delay ratio, path
cost, source/output phase, candidate flag, and eight context features. Context
uses two fixed reverse Region-DAG mean aggregations of original region size,
boundary size, fanout and downstream depth plus expected edit features.

The independent residual head has two 32-unit SiLU layers. It sees original PO
timing/structural features, global edit summaries and PO-region context. Both
heads are shared across circuits. Output is `original_PO + F(x) - F(noop)`;
global edit summaries use circuit size, never the number of retained candidates,
so pruning or adding unused choices does not dilute edit features.
anchoring does not prove accuracy. Candidate measurements never enter forward
features. The original gate-count area estimator is retained and real area
violations remain independently reported. Soft conflicting mixtures are gradient
extensions only; actual selections must pass `CandidateFeasibility` and Replay.

## Learning and validation protocol

`learning_manifest.json` freezes the historical family/hash split, moving all
families of the previous 12 development circuits to development-only. The first
learning pilot is source/size-stratified, one family per selected circuit, with
12 training and 12 validation circuits of at most 64,000 AND nodes. This scope
does not establish large-circuit generalization. Existing 905 aggregate results
were already inspected, so the historical split is not described as pristine.

`joint_gradient_learning_data.py` samples a reservoir of 128 decisions from the
full verified domain, up to two candidates per decision, then applies candidate
feasibility. It freezes at most eight unique nonempty combinations, including
single, alternative, serial, parallel and legal joint examples where available.
Sampling never uses candidate timing or the historical optimizer's choice.
One serialized no-op per circuit measures serialization/remapping drift.
The total ceiling is 216 new Mapping calls. Missing/failing cases are retained;
they are not replaced by successful examples. Existing fully measured historical
combinations are imported only after the new domain is frozen.

`train_joint_gradient_ports.py` refuses unresolved structural discrepancies.
Complete CEC-PASS timing labels remain eligible even when area exceeds 1.05;
area feasibility is a separate flag. No-op replay drift is reported rather than
forced to equal the original Mapping. The loss combines PO-delta Huber,
Critical-delta Huber and 0.1 pairwise logistic loss, normalized at 1% of original
delay. Each update averages within a circuit. AdamW uses lr 1e-3, weight decay
1e-4, up to 200 epochs and validation patience 20; seeds 20261117/18/19.
Neural heads train in float32, timing propagation in float64. No Student training.

Validation reports rank ties, undefined correlations, predicted-improvement
coverage, precision, area and exclusions. For the expansion gate an undefined
correlation contributes zero; median Spearman must reach 0.5 and improvement
precision 0.7. This stricter tie treatment is not numerically interchangeable
with v2's exclusion of undefined correlations. No successful gate is asserted
until actual evaluation completes.
Before fitting, aggregation is fixed as follows: all three seeds must finish;
their median rank and precision metrics must pass, with at least one seed
passing both individually. Ports or combined must pass to expand; global-only
is an explanatory control. Primary development optimizer comparisons use seed
20261117. An independent-stage checkpoint, if allowed, uses lowest validation
loss among passing seeds, preferring combined when both structural variants pass.

`evaluate_joint_gradient_v3.py` freezes aligned-STE and joint-gradient searches
with the existing budgets on the old factorial domain, without reading candidate
measurement files. Only afterwards may results be joined with saved real QoR.
The independent 24-circuit/72-Mapping phase remains conditional on the gates.
No 100/905 run is automatic.

## Verification and accounting

Slurm 66000: 16 tests pass, including polarity cancellation, serial composition,
downstream context, finite differences, parameter/selection gradients against
the reference, GPU/CPU parity, mixed-precision learning and existing optimizer
budget/critical-objective tests and unused-candidate feature invariance.
Slurm 65995: full 192-row structural audit passes.
Earlier wrapper failures and the floating-point anchor regression are preserved
in run logs; they are not successful experimental measurements.

All source snapshots, binary/input hashes, commands, Slurm allocation, raw AIGs,
weights and logs are recorded under `/data`. Search/training timings are not
complete end-to-end deployment timings. Independent candidate Mapping is a
research label cost, never feedback to a deployed optimizer.
