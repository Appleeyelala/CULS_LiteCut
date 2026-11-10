# Gate 2A mapping-free Student input audit

The adapter in `gate2a_candidate_inference.py` was run against the Gate1
provenance records for d1 subgraphs 0/1 and d2 subgraphs 0/1. It uses the
canonical `CompleteCutFeatureExtractor.extract_all` and the repository's
canonical `validate.eval_prediction_accuracy.load_student_model` loader. No
ABC mapping, STA, training, or dataset generation was run.

This is a **mapping-free input representability** result. It does not revise
the earlier Gate2A-R post-mapping correspondence audit (which was
`NOT_ALIGNABLE`).

Checkpoint: `/home/wushangge/cutLibOrigin/best.pt`  
SHA256: `b3f499d2973bdb36fb42690794e9838d4c8359f05d6e141aa8cbe2b9fe3ae64d`

## Input semantics

The extractor emits 25 columns in the order recorded in
`gate2a_mapping_free_out/mapping_free_input_audit.json`. Student inference
consumes exactly columns 1--19 as `x_raw`, standardized with the checkpoint's
`x_mu`/`x_inv_sigma`, then passes `x_scaled` to `StudentModel.forward`.
The model splits this tensor into group1 `[0:8]`, group2 `[8:14]`, and group3
`[14:19]`. The five Laplacian columns and WL hash are emitted canonical
features but are not arguments to the current Student `forward`; WL is only a
dataset auxiliary field in this model family.

All 25 columns are derivable before mapping from the original AIG's
`feat_graph`, the recorded fixed cut, and the materialized rewrite-library
DAG. For the full-correspondence/no-op record A, the existing Gate1 replay AIG
cone is used as that materialized DAG; for physical records the library DAG is
rebuilt directly. No feature requires mapper output. Root/context and leaf
statistics are computed against the original snapshot; candidate-local
topology, spectral features, and WL hash are computed on the materialized
candidate DAG.

## d1 A/B result

| Gate1 record | installation kind | predicted raw `abc_delay` |
|---|---|---:|
| `d1_sub0` (A) | `NO_OP_FULL_CORRESPONDENCE` | 29.53646804117542 |
| `d1_sub1` (B) | `PHYSICAL_REPLACEMENT` | 24.0842151795256 |

Both 25-column vectors and 19-dimensional vectors are finite, and A/B differ
in candidate-local dimensions (Node_Count, Avg_Width, Max_Internal_Fanout,
Inv_Ratio, Chain_Density, Fork_Density, Avg_Curvature). Two independent runs
produce byte-for-byte identical JSON and predictions for both candidates.

The existing Gate1 d1 pair is **not two physical candidates**: provenance
classifies subgraph 0 as a legitimate full-correspondence/no-op and subgraph 1
as the physical replacement. This report preserves that classification rather
than relabeling A.

## d2 expansion

The same mapping-free path also produced finite predictions for `d2_sub0`
(20.574257743484107) and `d2_sub1` (35.493827015539594). These are both
`PHYSICAL_REPLACEMENT` according to Gate1 provenance.

An independent repeat of all four records (`gate2a_mapping_free_repeat_final/`)
is byte-for-byte identical, including raw features, scaled inputs, model output,
and inverse-transformed prediction.

No whole-design timing or global-delay claim is made here. The complete raw
records, replacement DAGs, scaled vectors, model outputs, inverse transforms,
and checkpoint hash are in `gate2a_mapping_free_out/`.
