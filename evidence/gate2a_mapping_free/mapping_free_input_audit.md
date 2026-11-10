# Gate 2A mapping-free LiteCut Student input audit

Status: **PASS_MAPPING_FREE_INPUT_AND_INFERENCE_WITH_D1_NOOP_CAVEAT**

This adapter uses the canonical `CompleteCutFeatureExtractor` and the restored Student checkpoint. It does not call ABC mapping, STA, or training.
This is mapping-free input representability only; it does not revise the earlier post-mapping correspondence audit (`NOT_ALIGNABLE`).

## Student input semantics

`StudentModel.forward(x_scaled)` receives exactly 19 continuous values, split as group1 `[0:8]`, group2 `[8:14]`, group3 `[14:19]`. The WL hash is retained by the dataset format but is not an argument to Student.forward and therefore does not affect this prediction path. Columns 20--24 (Laplacian features) are emitted by the canonical extractor but are also outside the Student tensor.

## Feature audit

| feature | source before mapping | changes d1 A→B | Student input | post-mapping dependency |
|---|---|---:|---:|---:|
| R_root | original feat_graph + fixed cut | no | yes | no |
| SlackProxy_root | original feat_graph + fixed cut | no | yes | no |
| Root_Global_Fanout | original feat_graph + fixed cut | no | yes | no |
| Max_L_leaf | original feat_graph + fixed cut | no | yes | no |
| Max_Leaf_Fanout | original feat_graph + fixed cut | no | yes | no |
| Mean_Leaf_Fanout | original feat_graph + fixed cut | no | yes | no |
| Std_Leaf_Fanout | original feat_graph + fixed cut | no | yes | no |
| Frac_Leaf_PI | original feat_graph + fixed cut | no | yes | no |
| Weighted_Depth | materialized replacement DAG | no | yes | no |
| Topo_Depth | materialized replacement DAG | no | yes | no |
| Inv_Penalty | materialized replacement DAG | no | yes | no |
| Node_Count | materialized replacement DAG | yes | yes | no |
| Avg_Width | materialized replacement DAG | yes | yes | no |
| Max_Internal_Fanout | materialized replacement DAG | yes | yes | no |
| Inv_Ratio | materialized replacement DAG | yes | yes | no |
| Root_Inv_Count | materialized replacement DAG | no | yes | no |
| Chain_Density | materialized replacement DAG | yes | yes | no |
| Fork_Density | materialized replacement DAG | yes | yes | no |
| Avg_Curvature | materialized replacement DAG | yes | yes | no |
| Lap_Eig1 | materialized replacement DAG | no | no | no |
| Lap_Eig2 | materialized replacement DAG | yes | no | no |
| Lap_Eig3 | materialized replacement DAG | no | no | no |
| Lap_Eig4 | materialized replacement DAG | yes | no | no |
| Lap_Eig5 | materialized replacement DAG | yes | no | no |
| WL_Hash | materialized replacement DAG | yes | no | no |

## Candidate results

| candidate | installation kind | raw feature difference | predicted raw abc_delay | repeat check |
|---|---|---:|---:|---|
| d1_sub0 | NO_OP_FULL_CORRESPONDENCE | finite | 29.5364680411754 | see repeat run |
| d1_sub1 | PHYSICAL_REPLACEMENT | finite | 24.0842151795256 | see repeat run |
| d2_sub0 | PHYSICAL_REPLACEMENT | finite | 20.5742577434841 | see repeat run |
| d2_sub1 | PHYSICAL_REPLACEMENT | finite | 35.4938270155396 | see repeat run |

A/B labels follow Gate1 provenance: A (`d1_sub0`) is a legitimate `NO_OP_FULL_CORRESPONDENCE`, not a physical modification; B (`d1_sub1`) is `PHYSICAL_REPLACEMENT`. No relabeling was performed.

Independent repeat directory: `gate2a_mapping_free_repeat_final`; exact JSON equality for all emitted candidates: **True**.
