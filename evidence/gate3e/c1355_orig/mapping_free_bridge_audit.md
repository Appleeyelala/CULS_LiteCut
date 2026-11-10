# Gate 3E mapping-free bridge audit

- original AIG: `/data/wushangge/runs/CULS-gate3e-20260922/input/c1355_orig.aig`
- checkpoint: `/data/wushangge/runs/CULS-gate3e-20260922/cutlib/best.pt` (b3f499d2973bdb36fb42690794e9838d4c8359f05d6e141aa8cbe2b9fe3ae64d)
- feature implementation: `/data/wushangge/runs/CULS-gate3e-20260922/cutlib/src/data_preprocess/feat_extraction_v3.py:CompleteCutFeatureExtractor`
- model implementation: `/data/wushangge/runs/CULS-gate3e-20260922/cutlib/src/extract_cutlib_student/modeling.py:StudentModel`
- replacement materialization: Gate 2A `materialize_candidate_subgraph`
- ABC mapping called: `false`
- STA called: `false`
- training called: `false`
- prediction field: `predicted_local_abc_delay`
