# Gate 1C-R: Multi-Decision Batch Replay Repair

**Status: PASS**

## 1. Singleton state
All forced evaluation/build fields are indexed by decision (`MAX_FORCED_DECISIONS=4`). `GATE1C_SPEC="146:0:0,226:0:0"` produces independent `decisions[]` records.

## 2. Full-correspondence semantics
`ReplaceSubgr()` treats `match[rt] == root` as legitimate full correspondence: no new node and no self-cycle are installed. The decision is recorded as `candidate_selected=true`, `installation_kind=NO_OP_FULL_CORRESPONDENCE`. Physical candidates use `PHYSICAL_REPLACEMENT`.

Candidate audit: root 146 has sub0 no-op and sub1 physical; root 226 sub0-sub4 are physical.

## 3. CEC harness audit
`AIGMan::readFromMemory()` now preserves the AIGER symbol/comment suffix and `saveFile()` writes it back. The c1355 raw and no-rewrite roundtrip both have 41 PIs, 32 POs, 502 ANDs, and CEC PASS. The earlier 37597 PI artifact came from selecting `dft_orig.aig` as the normalization input; that file header itself declares 37597 PIs.

## 4. Regression
Gate1A sub0 and sub2: CEC PASS. d1 sub1 and d2 sub0: CEC PASS.

## 5. Batch combinations
AC, AD, BC, BD were each read from the same original c1355 snapshot and replayed independently. All four have 41/32 I/O, two selected decisions, no unexpected roots, dependency `146 -> 226`, and CEC PASS. AC/AD use d1 no-op + d2 physical. BC/BD use physical replacements for both d1 and d2, satisfying the structural-modification requirement.

## 6. Determinism
BC replayed twice (Slurm job 43743). Provenance/candidate/installation semantics and output SHA256 were identical.

Therefore the implementation supports the joint candidate configuration `C=(c_1,c_2)` for two compatible, snapshot-defined decisions.
