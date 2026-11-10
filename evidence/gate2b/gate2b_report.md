# Gate 2B — Local LiteCut vs Global STA Interaction Validation

Status: **PASS**

This run used the already validated Gate 1C-R AIGs and did not change
candidate semantics, the LiteCut checkpoint, or Gate 1C-R.  The four joint
configurations were independently read from the same normalized original
snapshot.  Existing Gate 1C-R CEC logs report `Networks are equivalent` for
AC, AD, BC, and BD.

## Frozen flow

All five inputs used the same cutLib ABC executable and the same ASAP7 library:

```text
ABC: /data/public/AIGRewrite/abc/abc
version: UC Berkeley ABC 1.01 (compiled Aug 6 2025 14:07:48)
library: /home/wushangge/cutLibOrigin/lib/asap7.lib
library SHA256: 819ecfc2609a1f53b9d298bf0cba75af0f1a8c737f33d4bbc5c9fb1156e8fa55
commands: read <library>; read <input_aig>; map; topo; stime
```

The library was byte-for-byte staged into the Slurm execution workspace because
compute nodes do not mount the source `/home` path.  The execution copy has the
same SHA256.  Successful Slurm job: `44300` (2 CPUs, 4G, 30 minutes,
`OMP_NUM_THREADS=1`).  Full commands, environment, library manifest, and raw
logs are in this directory.

## Input provenance

| input | SHA256 | Gate1/CEC status |
|---|---|---|
| `normalized_original.aig` | `88df0a75b9328378b869843ba675c316d05a8bd812b7de22effc12af51f430ea` | baseline |
| `AC.aig` | `ff95290981fb5ff09ed5f2b68c04a92d51988457a4825fd0753cbb6630b5df02` | PASS |
| `AD.aig` | `4a50555c3ca2e08863a73929fa28b9e3d66ce2bf86535e3c0e556b23bd45e1ad` | PASS |
| `BC.aig` | `d2aaeada2c99ca6790fbd9cccdf79446872128288ac9a950fccdb9d557983498` | PASS |
| `BD.aig` | `082cae2fe9ef68eed84a2ecba6e39b253a6ae9b8aa8a6cd33a9f6fa0f75fa59e` | PASS |

The candidate labels remain: A = `NO_OP_FULL_CORRESPONDENCE`, B/C/D =
`PHYSICAL_REPLACEMENT`.  No candidate was replaced or redefined.

## Metric semantics

The reported whole-design values below are the `Area = ... Delay = ... ps`
summary emitted by the frozen `map; topo; stime` flow.  They are distinct from
the four LiteCut-predicted local `abc_delay` values.  Local LiteCut scores were
not added and were not used as a global timing model.

| configuration | d1 | d2 | whole-design delay (ps) | area | mapped gate count |
|---|---:|---:|---:|---:|---:|
| original | — | — | 263.32 | 557.07 | 432 |
| AC | A | C | 263.60 | 557.54 | 435 |
| AD | A | D | 263.60 | 554.74 | 435 |
| BC | B | C | 263.60 | 556.84 | 436 |
| BD | B | D | 263.60 | 554.74 | 435 |

Area and gate count are recorded only as auxiliary metrics.  Relative to the
original, every joint configuration has a reported delay change of `+0.28 ps`.

## Repeatability

AC, AD, BC, and BD were each run twice.  Delay, area, and mapped gate count
were identical between runs for every configuration.  Thus the observed
run-to-run delay noise is `0.00 ps`.  ABC prints delay to two decimal places,
so comparisons use an effective tolerance of `0.01 ps` (the maximum of the
repeatability noise and the reported precision).

## Context effects and interaction

```text
Delta1(C) = D_BC - D_AC =  0.00 ps
Delta1(D) = D_BD - D_AD =  0.00 ps
Delta2(A) = D_AD - D_AC =  0.00 ps
Delta2(B) = D_BD - D_BC =  0.00 ps

I = D_AC + D_BD - D_AD - D_BC = 0.00 ps
```

Both identities, `I = Delta1(D) - Delta1(C)` and
`I = Delta2(B) - Delta2(A)`, hold exactly for the parsed values.  The measured
interaction is therefore zero and is not meaningful relative to the 0.01 ps
comparison tolerance.  This experiment does not resolve effects below the
reported timing precision.

There is no context-dependent ranking reversal: all d1 comparisons (A vs B
under C/D) and all d2 comparisons (C vs D under A/B) are
`TIE_WITHIN_TOLERANCE`.

## LiteCut comparison and requested questions

LiteCut local predictions were A=`29.5364680412`, B=`24.0842151795`,
C=`20.5742577435`, D=`35.4938270155` (all are local predictions only).  Thus
the independent local-greedy choice is **BC**.

1. **Q1:** Yes, LiteCut local greedy remains BC.
2. **Q2:** `D_AC = D_AD = D_BC = D_BD = 263.60 ps` (baseline
   `D_original = 263.32 ps`).
3. **Q3:** BC is tied for the lowest measured whole-design delay among the
   four configurations; it is not a unique winner.
4. **Q4:** No d1 or d2 context-dependent ranking reversal was observed.
5. **Q5:** `I = 0.00 ps`; with zero repeatability drift and 0.01 ps reported
   precision, it has no interpretable interaction magnitude in this test.

The narrow conclusion is that this compatible pair shows no measurable global
timing interaction or ranking reversal under this fixed mapping/STA flow.  It
does not establish that local greedy selection is generally globally optimal,
and it does not validate any predictor or gradient method.
