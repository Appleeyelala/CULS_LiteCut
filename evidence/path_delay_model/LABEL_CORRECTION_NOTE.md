# Correction of the earlier endpoint labels

The old `print_delay po11 pi03` output was interpreted as a PI4→PO11 pair delay. ABC propagated arrival from **all** PIs; `pi03` only constrained the printed traceback. The displayed 194.4 ps is the all-PI PO arrival (`194.429977` internally), **not** a validated source-restricted pair delay. The corrected baseline under the same mapped MIO model is `D(pi03→po11) = 194.169998 ps`.

Do not use the old 15-candidate or 67-candidate `single_candidate_calibration.csv` labels, Pearson/Spearman, MAE, sign accuracy, region scaling, or path-dependence analyses as formal candidate→PI/PO pair-timing conclusions. They are measurement-pipeline debugging history; files are preserved unchanged. The new 16-candidate pilot and 32-path baseline matrix use `print_pair_delay` with explicit source restriction.

This correction does not turn the Gate3B original AIG node sequence into a mapped exact path: the new objective is the maximum mapped delay from the specified PI to the specified PO. It also does not make the fixed-MIO model equivalent to Liberty `stime`, nor to LiteCut's local cone proxy.
