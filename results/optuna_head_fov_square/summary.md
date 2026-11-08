# RETFound-Green: frozen backbone + Optuna-tuned head

Nested: PAPILA official 5 folds outside, 3-fold patient-grouped CV inside; 40 trials per study; final head = mean of 3 seeds. Threshold: Youden's J on inner predictions of the training data; frozen from all PAPILA for Chákṣu. Inner AUROC is the selection score (best of many trials), not a performance estimate.

## PAPILA (out-of-fold)

| Setting | n (glaucoma) | AUROC | AUPRC [guessing] | Sensitivity | Specificity | Precision (PPV) | Sens. at 90% spec. |
|---|---|---|---|---|---|---|---|
| Optuna head (frozen) | 420 (87) | 0.844 (0.78–0.89) | 0.618 (0.48–0.75) [0.21] | 0.759 (0.63–0.86) | 0.769 (0.71–0.82) | 0.462 (0.35–0.56) | 0.621 (0.41–0.74) |
| Logistic regression (step 1) | 420 (87) | 0.851 (0.79–0.90) | 0.625 (0.48–0.75) [0.21] | 0.747 (0.63–0.85) | 0.727 (0.67–0.78) | 0.417 (0.31–0.51) | 0.609 (0.46–0.75) |

| Setting | TP | FP | FN (missed) | TN | NPV | PPV here → PPV in screening |
|---|---|---|---|---|---|---|
| Optuna head (frozen) | 66 | 77 | 21 | 256 | 0.924 | 0.46 at 21% → 0.11 at 3.5% |
| Logistic regression (step 1) | 65 | 91 | 22 | 242 | 0.917 | 0.42 at 21% → 0.09 at 3.5% |

AUROC inside each fold (mean ± SD of 5): Optuna head (frozen) 0.853 ± 0.042; Logistic regression (step 1) 0.850 ± 0.042

Optuna head minus logistic regression, AUROC on within-fold ranks: +0.003 (95% CI -0.007 to +0.014): no clear winner (0 in CI)

| Fold | Inner AUROC | Test AUROC | Best head |
|---|---|---|---|
| 1 | 0.856 | 0.839 | logistic regression, C 0.0058 |
| 2 | 0.838 | 0.829 | logistic regression, C 0.002 |
| 3 | 0.817 | 0.922 | logistic regression, C 0.0015 |
| 4 | 0.839 | 0.813 | logistic regression, C 0.0015 |
| 5 | 0.814 | 0.863 | logistic regression, C 0.0015 |

## Chákṣu external test (head tuned on all PAPILA: logistic regression, C 0.0028)

| Setting | n (glaucoma) | AUROC | AUPRC [guessing] | Sensitivity | Specificity | Precision (PPV) | Sens. at 90% spec. |
|---|---|---|---|---|---|---|---|
| All cameras | 1338 (187) | 0.692 (0.65–0.73) | 0.283 (0.23–0.35) [0.14] | 0.652 (0.58–0.72) | 0.631 (0.60–0.66) | 0.223 (0.19–0.26) | 0.257 (0.20–0.33) |
| Bosch | 138 (19) | 0.559 (0.44–0.68) | 0.173 (0.10–0.33) [0.14] | 0.211 (0.05–0.43) | 0.731 (0.64–0.82) | 0.111 (0.03–0.22) | 0.105 (0.00–0.33) |
| Forus | 126 (18) | 0.689 (0.52–0.84) | 0.415 (0.20–0.64) [0.14] | 0.556 (0.31–0.79) | 0.824 (0.75–0.89) | 0.345 (0.17–0.54) | 0.389 (0.12–0.73) |
| Remidio | 1074 (150) | 0.713 (0.67–0.75) | 0.290 (0.23–0.36) [0.14] | 0.720 (0.65–0.79) | 0.595 (0.56–0.63) | 0.224 (0.19–0.26) | 0.267 (0.18–0.34) |

| Setting | TP | FP | FN (missed) | TN | NPV | PPV here → PPV in screening |
|---|---|---|---|---|---|---|
| All cameras | 122 | 425 | 65 | 726 | 0.918 | 0.22 at 14% → 0.06 at 3.5% |
| Bosch | 4 | 32 | 15 | 87 | 0.853 | 0.11 at 14% → 0.03 at 3.5% |
| Forus | 10 | 19 | 8 | 89 | 0.918 | 0.34 at 14% → 0.10 at 3.5% |
| Remidio | 108 | 374 | 42 | 550 | 0.929 | 0.22 at 14% → 0.06 at 3.5% |

PPV in screening applies this sensitivity and specificity to a population where 3.5% have glaucoma (Tham et al. 2014): the same model raises far more false alarms when the disease is rarer.

Errors: 490 of 1338; dangerous (experts unanimous) 247, safe (experts split) 243.
