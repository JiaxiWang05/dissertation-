# RETFound-Green: frozen backbone + Optuna-tuned head

Nested: PAPILA official 5 folds outside, 3-fold patient-grouped CV inside; 40 trials per study; final head = mean of 3 seeds. Threshold: Youden's J on inner predictions of the training data; frozen from all PAPILA for Chákṣu. Inner AUROC is the selection score (best of many trials), not a performance estimate.

## PAPILA (out-of-fold)

| Setting | n (glaucoma) | AUROC | AUPRC [guessing] | Sensitivity | Specificity | Precision (PPV) | Sens. at 90% spec. |
|---|---|---|---|---|---|---|---|
| Optuna head (frozen) | 420 (87) | 0.854 (0.80–0.90) | 0.649 (0.51–0.77) [0.21] | 0.736 (0.62–0.83) | 0.829 (0.78–0.88) | 0.529 (0.41–0.64) | 0.621 (0.48–0.75) |
| Logistic regression (step 1) | 420 (87) | 0.840 (0.77–0.90) | 0.648 (0.51–0.76) [0.21] | 0.770 (0.66–0.87) | 0.808 (0.75–0.86) | 0.511 (0.40–0.62) | 0.609 (0.44–0.74) |

| Setting | TP | FP | FN (missed) | TN | NPV | PPV here → PPV in screening |
|---|---|---|---|---|---|---|
| Optuna head (frozen) | 64 | 57 | 23 | 276 | 0.923 | 0.53 at 21% → 0.13 at 3.5% |
| Logistic regression (step 1) | 67 | 64 | 20 | 269 | 0.931 | 0.51 at 21% → 0.13 at 3.5% |

AUROC inside each fold (mean ± SD of 5): Optuna head (frozen) 0.855 ± 0.033; Logistic regression (step 1) 0.856 ± 0.045

Optuna head minus logistic regression, AUROC on within-fold ranks: +0.000 (95% CI -0.012 to +0.011): no clear winner (0 in CI)

| Fold | Inner AUROC | Test AUROC | Best head |
|---|---|---|---|
| 1 | 0.862 | 0.858 | logistic regression, C 0.007 |
| 2 | 0.859 | 0.822 | logistic regression, C 0.0039 |
| 3 | 0.835 | 0.901 | logistic regression, C 0.0091 |
| 4 | 0.845 | 0.824 | logistic regression, C 0.0022 |
| 5 | 0.815 | 0.872 | logistic regression, C 0.0039 |

## Chákṣu external test (head tuned on all PAPILA: logistic regression, C 0.0024)

| Setting | n (glaucoma) | AUROC | AUPRC [guessing] | Sensitivity | Specificity | Precision (PPV) | Sens. at 90% spec. |
|---|---|---|---|---|---|---|---|
| All cameras | 1338 (187) | 0.703 (0.66–0.74) | 0.285 (0.23–0.35) [0.14] | 0.834 (0.78–0.89) | 0.384 (0.36–0.41) | 0.180 (0.16–0.21) | 0.316 (0.24–0.39) |
| Bosch | 138 (19) | 0.717 (0.60–0.82) | 0.247 (0.15–0.43) [0.14] | 0.684 (0.48–0.88) | 0.681 (0.58–0.77) | 0.255 (0.14–0.39) | 0.158 (0.00–0.46) |
| Forus | 126 (18) | 0.738 (0.58–0.87) | 0.426 (0.23–0.67) [0.14] | 0.611 (0.37–0.83) | 0.778 (0.70–0.85) | 0.314 (0.16–0.48) | 0.389 (0.19–0.71) |
| Remidio | 1074 (150) | 0.711 (0.67–0.75) | 0.293 (0.23–0.37) [0.14] | 0.880 (0.83–0.93) | 0.300 (0.27–0.33) | 0.169 (0.14–0.20) | 0.307 (0.22–0.41) |

| Setting | TP | FP | FN (missed) | TN | NPV | PPV here → PPV in screening |
|---|---|---|---|---|---|---|
| All cameras | 156 | 709 | 31 | 442 | 0.934 | 0.18 at 14% → 0.05 at 3.5% |
| Bosch | 13 | 38 | 6 | 81 | 0.931 | 0.25 at 14% → 0.07 at 3.5% |
| Forus | 11 | 24 | 7 | 84 | 0.923 | 0.31 at 14% → 0.09 at 3.5% |
| Remidio | 132 | 647 | 18 | 277 | 0.939 | 0.17 at 14% → 0.04 at 3.5% |

PPV in screening applies this sensitivity and specificity to a population where 3.5% have glaucoma (Tham et al. 2014): the same model raises far more false alarms when the disease is rarer.

Errors: 740 of 1338; dangerous (experts unanimous) 415, safe (experts split) 325.
