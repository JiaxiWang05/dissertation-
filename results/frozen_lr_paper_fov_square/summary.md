# RETFound-Green: frozen backbone + logistic regression

Binary glaucoma vs healthy. PAPILA suspects: exclude. Folds: official. Head: sklearn defaults as in the RETFound-Green paper (C = 1, no class weights). Threshold: Youden J on inner cross-validated predictions of the training data (never the test fold); for Chákṣu, the threshold from all of PAPILA is frozen. 95% CIs: 2000x cluster bootstrap (patients / eyes). AUPRC [guessing] = the score of random guessing, i.e. the share of glaucoma images. Sens. at 90% spec. is read from the ROC curve. 'Paper AUROC' = median of 100 image-level bootstrap resamples, the paper's metric.

## PAPILA, 5-fold patient-level CV (out-of-fold, pooled)

| Setting | n (glaucoma) | AUROC | AUPRC [guessing] | Sensitivity | Specificity | Precision (PPV) | Sens. at 90% spec. | Paper AUROC |
|---|---|---|---|---|---|---|---|---|
| image | 420 (87) | 0.794 (0.72–0.86) | 0.582 (0.45–0.71) [0.21] | 0.690 (0.57–0.80) | 0.811 (0.76–0.86) | 0.488 (0.37–0.60) | 0.575 (0.43–0.70) | 0.797 |
| clinical (age, sex) | 420 (87) | 0.761 (0.69–0.83) | 0.434 (0.31–0.59) [0.21] | 0.782 (0.65–0.89) | 0.634 (0.56–0.70) | 0.358 (0.26–0.45) | 0.345 (0.19–0.53) | 0.760 |
| image + clinical | 420 (87) | 0.801 (0.73–0.86) | 0.582 (0.45–0.71) [0.21] | 0.678 (0.55–0.79) | 0.787 (0.73–0.84) | 0.454 (0.35–0.56) | 0.563 (0.44–0.71) | 0.803 |

| Setting | TP | FP | FN (missed) | TN | NPV | PPV here → PPV in screening |
|---|---|---|---|---|---|---|
| image | 60 | 63 | 27 | 270 | 0.909 | 0.49 at 21% → 0.12 at 3.5% |
| clinical (age, sex) | 68 | 122 | 19 | 211 | 0.917 | 0.36 at 21% → 0.07 at 3.5% |
| image + clinical | 59 | 71 | 28 | 262 | 0.903 | 0.45 at 21% → 0.10 at 3.5% |

AUROC inside each fold (mean ± SD of 5): image 0.798 ± 0.059; clinical (age, sex) 0.772 ± 0.045; image + clinical 0.804 ± 0.051

Paired bootstrap of the AUROC difference (scores ranked within each fold):

- image minus clinical (age, sex): +0.023 (95% CI -0.054 to +0.101): no clear winner (0 in CI)
- image + clinical minus image: +0.006 (95% CI -0.004 to +0.016): no clear winner (0 in CI)

## Chákṣu external test (trained on all PAPILA, C = 1, threshold 0.404; no retraining)

| Setting | n (glaucoma) | AUROC | AUPRC [guessing] | Sensitivity | Specificity | Precision (PPV) | Sens. at 90% spec. | Paper AUROC |
|---|---|---|---|---|---|---|---|---|
| All cameras | 1338 (187) | 0.597 (0.55–0.64) | 0.187 (0.16–0.23) [0.14] | 0.021 (0.00–0.04) | 0.989 (0.98–0.99) | 0.235 (0.05–0.46) | 0.128 (0.07–0.19) | 0.599 |
| Bosch | 138 (19) | 0.488 (0.35–0.63) | 0.146 (0.09–0.26) [0.14] | 0.000 (0.00–0.00) | 1.000 (1.00–1.00) | 0.000 (0.00–0.00) | 0.105 (0.00–0.26) | 0.500 |
| Forus | 126 (18) | 0.637 (0.45–0.79) | 0.292 (0.14–0.50) [0.14] | 0.056 (0.00–0.18) | 1.000 (1.00–1.00) | 1.000 (0.00–1.00) | 0.333 (0.00–0.58) | 0.640 |
| Remidio | 1074 (150) | 0.605 (0.56–0.65) | 0.189 (0.15–0.24) [0.14] | 0.020 (0.00–0.05) | 0.986 (0.98–0.99) | 0.188 (0.00–0.40) | 0.107 (0.06–0.19) | 0.606 |

| Setting | TP | FP | FN (missed) | TN | NPV | PPV here → PPV in screening |
|---|---|---|---|---|---|---|
| All cameras | 4 | 13 | 183 | 1138 | 0.861 | 0.24 at 14% → 0.06 at 3.5% |
| Bosch | 0 | 0 | 19 | 119 | 0.862 | 0.00 at 14% → nan at 3.5% |
| Forus | 1 | 0 | 17 | 108 | 0.864 | 1.00 at 14% → 1.00 at 3.5% |
| Remidio | 3 | 13 | 147 | 911 | 0.861 | 0.19 at 14% → 0.05 at 3.5% |

PPV in screening applies this sensitivity and specificity to a population where 3.5% have glaucoma (Tham et al. 2014): the same model raises far more false alarms when the disease is rarer.

### By expert agreement

| Setting | n (glaucoma) | AUROC | AUPRC [guessing] | Sensitivity | Specificity | Precision (PPV) | Sens. at 90% spec. | Paper AUROC |
|---|---|---|---|---|---|---|---|---|
| All cameras, experts 5-0 | 782 (47) | 0.657 (0.57–0.74) | 0.126 (0.08–0.21) [0.06] | 0.021 (0.00–0.07) | 0.990 (0.98–1.00) | 0.125 (0.00–0.42) | 0.213 (0.10–0.35) | 0.660 |
| All cameras, experts 4-1 | 354 (66) | 0.562 (0.48–0.63) | 0.238 (0.17–0.33) [0.19] | 0.045 (0.00–0.10) | 0.990 (0.98–1.00) | 0.500 (0.00–1.00) | 0.091 (0.02–0.19) | 0.569 |
| All cameras, experts 3-2 | 202 (74) | 0.544 (0.47–0.63) | 0.385 (0.31–0.50) [0.37] | 0.000 (0.00–0.00) | 0.977 (0.95–1.00) | 0.000 (0.00–0.00) | 0.041 (0.00–0.17) | 0.541 |
| Remidio, experts 5-0 | 631 (39) | 0.664 (0.57–0.76) | 0.139 (0.08–0.24) [0.06] | 0.026 (0.00–0.09) | 0.988 (0.98–1.00) | 0.125 (0.00–0.40) | 0.256 (0.11–0.41) | 0.676 |
| Remidio, experts 4-1 | 284 (54) | 0.549 (0.46–0.63) | 0.231 (0.16–0.32) [0.19] | 0.037 (0.00–0.10) | 0.987 (0.97–1.00) | 0.400 (0.00–1.00) | 0.056 (0.00–0.16) | 0.546 |
| Remidio, experts 3-2 | 159 (57) | 0.555 (0.46–0.65) | 0.385 (0.30–0.51) [0.36] | 0.000 (0.00–0.00) | 0.971 (0.94–1.00) | 0.000 (0.00–0.00) | 0.053 (0.00–0.20) | 0.551 |

### Errors: safe (experts split) vs dangerous (experts unanimous)

| Camera | Images | Wrong | Dangerous | Safe | Wrong, by glaucoma votes (0-5) |
|---|---|---|---|---|---|
| All cameras | 1338 | 196 | 53 | 143 | {0: 7, 1: 3, 2: 3, 3: 74, 4: 63, 5: 46} |
| Bosch | 138 | 19 | 3 | 16 | {3: 9, 4: 7, 5: 3} |
| Forus | 126 | 17 | 5 | 12 | {3: 8, 4: 4, 5: 5} |
| Remidio | 1074 | 160 | 45 | 115 | {0: 7, 1: 3, 2: 3, 3: 57, 4: 52, 5: 38} |
