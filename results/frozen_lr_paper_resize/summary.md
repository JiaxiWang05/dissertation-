# RETFound-Green: frozen backbone + logistic regression

Binary glaucoma vs healthy. PAPILA suspects: exclude. Folds: official. Head: sklearn defaults as in the RETFound-Green paper (C = 1, no class weights). Threshold: Youden J on inner cross-validated predictions of the training data (never the test fold); for Chákṣu, the threshold from all of PAPILA is frozen. 95% CIs: 2000x cluster bootstrap (patients / eyes). AUPRC [guessing] = the score of random guessing, i.e. the share of glaucoma images. Sens. at 90% spec. is read from the ROC curve. 'Paper AUROC' = median of 100 image-level bootstrap resamples, the paper's metric.

## PAPILA, 5-fold patient-level CV (out-of-fold, pooled)

| Setting | n (glaucoma) | AUROC | AUPRC [guessing] | Sensitivity | Specificity | Precision (PPV) | Sens. at 90% spec. | Paper AUROC |
|---|---|---|---|---|---|---|---|---|
| image | 420 (87) | 0.794 (0.72–0.86) | 0.601 (0.47–0.72) [0.21] | 0.655 (0.54–0.77) | 0.808 (0.76–0.85) | 0.471 (0.36–0.58) | 0.563 (0.43–0.71) | 0.793 |
| clinical (age, sex) | 420 (87) | 0.761 (0.69–0.83) | 0.434 (0.31–0.59) [0.21] | 0.782 (0.65–0.89) | 0.634 (0.56–0.70) | 0.358 (0.26–0.45) | 0.345 (0.19–0.53) | 0.760 |
| image + clinical | 420 (87) | 0.804 (0.73–0.87) | 0.616 (0.49–0.74) [0.21] | 0.678 (0.57–0.78) | 0.820 (0.77–0.87) | 0.496 (0.39–0.60) | 0.598 (0.46–0.73) | 0.806 |

| Setting | TP | FP | FN (missed) | TN | NPV | PPV here → PPV in screening |
|---|---|---|---|---|---|---|
| image | 57 | 64 | 30 | 269 | 0.900 | 0.47 at 21% → 0.11 at 3.5% |
| clinical (age, sex) | 68 | 122 | 19 | 211 | 0.917 | 0.36 at 21% → 0.07 at 3.5% |
| image + clinical | 59 | 60 | 28 | 273 | 0.907 | 0.50 at 21% → 0.12 at 3.5% |

AUROC inside each fold (mean ± SD of 5): image 0.803 ± 0.042; clinical (age, sex) 0.772 ± 0.045; image + clinical 0.809 ± 0.042

Paired bootstrap of the AUROC difference (scores ranked within each fold):

- image minus clinical (age, sex): +0.028 (95% CI -0.057 to +0.110): no clear winner (0 in CI)
- image + clinical minus image: +0.006 (95% CI -0.005 to +0.017): no clear winner (0 in CI)

## Chákṣu external test (trained on all PAPILA, C = 1, threshold 0.335; no retraining)

| Setting | n (glaucoma) | AUROC | AUPRC [guessing] | Sensitivity | Specificity | Precision (PPV) | Sens. at 90% spec. | Paper AUROC |
|---|---|---|---|---|---|---|---|---|
| All cameras | 1338 (187) | 0.565 (0.52–0.61) | 0.180 (0.15–0.23) [0.14] | 0.011 (0.00–0.03) | 0.998 (1.00–1.00) | 0.500 (0.00–1.00) | 0.144 (0.09–0.20) | 0.565 |
| Bosch | 138 (19) | 0.584 (0.45–0.71) | 0.180 (0.11–0.34) [0.14] | 0.000 (0.00–0.00) | 1.000 (1.00–1.00) | 0.000 (0.00–0.00) | 0.158 (0.00–0.33) | 0.587 |
| Forus | 126 (18) | 0.706 (0.56–0.83) | 0.295 (0.16–0.53) [0.14] | 0.111 (0.00–0.27) | 0.981 (0.95–1.00) | 0.500 (0.00–1.00) | 0.222 (0.05–0.45) | 0.712 |
| Remidio | 1074 (150) | 0.574 (0.52–0.63) | 0.180 (0.15–0.23) [0.14] | 0.000 (0.00–0.00) | 1.000 (1.00–1.00) | 0.000 (0.00–0.00) | 0.140 (0.09–0.23) | 0.576 |

| Setting | TP | FP | FN (missed) | TN | NPV | PPV here → PPV in screening |
|---|---|---|---|---|---|---|
| All cameras | 2 | 2 | 185 | 1149 | 0.861 | 0.50 at 14% → 0.18 at 3.5% |
| Bosch | 0 | 0 | 19 | 119 | 0.862 | 0.00 at 14% → nan at 3.5% |
| Forus | 2 | 2 | 16 | 106 | 0.869 | 0.50 at 14% → 0.18 at 3.5% |
| Remidio | 0 | 0 | 150 | 924 | 0.860 | 0.00 at 14% → nan at 3.5% |

PPV in screening applies this sensitivity and specificity to a population where 3.5% have glaucoma (Tham et al. 2014): the same model raises far more false alarms when the disease is rarer.

### By expert agreement

| Setting | n (glaucoma) | AUROC | AUPRC [guessing] | Sensitivity | Specificity | Precision (PPV) | Sens. at 90% spec. | Paper AUROC |
|---|---|---|---|---|---|---|---|---|
| All cameras, experts 5-0 | 782 (47) | 0.678 (0.59–0.75) | 0.110 (0.07–0.17) [0.06] | 0.000 (0.00–0.00) | 0.999 (1.00–1.00) | 0.000 (0.00–0.00) | 0.191 (0.09–0.36) | 0.681 |
| All cameras, experts 4-1 | 354 (66) | 0.545 (0.47–0.62) | 0.227 (0.17–0.32) [0.19] | 0.015 (0.00–0.05) | 0.997 (0.99–1.00) | 0.500 (0.00–1.00) | 0.136 (0.05–0.22) | 0.547 |
| All cameras, experts 3-2 | 202 (74) | 0.459 (0.38–0.54) | 0.393 (0.31–0.50) [0.37] | 0.014 (0.00–0.04) | 1.000 (1.00–1.00) | 1.000 (0.00–1.00) | 0.122 (0.04–0.22) | 0.473 |
| Remidio, experts 5-0 | 631 (39) | 0.704 (0.61–0.79) | 0.144 (0.09–0.25) [0.06] | 0.000 (0.00–0.00) | 1.000 (1.00–1.00) | 0.000 (0.00–0.00) | 0.308 (0.11–0.49) | 0.714 |
| Remidio, experts 4-1 | 284 (54) | 0.552 (0.47–0.64) | 0.235 (0.16–0.33) [0.19] | 0.000 (0.00–0.00) | 1.000 (1.00–1.00) | 0.000 (0.00–0.00) | 0.093 (0.02–0.21) | 0.555 |
| Remidio, experts 3-2 | 159 (57) | 0.466 (0.37–0.56) | 0.339 (0.26–0.45) [0.36] | 0.000 (0.00–0.00) | 1.000 (1.00–1.00) | 0.000 (0.00–0.00) | 0.088 (0.02–0.17) | 0.466 |

### Errors: safe (experts split) vs dangerous (experts unanimous)

| Camera | Images | Wrong | Dangerous | Safe | Wrong, by glaucoma votes (0-5) |
|---|---|---|---|---|---|
| All cameras | 1338 | 187 | 48 | 139 | {0: 1, 1: 1, 3: 73, 4: 65, 5: 47} |
| Bosch | 138 | 19 | 3 | 16 | {3: 9, 4: 7, 5: 3} |
| Forus | 126 | 18 | 6 | 12 | {0: 1, 1: 1, 3: 7, 4: 4, 5: 5} |
| Remidio | 1074 | 150 | 39 | 111 | {3: 57, 4: 54, 5: 39} |
