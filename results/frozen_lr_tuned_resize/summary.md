# RETFound-Green: frozen backbone + logistic regression

Binary glaucoma vs healthy. PAPILA suspects: exclude. Folds: official. Head: C tuned by inner patient-grouped CV, balanced class weights. Threshold: Youden J on inner cross-validated predictions of the training data (never the test fold); for Chákṣu, the threshold from all of PAPILA is frozen. 95% CIs: 2000x cluster bootstrap (patients / eyes). AUPRC [guessing] = the score of random guessing, i.e. the share of glaucoma images. Sens. at 90% spec. is read from the ROC curve. 'Paper AUROC' = median of 100 image-level bootstrap resamples, the paper's metric.

## PAPILA, 5-fold patient-level CV (out-of-fold, pooled)

| Setting | n (glaucoma) | AUROC | AUPRC [guessing] | Sensitivity | Specificity | Precision (PPV) | Sens. at 90% spec. | Paper AUROC |
|---|---|---|---|---|---|---|---|---|
| image | 420 (87) | 0.840 (0.77–0.90) | 0.648 (0.51–0.76) [0.21] | 0.770 (0.66–0.87) | 0.808 (0.75–0.86) | 0.511 (0.40–0.62) | 0.609 (0.44–0.74) | 0.843 |
| clinical (age, sex) | 420 (87) | 0.733 (0.65–0.81) | 0.380 (0.27–0.53) [0.21] | 0.759 (0.63–0.88) | 0.646 (0.57–0.72) | 0.359 (0.26–0.45) | 0.230 (0.11–0.37) | 0.733 |
| image + clinical | 420 (87) | 0.853 (0.79–0.91) | 0.662 (0.53–0.78) [0.21] | 0.782 (0.68–0.88) | 0.814 (0.76–0.86) | 0.523 (0.41–0.63) | 0.621 (0.46–0.75) | 0.856 |

| Setting | TP | FP | FN (missed) | TN | NPV | PPV here → PPV in screening |
|---|---|---|---|---|---|---|
| image | 67 | 64 | 20 | 269 | 0.931 | 0.51 at 21% → 0.13 at 3.5% |
| clinical (age, sex) | 66 | 118 | 21 | 215 | 0.911 | 0.36 at 21% → 0.07 at 3.5% |
| image + clinical | 68 | 62 | 19 | 271 | 0.934 | 0.52 at 21% → 0.13 at 3.5% |

AUROC inside each fold (mean ± SD of 5): image 0.856 ± 0.045; clinical (age, sex) 0.766 ± 0.037; image + clinical 0.860 ± 0.041

Paired bootstrap of the AUROC difference (scores ranked within each fold):

- image minus clinical (age, sex): +0.086 (95% CI +0.018 to +0.158): A better
- image + clinical minus image: +0.004 (95% CI -0.001 to +0.009): no clear winner (0 in CI)

## Chákṣu external test (trained on all PAPILA, C = 0.00316, threshold 0.514; no retraining)

| Setting | n (glaucoma) | AUROC | AUPRC [guessing] | Sensitivity | Specificity | Precision (PPV) | Sens. at 90% spec. | Paper AUROC |
|---|---|---|---|---|---|---|---|---|
| All cameras | 1338 (187) | 0.698 (0.66–0.74) | 0.280 (0.23–0.34) [0.14] | 0.733 (0.67–0.80) | 0.575 (0.55–0.60) | 0.219 (0.19–0.25) | 0.326 (0.23–0.39) | 0.695 |
| Bosch | 138 (19) | 0.723 (0.61–0.83) | 0.250 (0.15–0.43) [0.14] | 0.421 (0.21–0.64) | 0.815 (0.74–0.88) | 0.267 (0.12–0.44) | 0.158 (0.00–0.47) | 0.729 |
| Forus | 126 (18) | 0.733 (0.57–0.87) | 0.424 (0.23–0.67) [0.14] | 0.556 (0.31–0.79) | 0.843 (0.77–0.91) | 0.370 (0.19–0.57) | 0.389 (0.18–0.70) | 0.739 |
| Remidio | 1074 (150) | 0.705 (0.66–0.75) | 0.286 (0.23–0.36) [0.14] | 0.793 (0.73–0.86) | 0.513 (0.48–0.55) | 0.209 (0.18–0.24) | 0.293 (0.21–0.40) | 0.702 |

| Setting | TP | FP | FN (missed) | TN | NPV | PPV here → PPV in screening |
|---|---|---|---|---|---|---|
| All cameras | 137 | 489 | 50 | 662 | 0.930 | 0.22 at 14% → 0.06 at 3.5% |
| Bosch | 8 | 22 | 11 | 97 | 0.898 | 0.27 at 14% → 0.08 at 3.5% |
| Forus | 10 | 17 | 8 | 91 | 0.919 | 0.37 at 14% → 0.11 at 3.5% |
| Remidio | 119 | 450 | 31 | 474 | 0.939 | 0.21 at 14% → 0.06 at 3.5% |

PPV in screening applies this sensitivity and specificity to a population where 3.5% have glaucoma (Tham et al. 2014): the same model raises far more false alarms when the disease is rarer.

### By expert agreement

| Setting | n (glaucoma) | AUROC | AUPRC [guessing] | Sensitivity | Specificity | Precision (PPV) | Sens. at 90% spec. | Paper AUROC |
|---|---|---|---|---|---|---|---|---|
| All cameras, experts 5-0 | 782 (47) | 0.823 (0.76–0.88) | 0.272 (0.18–0.42) [0.06] | 0.915 (0.82–0.98) | 0.633 (0.60–0.67) | 0.137 (0.10–0.18) | 0.532 (0.39–0.69) | 0.823 |
| All cameras, experts 4-1 | 354 (66) | 0.663 (0.59–0.73) | 0.321 (0.23–0.43) [0.19] | 0.682 (0.57–0.79) | 0.514 (0.45–0.57) | 0.243 (0.19–0.30) | 0.227 (0.12–0.40) | 0.659 |
| All cameras, experts 3-2 | 202 (74) | 0.537 (0.45–0.62) | 0.397 (0.31–0.51) [0.37] | 0.662 (0.55–0.77) | 0.383 (0.30–0.47) | 0.383 (0.30–0.47) | 0.108 (0.03–0.27) | 0.538 |
| Remidio, experts 5-0 | 631 (39) | 0.844 (0.79–0.90) | 0.301 (0.19–0.46) [0.06] | 0.949 (0.87–1.00) | 0.578 (0.54–0.62) | 0.129 (0.09–0.17) | 0.564 (0.39–0.74) | 0.842 |
| Remidio, experts 4-1 | 284 (54) | 0.673 (0.59–0.75) | 0.324 (0.23–0.45) [0.19] | 0.759 (0.64–0.87) | 0.448 (0.38–0.51) | 0.244 (0.18–0.31) | 0.241 (0.08–0.39) | 0.669 |
| Remidio, experts 3-2 | 159 (57) | 0.517 (0.42–0.61) | 0.387 (0.29–0.52) [0.36] | 0.719 (0.60–0.83) | 0.284 (0.20–0.38) | 0.360 (0.27–0.45) | 0.105 (0.02–0.30) | 0.524 |

### Errors: safe (experts split) vs dangerous (experts unanimous)

| Camera | Images | Wrong | Dangerous | Safe | Wrong, by glaucoma votes (0-5) |
|---|---|---|---|---|---|
| All cameras | 1338 | 539 | 274 | 265 | {0: 270, 1: 140, 2: 79, 3: 25, 4: 21, 5: 4} |
| Bosch | 138 | 33 | 11 | 22 | {0: 10, 1: 9, 2: 3, 3: 5, 4: 5, 5: 1} |
| Forus | 126 | 25 | 11 | 14 | {0: 10, 1: 4, 2: 3, 3: 4, 4: 3, 5: 1} |
| Remidio | 1074 | 481 | 252 | 229 | {0: 250, 1: 127, 2: 73, 3: 16, 4: 13, 5: 2} |
