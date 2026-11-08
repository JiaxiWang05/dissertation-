# RETFound-Green: frozen backbone + logistic regression

Binary glaucoma vs healthy. PAPILA suspects: exclude. Folds: official. Head: C tuned by inner patient-grouped CV, balanced class weights. Threshold: 90% sensitivity on inner cross-validated predictions of the training data (never the test fold); for Chákṣu, the threshold from all of PAPILA is frozen. 95% CIs: 2000x cluster bootstrap (patients / eyes). AUPRC [guessing] = the score of random guessing, i.e. the share of glaucoma images. Sens. at 90% spec. is read from the ROC curve. 'Paper AUROC' = median of 100 image-level bootstrap resamples, the paper's metric.

## PAPILA, 5-fold patient-level CV (out-of-fold, pooled)

| Setting | n (glaucoma) | AUROC | AUPRC [guessing] | Sensitivity | Specificity | Precision (PPV) | Sens. at 90% spec. | Paper AUROC |
|---|---|---|---|---|---|---|---|---|
| image | 420 (87) | 0.855 (0.80–0.90) | 0.606 (0.46–0.73) [0.21] | 0.931 (0.88–0.98) | 0.589 (0.52–0.65) | 0.372 (0.29–0.46) | 0.575 (0.40–0.70) | 0.857 |
| clinical (age, sex) | 420 (87) | 0.733 (0.65–0.81) | 0.380 (0.27–0.53) [0.21] | 0.897 (0.80–0.98) | 0.441 (0.37–0.52) | 0.295 (0.22–0.37) | 0.230 (0.11–0.37) | 0.733 |
| image + clinical | 420 (87) | 0.860 (0.81–0.90) | 0.613 (0.47–0.74) [0.21] | 0.931 (0.88–0.98) | 0.592 (0.53–0.66) | 0.373 (0.29–0.46) | 0.598 (0.41–0.73) | 0.861 |

| Setting | TP | FP | FN (missed) | TN | NPV | PPV here → PPV in screening |
|---|---|---|---|---|---|---|
| image | 81 | 137 | 6 | 196 | 0.970 | 0.37 at 21% → 0.08 at 3.5% |
| clinical (age, sex) | 78 | 186 | 9 | 147 | 0.942 | 0.30 at 21% → 0.06 at 3.5% |
| image + clinical | 81 | 136 | 6 | 197 | 0.970 | 0.37 at 21% → 0.08 at 3.5% |

AUROC inside each fold (mean ± SD of 5): image 0.859 ± 0.034; clinical (age, sex) 0.766 ± 0.037; image + clinical 0.863 ± 0.032

Paired bootstrap of the AUROC difference (scores ranked within each fold):

- image minus clinical (age, sex): +0.091 (95% CI +0.025 to +0.163): A better
- image + clinical minus image: +0.004 (95% CI +0.001 to +0.008): A better

## Chákṣu external test (trained on all PAPILA, C = 0.01, threshold 0.279; no retraining)

| Setting | n (glaucoma) | AUROC | AUPRC [guessing] | Sensitivity | Specificity | Precision (PPV) | Sens. at 90% spec. | Paper AUROC |
|---|---|---|---|---|---|---|---|---|
| All cameras | 1338 (187) | 0.716 (0.68–0.76) | 0.329 (0.27–0.40) [0.14] | 0.840 (0.78–0.89) | 0.380 (0.35–0.41) | 0.180 (0.15–0.21) | 0.342 (0.27–0.43) | 0.712 |
| Bosch | 138 (19) | 0.698 (0.59–0.79) | 0.215 (0.13–0.38) [0.14] | 0.526 (0.29–0.75) | 0.639 (0.54–0.73) | 0.189 (0.08–0.31) | 0.105 (0.00–0.37) | 0.696 |
| Forus | 126 (18) | 0.717 (0.56–0.86) | 0.479 (0.25–0.71) [0.14] | 0.667 (0.43–0.88) | 0.556 (0.46–0.65) | 0.200 (0.11–0.31) | 0.444 (0.21–0.71) | 0.712 |
| Remidio | 1074 (150) | 0.734 (0.69–0.77) | 0.351 (0.28–0.43) [0.14] | 0.900 (0.85–0.94) | 0.326 (0.30–0.36) | 0.178 (0.15–0.20) | 0.367 (0.28–0.44) | 0.732 |

| Setting | TP | FP | FN (missed) | TN | NPV | PPV here → PPV in screening |
|---|---|---|---|---|---|---|
| All cameras | 157 | 714 | 30 | 437 | 0.936 | 0.18 at 14% → 0.05 at 3.5% |
| Bosch | 10 | 43 | 9 | 76 | 0.894 | 0.19 at 14% → 0.05 at 3.5% |
| Forus | 12 | 48 | 6 | 60 | 0.909 | 0.20 at 14% → 0.05 at 3.5% |
| Remidio | 135 | 623 | 15 | 301 | 0.953 | 0.18 at 14% → 0.05 at 3.5% |

PPV in screening applies this sensitivity and specificity to a population where 3.5% have glaucoma (Tham et al. 2014): the same model raises far more false alarms when the disease is rarer.

### By expert agreement

| Setting | n (glaucoma) | AUROC | AUPRC [guessing] | Sensitivity | Specificity | Precision (PPV) | Sens. at 90% spec. | Paper AUROC |
|---|---|---|---|---|---|---|---|---|
| All cameras, experts 5-0 | 782 (47) | 0.830 (0.78–0.88) | 0.293 (0.19–0.44) [0.06] | 0.957 (0.89–1.00) | 0.422 (0.39–0.46) | 0.096 (0.07–0.12) | 0.511 (0.35–0.65) | 0.832 |
| All cameras, experts 4-1 | 354 (66) | 0.689 (0.61–0.76) | 0.401 (0.30–0.52) [0.19] | 0.818 (0.72–0.91) | 0.323 (0.27–0.38) | 0.217 (0.17–0.27) | 0.348 (0.21–0.49) | 0.683 |
| All cameras, experts 3-2 | 202 (74) | 0.566 (0.48–0.65) | 0.450 (0.35–0.57) [0.37] | 0.784 (0.69–0.88) | 0.266 (0.19–0.34) | 0.382 (0.31–0.46) | 0.135 (0.06–0.31) | 0.568 |
| Remidio, experts 5-0 | 631 (39) | 0.850 (0.80–0.90) | 0.297 (0.19–0.45) [0.06] | 1.000 (1.00–1.00) | 0.372 (0.33–0.41) | 0.095 (0.07–0.13) | 0.513 (0.37–0.71) | 0.856 |
| Remidio, experts 4-1 | 284 (54) | 0.722 (0.64–0.80) | 0.451 (0.33–0.59) [0.19] | 0.889 (0.80–0.96) | 0.270 (0.21–0.33) | 0.222 (0.17–0.28) | 0.370 (0.24–0.54) | 0.722 |
| Remidio, experts 3-2 | 159 (57) | 0.560 (0.47–0.66) | 0.460 (0.35–0.58) [0.36] | 0.842 (0.75–0.93) | 0.186 (0.11–0.26) | 0.366 (0.28–0.45) | 0.158 (0.05–0.34) | 0.567 |

### Errors: safe (experts split) vs dangerous (experts unanimous)

| Camera | Images | Wrong | Dangerous | Safe | Wrong, by glaucoma votes (0-5) |
|---|---|---|---|---|---|
| All cameras | 1338 | 744 | 427 | 317 | {0: 425, 1: 195, 2: 94, 3: 16, 4: 12, 5: 2} |
| Bosch | 138 | 52 | 25 | 27 | {0: 24, 1: 16, 2: 3, 3: 5, 4: 3, 5: 1} |
| Forus | 126 | 54 | 30 | 24 | {0: 29, 1: 11, 2: 8, 3: 2, 4: 3, 5: 1} |
| Remidio | 1074 | 638 | 372 | 266 | {0: 372, 1: 168, 2: 83, 3: 9, 4: 6} |
