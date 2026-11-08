# RETFound-Green: frozen backbone + logistic regression

Binary glaucoma vs healthy. PAPILA suspects: exclude. Folds: official. Head: C tuned by inner patient-grouped CV, balanced class weights. Threshold: Youden J on inner cross-validated predictions of the training data (never the test fold); for Chákṣu, the threshold from all of PAPILA is frozen. 95% CIs: 2000x cluster bootstrap (patients / eyes). AUPRC [guessing] = the score of random guessing, i.e. the share of glaucoma images. Sens. at 90% spec. is read from the ROC curve. 'Paper AUROC' = median of 100 image-level bootstrap resamples, the paper's metric.

## PAPILA, 5-fold patient-level CV (out-of-fold, pooled)

| Setting | n (glaucoma) | AUROC | AUPRC [guessing] | Sensitivity | Specificity | Precision (PPV) | Sens. at 90% spec. | Paper AUROC |
|---|---|---|---|---|---|---|---|---|
| image | 420 (87) | 0.851 (0.79–0.90) | 0.625 (0.48–0.75) [0.21] | 0.747 (0.63–0.85) | 0.727 (0.67–0.78) | 0.417 (0.31–0.51) | 0.609 (0.46–0.75) | 0.851 |
| clinical (age, sex) | 420 (87) | 0.733 (0.65–0.81) | 0.380 (0.27–0.53) [0.21] | 0.759 (0.63–0.88) | 0.646 (0.57–0.72) | 0.359 (0.26–0.45) | 0.230 (0.11–0.37) | 0.733 |
| image + clinical | 420 (87) | 0.853 (0.79–0.90) | 0.630 (0.49–0.75) [0.21] | 0.816 (0.71–0.90) | 0.745 (0.68–0.80) | 0.455 (0.36–0.55) | 0.632 (0.47–0.75) | 0.854 |

| Setting | TP | FP | FN (missed) | TN | NPV | PPV here → PPV in screening |
|---|---|---|---|---|---|---|
| image | 65 | 91 | 22 | 242 | 0.917 | 0.42 at 21% → 0.09 at 3.5% |
| clinical (age, sex) | 66 | 118 | 21 | 215 | 0.911 | 0.36 at 21% → 0.07 at 3.5% |
| image + clinical | 71 | 85 | 16 | 248 | 0.939 | 0.46 at 21% → 0.10 at 3.5% |

AUROC inside each fold (mean ± SD of 5): image 0.850 ± 0.042; clinical (age, sex) 0.766 ± 0.037; image + clinical 0.856 ± 0.040

Paired bootstrap of the AUROC difference (scores ranked within each fold):

- image minus clinical (age, sex): +0.081 (95% CI +0.014 to +0.152): A better
- image + clinical minus image: +0.006 (95% CI -0.001 to +0.013): no clear winner (0 in CI)

## Chákṣu external test (trained on all PAPILA, C = 0.00316, threshold 0.474; no retraining)

| Setting | n (glaucoma) | AUROC | AUPRC [guessing] | Sensitivity | Specificity | Precision (PPV) | Sens. at 90% spec. | Paper AUROC |
|---|---|---|---|---|---|---|---|---|
| All cameras | 1338 (187) | 0.691 (0.65–0.73) | 0.281 (0.23–0.34) [0.14] | 0.529 (0.46–0.60) | 0.778 (0.75–0.80) | 0.279 (0.23–0.32) | 0.262 (0.19–0.33) | 0.688 |
| Bosch | 138 (19) | 0.560 (0.44–0.69) | 0.170 (0.10–0.32) [0.14] | 0.158 (0.00–0.35) | 0.882 (0.81–0.94) | 0.176 (0.00–0.36) | 0.158 (0.00–0.35) | 0.565 |
| Forus | 126 (18) | 0.687 (0.52–0.84) | 0.406 (0.20–0.64) [0.14] | 0.500 (0.25–0.74) | 0.880 (0.82–0.94) | 0.409 (0.19–0.62) | 0.389 (0.12–0.73) | 0.686 |
| Remidio | 1074 (150) | 0.711 (0.67–0.75) | 0.288 (0.23–0.35) [0.14] | 0.580 (0.50–0.66) | 0.752 (0.72–0.78) | 0.275 (0.23–0.32) | 0.267 (0.18–0.34) | 0.711 |

| Setting | TP | FP | FN (missed) | TN | NPV | PPV here → PPV in screening |
|---|---|---|---|---|---|---|
| All cameras | 99 | 256 | 88 | 895 | 0.910 | 0.28 at 14% → 0.08 at 3.5% |
| Bosch | 3 | 14 | 16 | 105 | 0.868 | 0.18 at 14% → 0.05 at 3.5% |
| Forus | 9 | 13 | 9 | 95 | 0.913 | 0.41 at 14% → 0.13 at 3.5% |
| Remidio | 87 | 229 | 63 | 695 | 0.917 | 0.28 at 14% → 0.08 at 3.5% |

PPV in screening applies this sensitivity and specificity to a population where 3.5% have glaucoma (Tham et al. 2014): the same model raises far more false alarms when the disease is rarer.

### By expert agreement

| Setting | n (glaucoma) | AUROC | AUPRC [guessing] | Sensitivity | Specificity | Precision (PPV) | Sens. at 90% spec. | Paper AUROC |
|---|---|---|---|---|---|---|---|---|
| All cameras, experts 5-0 | 782 (47) | 0.796 (0.73–0.86) | 0.237 (0.15–0.38) [0.06] | 0.681 (0.55–0.82) | 0.815 (0.79–0.84) | 0.190 (0.14–0.25) | 0.404 (0.23–0.56) | 0.797 |
| All cameras, experts 4-1 | 354 (66) | 0.673 (0.60–0.74) | 0.366 (0.27–0.47) [0.19] | 0.561 (0.44–0.68) | 0.719 (0.66–0.77) | 0.314 (0.23–0.40) | 0.273 (0.14–0.41) | 0.677 |
| All cameras, experts 3-2 | 202 (74) | 0.504 (0.42–0.59) | 0.370 (0.30–0.48) [0.37] | 0.405 (0.30–0.52) | 0.695 (0.61–0.77) | 0.435 (0.32–0.55) | 0.054 (0.01–0.21) | 0.511 |
| Remidio, experts 5-0 | 631 (39) | 0.823 (0.77–0.88) | 0.252 (0.15–0.41) [0.06] | 0.744 (0.60–0.88) | 0.799 (0.77–0.83) | 0.196 (0.13–0.26) | 0.359 (0.22–0.57) | 0.823 |
| Remidio, experts 4-1 | 284 (54) | 0.681 (0.60–0.76) | 0.368 (0.26–0.50) [0.19] | 0.593 (0.46–0.72) | 0.665 (0.60–0.72) | 0.294 (0.21–0.38) | 0.278 (0.11–0.43) | 0.678 |
| Remidio, experts 3-2 | 159 (57) | 0.518 (0.42–0.61) | 0.369 (0.29–0.48) [0.36] | 0.456 (0.33–0.58) | 0.676 (0.58–0.77) | 0.441 (0.31–0.57) | 0.053 (0.00–0.20) | 0.528 |

### Errors: safe (experts split) vs dangerous (experts unanimous)

| Camera | Images | Wrong | Dangerous | Safe | Wrong, by glaucoma votes (0-5) |
|---|---|---|---|---|---|
| All cameras | 1338 | 344 | 151 | 193 | {0: 136, 1: 81, 2: 39, 3: 44, 4: 29, 5: 15} |
| Bosch | 138 | 30 | 13 | 17 | {0: 10, 1: 2, 2: 2, 3: 9, 4: 4, 5: 3} |
| Forus | 126 | 22 | 9 | 13 | {0: 7, 1: 2, 2: 4, 3: 4, 4: 3, 5: 2} |
| Remidio | 1074 | 292 | 129 | 163 | {0: 119, 1: 77, 2: 33, 3: 31, 4: 22, 5: 10} |
