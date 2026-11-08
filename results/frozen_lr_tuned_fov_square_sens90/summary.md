# RETFound-Green: frozen backbone + logistic regression

Binary glaucoma vs healthy. PAPILA suspects: exclude. Folds: official. Head: C tuned by inner patient-grouped CV, balanced class weights. Threshold: 90% sensitivity on inner cross-validated predictions of the training data (never the test fold); for Chákṣu, the threshold from all of PAPILA is frozen. 95% CIs: 2000x cluster bootstrap (patients / eyes). AUPRC [guessing] = the score of random guessing, i.e. the share of glaucoma images. Sens. at 90% spec. is read from the ROC curve. 'Paper AUROC' = median of 100 image-level bootstrap resamples, the paper's metric.

## PAPILA, 5-fold patient-level CV (out-of-fold, pooled)

| Setting | n (glaucoma) | AUROC | AUPRC [guessing] | Sensitivity | Specificity | Precision (PPV) | Sens. at 90% spec. | Paper AUROC |
|---|---|---|---|---|---|---|---|---|
| image | 420 (87) | 0.851 (0.79–0.90) | 0.625 (0.48–0.75) [0.21] | 0.874 (0.78–0.95) | 0.586 (0.52–0.65) | 0.355 (0.27–0.44) | 0.609 (0.46–0.75) | 0.851 |
| clinical (age, sex) | 420 (87) | 0.733 (0.65–0.81) | 0.380 (0.27–0.53) [0.21] | 0.897 (0.80–0.98) | 0.441 (0.37–0.52) | 0.295 (0.22–0.37) | 0.230 (0.11–0.37) | 0.733 |
| image + clinical | 420 (87) | 0.853 (0.79–0.90) | 0.630 (0.49–0.75) [0.21] | 0.874 (0.78–0.95) | 0.583 (0.52–0.65) | 0.353 (0.27–0.44) | 0.632 (0.47–0.75) | 0.854 |

| Setting | TP | FP | FN (missed) | TN | NPV | PPV here → PPV in screening |
|---|---|---|---|---|---|---|
| image | 76 | 138 | 11 | 195 | 0.947 | 0.36 at 21% → 0.07 at 3.5% |
| clinical (age, sex) | 78 | 186 | 9 | 147 | 0.942 | 0.30 at 21% → 0.06 at 3.5% |
| image + clinical | 76 | 139 | 11 | 194 | 0.946 | 0.35 at 21% → 0.07 at 3.5% |

AUROC inside each fold (mean ± SD of 5): image 0.850 ± 0.042; clinical (age, sex) 0.766 ± 0.037; image + clinical 0.856 ± 0.040

Paired bootstrap of the AUROC difference (scores ranked within each fold):

- image minus clinical (age, sex): +0.081 (95% CI +0.014 to +0.152): A better
- image + clinical minus image: +0.006 (95% CI -0.001 to +0.013): no clear winner (0 in CI)

## Chákṣu external test (trained on all PAPILA, C = 0.00316, threshold 0.230; no retraining)

| Setting | n (glaucoma) | AUROC | AUPRC [guessing] | Sensitivity | Specificity | Precision (PPV) | Sens. at 90% spec. | Paper AUROC |
|---|---|---|---|---|---|---|---|---|
| All cameras | 1338 (187) | 0.691 (0.65–0.73) | 0.281 (0.23–0.34) [0.14] | 0.872 (0.82–0.92) | 0.291 (0.26–0.32) | 0.166 (0.14–0.19) | 0.262 (0.19–0.33) | 0.688 |
| Bosch | 138 (19) | 0.560 (0.44–0.69) | 0.170 (0.10–0.32) [0.14] | 0.579 (0.37–0.79) | 0.445 (0.35–0.55) | 0.143 (0.07–0.23) | 0.158 (0.00–0.35) | 0.565 |
| Forus | 126 (18) | 0.687 (0.52–0.84) | 0.406 (0.20–0.64) [0.14] | 0.722 (0.50–0.92) | 0.472 (0.37–0.57) | 0.186 (0.10–0.28) | 0.389 (0.12–0.73) | 0.686 |
| Remidio | 1074 (150) | 0.711 (0.67–0.75) | 0.288 (0.23–0.35) [0.14] | 0.927 (0.88–0.97) | 0.250 (0.22–0.28) | 0.167 (0.14–0.19) | 0.267 (0.18–0.34) | 0.711 |

| Setting | TP | FP | FN (missed) | TN | NPV | PPV here → PPV in screening |
|---|---|---|---|---|---|---|
| All cameras | 163 | 816 | 24 | 335 | 0.933 | 0.17 at 14% → 0.04 at 3.5% |
| Bosch | 11 | 66 | 8 | 53 | 0.869 | 0.14 at 14% → 0.04 at 3.5% |
| Forus | 13 | 57 | 5 | 51 | 0.911 | 0.19 at 14% → 0.05 at 3.5% |
| Remidio | 139 | 693 | 11 | 231 | 0.955 | 0.17 at 14% → 0.04 at 3.5% |

PPV in screening applies this sensitivity and specificity to a population where 3.5% have glaucoma (Tham et al. 2014): the same model raises far more false alarms when the disease is rarer.

### By expert agreement

| Setting | n (glaucoma) | AUROC | AUPRC [guessing] | Sensitivity | Specificity | Precision (PPV) | Sens. at 90% spec. | Paper AUROC |
|---|---|---|---|---|---|---|---|---|
| All cameras, experts 5-0 | 782 (47) | 0.796 (0.73–0.86) | 0.237 (0.15–0.38) [0.06] | 0.936 (0.86–1.00) | 0.339 (0.31–0.37) | 0.083 (0.06–0.11) | 0.404 (0.23–0.56) | 0.797 |
| All cameras, experts 4-1 | 354 (66) | 0.673 (0.60–0.74) | 0.366 (0.27–0.47) [0.19] | 0.864 (0.78–0.94) | 0.240 (0.19–0.29) | 0.207 (0.16–0.25) | 0.273 (0.14–0.41) | 0.677 |
| All cameras, experts 3-2 | 202 (74) | 0.504 (0.42–0.59) | 0.370 (0.30–0.48) [0.37] | 0.838 (0.75–0.92) | 0.133 (0.08–0.19) | 0.358 (0.29–0.43) | 0.054 (0.01–0.21) | 0.511 |
| Remidio, experts 5-0 | 631 (39) | 0.823 (0.77–0.88) | 0.252 (0.15–0.41) [0.06] | 1.000 (1.00–1.00) | 0.306 (0.27–0.34) | 0.087 (0.06–0.11) | 0.359 (0.22–0.57) | 0.823 |
| Remidio, experts 4-1 | 284 (54) | 0.681 (0.60–0.76) | 0.368 (0.26–0.50) [0.19] | 0.926 (0.85–0.98) | 0.178 (0.13–0.23) | 0.209 (0.16–0.26) | 0.278 (0.11–0.43) | 0.678 |
| Remidio, experts 3-2 | 159 (57) | 0.518 (0.42–0.61) | 0.369 (0.29–0.48) [0.36] | 0.877 (0.79–0.95) | 0.088 (0.04–0.15) | 0.350 (0.27–0.43) | 0.053 (0.00–0.20) | 0.528 |

### Errors: safe (experts split) vs dangerous (experts unanimous)

| Camera | Images | Wrong | Dangerous | Safe | Wrong, by glaucoma votes (0-5) |
|---|---|---|---|---|---|
| All cameras | 1338 | 840 | 489 | 351 | {0: 486, 1: 219, 2: 111, 3: 12, 4: 9, 5: 3} |
| Bosch | 138 | 74 | 42 | 32 | {0: 40, 1: 18, 2: 8, 3: 3, 4: 3, 5: 2} |
| Forus | 126 | 62 | 36 | 26 | {0: 35, 1: 12, 2: 10, 3: 2, 4: 2, 5: 1} |
| Remidio | 1074 | 704 | 411 | 293 | {0: 411, 1: 189, 2: 93, 3: 7, 4: 4} |
