# Fairness (Fairlearn MetricFrame)

### Chákṣu by camera (trained on PAPILA, threshold fixed from PAPILA)

| Group | Images (glaucoma) | AUROC | Sensitivity | Specificity | Precision |
|---|---|---|---|---|---|
| Bosch | 138 (19) | 0.72 (0.61–0.83) | 0.42 (0.21–0.64) | 0.82 (0.74–0.88) | 0.27 (0.12–0.44) |
| Forus | 126 (18) | 0.73 (0.57–0.87) | 0.56 (0.31–0.79) | 0.84 (0.77–0.91) | 0.37 (0.19–0.57) |
| Remidio | 1074 (150) | 0.71 (0.66–0.75) | 0.79 (0.73–0.86) | 0.51 (0.48–0.55) | 0.21 (0.18–0.24) |

Fairlearn difference (largest minus smallest): sensitivity 0.37; specificity 0.33; false positive rate 0.33

### PAPILA by sex (out-of-fold)

| Group | Images (glaucoma) | AUROC | Sensitivity | Specificity | Precision |
|---|---|---|---|---|---|
| sex code 0 | 146 (35) | 0.81 (0.68–0.92) | 0.69 (0.50–0.86) | 0.82 (0.73–0.90) | 0.55 (0.36–0.71) |
| sex code 1 | 274 (52) | 0.86 (0.79–0.93) | 0.83 (0.69–0.94) | 0.80 (0.74–0.87) | 0.49 (0.36–0.63) |

Fairlearn difference (largest minus smallest): sensitivity 0.14; specificity 0.02; false positive rate 0.02

### PAPILA by age (out-of-fold)

| Group | Images (glaucoma) | AUROC | Sensitivity | Specificity | Precision |
|---|---|---|---|---|---|
| age < 63 | 206 (23) | 0.82 (0.70–0.93) | 0.57 (0.32–0.81) | 0.90 (0.85–0.95) | 0.42 (0.21–0.63) |
| age ≥ 63 | 214 (64) | 0.82 (0.72–0.90) | 0.84 (0.73–0.94) | 0.69 (0.60–0.78) | 0.54 (0.42–0.66) |

Fairlearn difference (largest minus smallest): sensitivity 0.28; specificity 0.21; false positive rate 0.21

