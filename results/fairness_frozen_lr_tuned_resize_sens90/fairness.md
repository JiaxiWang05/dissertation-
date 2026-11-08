# Fairness (Fairlearn MetricFrame)

### Chákṣu by camera (trained on PAPILA, threshold fixed from PAPILA)

| Group | Images (glaucoma) | AUROC | Sensitivity | Specificity | Precision |
|---|---|---|---|---|---|
| Bosch | 138 (19) | 0.72 (0.61–0.83) | 1.00 (1.00–1.00) | 0.10 (0.05–0.16) | 0.15 (0.09–0.22) |
| Forus | 126 (18) | 0.73 (0.57–0.87) | 0.83 (0.64–1.00) | 0.36 (0.27–0.46) | 0.18 (0.10–0.27) |
| Remidio | 1074 (150) | 0.71 (0.66–0.75) | 0.99 (0.98–1.00) | 0.03 (0.02–0.04) | 0.14 (0.12–0.16) |

Fairlearn difference (largest minus smallest): sensitivity 0.17; specificity 0.33; false positive rate 0.33

### PAPILA by sex (out-of-fold)

| Group | Images (glaucoma) | AUROC | Sensitivity | Specificity | Precision |
|---|---|---|---|---|---|
| sex code 0 | 146 (35) | 0.81 (0.68–0.92) | 0.89 (0.74–1.00) | 0.64 (0.53–0.74) | 0.44 (0.28–0.58) |
| sex code 1 | 274 (52) | 0.86 (0.79–0.93) | 0.88 (0.79–0.96) | 0.59 (0.52–0.68) | 0.34 (0.23–0.44) |

Fairlearn difference (largest minus smallest): sensitivity 0.00; specificity 0.05; false positive rate 0.05

### PAPILA by age (out-of-fold)

| Group | Images (glaucoma) | AUROC | Sensitivity | Specificity | Precision |
|---|---|---|---|---|---|
| age < 63 | 206 (23) | 0.82 (0.70–0.93) | 0.83 (0.68–0.96) | 0.75 (0.67–0.82) | 0.29 (0.15–0.43) |
| age ≥ 63 | 214 (64) | 0.82 (0.72–0.90) | 0.91 (0.82–0.98) | 0.44 (0.34–0.54) | 0.41 (0.31–0.52) |

Fairlearn difference (largest minus smallest): sensitivity 0.08; specificity 0.31; false positive rate 0.31

