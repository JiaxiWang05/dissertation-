# Fairness (Fairlearn MetricFrame)

### Chákṣu by camera (trained on PAPILA, threshold fixed from PAPILA)

| Group | Images (glaucoma) | AUROC | Sensitivity | Specificity | Precision |
|---|---|---|---|---|---|
| Bosch | 138 (19) | 0.70 (0.59–0.79) | 0.53 (0.29–0.75) | 0.64 (0.54–0.73) | 0.19 (0.08–0.31) |
| Forus | 126 (18) | 0.72 (0.56–0.86) | 0.67 (0.43–0.88) | 0.56 (0.46–0.65) | 0.20 (0.11–0.31) |
| Remidio | 1074 (150) | 0.73 (0.69–0.77) | 0.90 (0.85–0.94) | 0.33 (0.30–0.36) | 0.18 (0.15–0.20) |

Fairlearn difference (largest minus smallest): sensitivity 0.37; specificity 0.31; false positive rate 0.31

### PAPILA by sex (out-of-fold)

| Group | Images (glaucoma) | AUROC | Sensitivity | Specificity | Precision |
|---|---|---|---|---|---|
| sex code 0 | 146 (35) | 0.81 (0.72–0.89) | 0.89 (0.78–0.97) | 0.62 (0.51–0.73) | 0.42 (0.27–0.56) |
| sex code 1 | 274 (52) | 0.89 (0.83–0.94) | 0.96 (0.90–1.00) | 0.57 (0.49–0.65) | 0.34 (0.24–0.45) |

Fairlearn difference (largest minus smallest): sensitivity 0.08; specificity 0.05; false positive rate 0.05

### PAPILA by age (out-of-fold)

| Group | Images (glaucoma) | AUROC | Sensitivity | Specificity | Precision |
|---|---|---|---|---|---|
| age < 63 | 206 (23) | 0.83 (0.74–0.91) | 0.83 (0.68–0.96) | 0.72 (0.63–0.80) | 0.27 (0.14–0.41) |
| age ≥ 63 | 214 (64) | 0.83 (0.76–0.90) | 0.97 (0.92–1.00) | 0.43 (0.34–0.53) | 0.42 (0.32–0.53) |

Fairlearn difference (largest minus smallest): sensitivity 0.14; specificity 0.28; false positive rate 0.28

