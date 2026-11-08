# Single-number summaries at the chosen threshold (95% CI)

F1 combines sensitivity and precision (ignores true negatives); balanced accuracy and Youden's J combine sensitivity and specificity and do not change with prevalence; MCC uses all four cells and changes with prevalence. Chákṣu uses the threshold frozen from PAPILA.

| Input, threshold | Data | Sensitivity | Specificity | F1 | Balanced accuracy | Youden's J | MCC |
|---|---|---|---|---|---|---|---|
| full image, Youden's J | PAPILA (out-of-fold) | 0.77 (0.66–0.87) | 0.81 (0.75–0.86) | 0.61 (0.51–0.70) | 0.79 (0.73–0.84) | 0.58 (0.45–0.69) | 0.51 (0.39–0.61) |
| full image, Youden's J | Chákṣu, All cameras | 0.73 (0.67–0.80) | 0.58 (0.55–0.60) | 0.34 (0.29–0.38) | 0.65 (0.62–0.69) | 0.31 (0.23–0.38) | 0.21 (0.16–0.26) |
| full image, Youden's J | Chákṣu, Remidio | 0.79 (0.73–0.86) | 0.51 (0.48–0.55) | 0.33 (0.29–0.37) | 0.65 (0.62–0.69) | 0.31 (0.23–0.38) | 0.21 (0.16–0.26) |
| full image, Youden's J | Chákṣu, Bosch | 0.42 (0.21–0.64) | 0.82 (0.74–0.88) | 0.33 (0.15–0.49) | 0.62 (0.50–0.73) | 0.24 (0.01–0.47) | 0.20 (0.01–0.38) |
| full image, Youden's J | Chákṣu, Forus | 0.56 (0.31–0.79) | 0.84 (0.77–0.91) | 0.44 (0.24–0.62) | 0.70 (0.58–0.82) | 0.40 (0.16–0.65) | 0.34 (0.13–0.54) |
| full image, 90% sensitivity | PAPILA (out-of-fold) | 0.89 (0.81–0.95) | 0.61 (0.54–0.67) | 0.52 (0.43–0.60) | 0.75 (0.70–0.79) | 0.49 (0.39–0.58) | 0.40 (0.31–0.48) |
| full image, 90% sensitivity | Chákṣu, All cameras | 0.98 (0.95–1.00) | 0.07 (0.05–0.08) | 0.25 (0.22–0.28) | 0.52 (0.51–0.54) | 0.05 (0.02–0.07) | 0.07 (0.03–0.10) |
| full image, 90% sensitivity | Chákṣu, Remidio | 0.99 (0.98–1.00) | 0.03 (0.02–0.04) | 0.25 (0.22–0.28) | 0.51 (0.50–0.52) | 0.02 (0.00–0.04) | 0.05 (0.01–0.07) |
| full image, 90% sensitivity | Chákṣu, Bosch | 1.00 (1.00–1.00) | 0.10 (0.05–0.16) | 0.26 (0.16–0.36) | 0.55 (0.52–0.58) | 0.10 (0.05–0.16) | 0.12 (0.08–0.17) |
| full image, 90% sensitivity | Chákṣu, Forus | 0.83 (0.64–1.00) | 0.36 (0.27–0.46) | 0.29 (0.18–0.41) | 0.60 (0.49–0.69) | 0.19 (-0.01–0.38) | 0.14 (-0.01–0.28) |
| optic-disc crop, 90% sensitivity | PAPILA (out-of-fold) | 0.93 (0.88–0.98) | 0.59 (0.52–0.65) | 0.53 (0.44–0.61) | 0.76 (0.72–0.80) | 0.52 (0.43–0.60) | 0.42 (0.34–0.50) |
| optic-disc crop, 90% sensitivity | Chákṣu, All cameras | 0.84 (0.78–0.89) | 0.38 (0.35–0.41) | 0.30 (0.26–0.33) | 0.61 (0.58–0.64) | 0.22 (0.15–0.28) | 0.16 (0.11–0.20) |
| optic-disc crop, 90% sensitivity | Chákṣu, Remidio | 0.90 (0.85–0.94) | 0.33 (0.30–0.36) | 0.30 (0.26–0.33) | 0.61 (0.58–0.64) | 0.23 (0.17–0.28) | 0.17 (0.13–0.21) |
| optic-disc crop, 90% sensitivity | Chákṣu, Bosch | 0.53 (0.29–0.75) | 0.64 (0.54–0.73) | 0.28 (0.13–0.42) | 0.58 (0.45–0.71) | 0.16 (-0.10–0.42) | 0.12 (-0.06–0.29) |
| optic-disc crop, 90% sensitivity | Chákṣu, Forus | 0.67 (0.43–0.88) | 0.56 (0.46–0.65) | 0.31 (0.17–0.44) | 0.61 (0.48–0.73) | 0.22 (-0.03–0.46) | 0.16 (-0.02–0.32) |
