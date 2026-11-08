# Chákṣu thresholds set on labelled images per camera (`frozen_lr_tuned_resize` scores)

Calibration on 30% of each camera's eyes (glaucoma images in it: Remidio 45, Bosch 5, Forus 5); tested on the rest. Mean (2.5–97.5 percentile) over 200 random splits. Gap = largest minus smallest camera mean. Per camera: under 10 calibration glaucoma images means all are kept. Fairlearn: equal sensitivity on the calibration images, at the best balanced accuracy. PAPILA thresholds: PAPILA threshold, Youden's J = 0.514, PAPILA threshold, 90% sensitivity = 0.229.

| Rule | Remidio sensitivity | Bosch sensitivity | Forus sensitivity | Gap between camera means |
|---|---|---|---|---|
| PAPILA threshold, Youden's J | 0.79 (0.74–0.84) | 0.42 (0.29–0.54) | 0.55 (0.46–0.69) | 0.37 |
| PAPILA threshold, 90% sensitivity | 0.99 (0.99–1.00) | 1.00 (1.00–1.00) | 0.84 (0.77–0.92) | 0.16 |
| One Chákṣu threshold, 90% sensitivity | 0.93 (0.82–0.99) | 0.78 (0.57–1.00) | 0.67 (0.46–0.85) | 0.26 |
| Threshold per camera, at least 90% sensitivity | 0.88 (0.74–0.98) | 0.86 (0.50–1.00) | 0.84 (0.46–1.00) | 0.05 |
| Fairlearn, equal sensitivity on calibration images | 0.70 (0.48–0.83) | 0.59 (0.21–0.92) | 0.58 (0.23–0.92) | 0.12 |

| Rule | Remidio specificity | Bosch specificity | Forus specificity | All: sensitivity | All: specificity | All: PPV |
|---|---|---|---|---|---|---|
| PAPILA threshold, Youden's J | 0.51 (0.49–0.53) | 0.82 (0.76–0.88) | 0.84 (0.80–0.88) | 0.73 (0.68–0.77) | 0.58 (0.56–0.59) | 0.22 (0.21–0.23) |
| PAPILA threshold, 90% sensitivity | 0.03 (0.02–0.03) | 0.10 (0.06–0.13) | 0.36 (0.30–0.42) | 0.98 (0.97–0.99) | 0.07 (0.06–0.08) | 0.15 (0.14–0.15) |
| One Chákṣu threshold, 90% sensitivity | 0.19 (0.08–0.34) | 0.55 (0.35–0.74) | 0.70 (0.54–0.80) | 0.89 (0.77–0.97) | 0.28 (0.15–0.42) | 0.17 (0.16–0.18) |
| Threshold per camera, at least 90% sensitivity | 0.28 (0.15–0.52) | 0.40 (0.20–0.79) | 0.29 (0.00–0.84) | 0.88 (0.76–0.97) | 0.30 (0.15–0.51) | 0.17 (0.16–0.20) |
| Fairlearn, equal sensitivity on calibration images | 0.61 (0.46–0.75) | 0.68 (0.39–0.86) | 0.70 (0.30–0.96) | 0.67 (0.49–0.80) | 0.63 (0.49–0.75) | 0.23 (0.20–0.27) |

## Size of the calibration set (threshold per camera)

| Calibration share | Glaucoma images per camera (R / B / F) | Gap between camera means | Bosch sensitivity | Forus sensitivity | All: specificity |
|---|---|---|---|---|---|
| 10% | 15 / 2 / 2 | 0.20 | 0.69 (0.24–1.00) | 0.67 (0.12–1.00) | 0.37 (0.17–0.66) |
| 20% | 30 / 4 / 4 | 0.07 | 0.80 (0.40–1.00) | 0.79 (0.28–1.00) | 0.35 (0.18–0.54) |
| 30% | 45 / 5 / 5 | 0.05 | 0.86 (0.50–1.00) | 0.84 (0.46–1.00) | 0.30 (0.15–0.51) |
| 50% | 75 / 9 / 9 | 0.04 | 0.86 (0.56–1.00) | 0.90 (0.56–1.00) | 0.26 (0.18–0.37) |
