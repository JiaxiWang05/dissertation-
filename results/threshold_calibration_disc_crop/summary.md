# Chákṣu thresholds set on labelled images per camera (`frozen_lr_tuned_disc_crop_sens90` scores)

Calibration on 30% of each camera's eyes (glaucoma images in it: Remidio 45, Bosch 5, Forus 5); tested on the rest. Mean (2.5–97.5 percentile) over 200 random splits. Gap = largest minus smallest camera mean. Per camera: under 10 calibration glaucoma images means all are kept. Fairlearn: equal sensitivity on the calibration images, at the best balanced accuracy. PAPILA thresholds: PAPILA threshold, 90% sensitivity = 0.279.

| Rule | Remidio sensitivity | Bosch sensitivity | Forus sensitivity | Gap between camera means |
|---|---|---|---|---|
| PAPILA threshold, 90% sensitivity | 0.90 (0.88–0.93) | 0.52 (0.36–0.64) | 0.67 (0.54–0.85) | 0.38 |
| One Chákṣu threshold, 90% sensitivity | 0.93 (0.84–0.98) | 0.78 (0.31–1.00) | 0.74 (0.54–0.92) | 0.19 |
| Threshold per camera, at least 90% sensitivity | 0.89 (0.77–0.97) | 0.84 (0.36–1.00) | 0.83 (0.46–1.00) | 0.06 |
| Fairlearn, equal sensitivity on calibration images | 0.66 (0.40–0.83) | 0.52 (0.14–0.86) | 0.57 (0.15–0.93) | 0.14 |

| Rule | Remidio specificity | Bosch specificity | Forus specificity | All: sensitivity | All: specificity | All: PPV |
|---|---|---|---|---|---|---|
| PAPILA threshold, 90% sensitivity | 0.33 (0.31–0.34) | 0.63 (0.58–0.69) | 0.56 (0.49–0.62) | 0.84 (0.81–0.87) | 0.38 (0.36–0.40) | 0.18 (0.17–0.19) |
| One Chákṣu threshold, 90% sensitivity | 0.26 (0.19–0.36) | 0.55 (0.46–0.71) | 0.42 (0.29–0.63) | 0.89 (0.77–0.98) | 0.30 (0.24–0.42) | 0.17 (0.17–0.18) |
| Threshold per camera, at least 90% sensitivity | 0.32 (0.21–0.49) | 0.53 (0.44–0.65) | 0.30 (0.12–0.84) | 0.88 (0.78–0.96) | 0.34 (0.25–0.47) | 0.18 (0.17–0.20) |
| Fairlearn, equal sensitivity on calibration images | 0.66 (0.54–0.86) | 0.67 (0.54–0.86) | 0.71 (0.30–0.97) | 0.64 (0.41–0.79) | 0.66 (0.54–0.84) | 0.24 (0.21–0.30) |

## Size of the calibration set (threshold per camera)

| Calibration share | Glaucoma images per camera (R / B / F) | Gap between camera means | Bosch sensitivity | Forus sensitivity | All: specificity |
|---|---|---|---|---|---|
| 10% | 15 / 2 / 2 | 0.21 | 0.69 (0.12–1.00) | 0.66 (0.12–1.00) | 0.40 (0.23–0.64) |
| 20% | 30 / 4 / 4 | 0.07 | 0.81 (0.40–1.00) | 0.79 (0.36–1.00) | 0.39 (0.26–0.61) |
| 30% | 45 / 5 / 5 | 0.06 | 0.84 (0.36–1.00) | 0.83 (0.46–1.00) | 0.34 (0.25–0.47) |
| 50% | 75 / 9 / 9 | 0.05 | 0.85 (0.44–1.00) | 0.90 (0.56–1.00) | 0.33 (0.25–0.43) |
