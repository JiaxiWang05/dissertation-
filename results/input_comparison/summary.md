# Paired AUROC difference vs `frozen_lr_tuned_resize` (tuned frozen logistic regression, full image; 95% CI)

PAPILA: scores ranked within each fold. Chákṣu: model trained on all PAPILA, eye-level bootstrap. If the interval contains 0 there is no clear difference.

| Input (same head) | PAPILA | Chákṣu, all | Remidio | Bosch | Forus |
|---|---|---|---|---|---|
| optic-disc crop | +0.005 (-0.021 to +0.031) | +0.018 (-0.012 to +0.047) | +0.028 (-0.006 to +0.064) | -0.025 (-0.137 to +0.067) | -0.015 (-0.092 to +0.062) |
| retina crop + pad | -0.006 (-0.022 to +0.011) | -0.007 (-0.029 to +0.016) | +0.006 (-0.020 to +0.032) | -0.162 (-0.261 to -0.066) | -0.046 (-0.092 to -0.003) |

| Optuna-chosen model (full image) | PAPILA | Chákṣu, all | Remidio | Bosch | Forus |
|---|---|---|---|---|---|
| Optuna head (frozen) | +0.000 (-0.012 to +0.011) | +0.005 (+0.002 to +0.007) | +0.006 (+0.003 to +0.009) | -0.005 (-0.017 to +0.006) | +0.006 (-0.002 to +0.015) |
| Optuna fine-tuning | +0.008 (-0.018 to +0.037) | -0.037 (-0.063 to -0.011) | -0.040 (-0.071 to -0.007) | -0.074 (-0.170 to +0.016) | +0.051 (+0.000 to +0.121) |
