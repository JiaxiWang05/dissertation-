# Ablation: unfrozen_blocks (best tuned setting per structure)

Paired differences are minus `frozen_lr_tuned_resize` (PAPILA: scores ranked within each fold; Chákṣu: eye-level bootstrap). If the 95% CI contains 0 there is no clear difference.

| unfrozen_blocks | PAPILA AUROC (95% CI) | Per fold | PAPILA vs reference | Chákṣu AUROC (95% CI) | Chákṣu vs reference |
|---|---|---|---|---|---|
| 0 | 0/5 folds |  |  |  |  |
| 1 | 0.845 (0.78–0.90) | 0.851 ± 0.042 | -0.005 (-0.020 to +0.010) | 0.715 (0.67–0.75) | +0.017 (-0.001 to +0.037) |
| 2 | 4/5 folds |  |  |  |  |
| 4 | 0.855 (0.80–0.90) | 0.858 ± 0.042 | +0.001 (-0.012 to +0.015) | 0.705 (0.66–0.75) | +0.007 (-0.018 to +0.032) |
| 6 | 0.861 (0.81–0.90) | 0.868 ± 0.045 | +0.011 (-0.006 to +0.030) | 0.661 (0.62–0.70) | -0.036 (-0.063 to -0.011) |
| 12 | 0.836 (0.77–0.90) | 0.866 ± 0.065 | +0.010 (-0.019 to +0.039) | 0.733 (0.69–0.77) | +0.035 (-0.001 to +0.071) |
