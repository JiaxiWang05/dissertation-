# Ablation: n_layers (best tuned setting per structure)

Paired differences are minus `n_layers 0` (PAPILA: scores ranked within each fold; Chákṣu: eye-level bootstrap). If the 95% CI contains 0 there is no clear difference.

| n_layers | PAPILA AUROC (95% CI) | Per fold | PAPILA vs reference | Chákṣu AUROC (95% CI) | Chákṣu vs reference |
|---|---|---|---|---|---|
| 0 | 0.854 (0.80–0.90) | 0.855 ± 0.033 |  | 0.703 (0.66–0.74) |  |
| 1 | 0.851 (0.79–0.90) | 0.853 ± 0.044 | -0.002 (-0.016 to +0.011) | 0.703 (0.66–0.74) | +0.001 (-0.018 to +0.020) |
| 2 | 0.814 (0.74–0.88) | 0.816 ± 0.033 | -0.039 (-0.070 to -0.006) | 0.607 (0.56–0.65) | -0.095 (-0.139 to -0.054) |
| 3 | 0.797 (0.73–0.85) | 0.800 ± 0.059 | -0.054 (-0.090 to -0.015) | 0.630 (0.59–0.67) | -0.073 (-0.110 to -0.037) |
