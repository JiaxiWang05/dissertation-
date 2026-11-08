# Learning curve: RETFound-Green, full image, tuned logistic regression on frozen features

Mean ± SD over 10 random draws of the training images. PAPILA: AUROC inside each official test fold, averaged over the 5 folds (not the pooled out-of-fold AUROC of the main table). Chákṣu: trained on PAPILA only. 'all' = every training image, one fit. AUPRC of guessing: PAPILA 0.21, Chákṣu 0.14.

| Training images | PAPILA AUROC | Chákṣu AUROC | PAPILA AUPRC | Chákṣu AUPRC |
|---|---|---|---|---|
| 25 | 0.736 ± 0.034 | 0.633 ± 0.051 | 0.483 ± 0.051 | 0.226 ± 0.044 |
| 50 | 0.800 ± 0.022 | 0.618 ± 0.056 | 0.547 ± 0.032 | 0.221 ± 0.041 |
| 100 | 0.816 ± 0.021 | 0.672 ± 0.032 | 0.578 ± 0.049 | 0.260 ± 0.034 |
| 200 | 0.837 ± 0.016 | 0.675 ± 0.029 | 0.620 ± 0.030 | 0.265 ± 0.024 |
| all (336 / 420) | 0.856 | 0.698 | 0.659 | 0.280 |
