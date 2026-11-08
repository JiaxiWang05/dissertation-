# Learning curve: RETFound-Green, optic-disc crop, tuned logistic regression on frozen features

Mean ± SD over 10 random draws of the training images. PAPILA: AUROC inside each official test fold, averaged over the 5 folds (not the pooled out-of-fold AUROC of the main table). Chákṣu: trained on PAPILA only. 'all' = every training image, one fit. AUPRC of guessing: PAPILA 0.21, Chákṣu 0.14.

| Training images | PAPILA AUROC | Chákṣu AUROC | PAPILA AUPRC | Chákṣu AUPRC |
|---|---|---|---|---|
| 25 | 0.753 ± 0.028 | 0.653 ± 0.058 | 0.481 ± 0.033 | 0.245 ± 0.044 |
| 50 | 0.771 ± 0.034 | 0.649 ± 0.047 | 0.495 ± 0.017 | 0.243 ± 0.039 |
| 100 | 0.827 ± 0.011 | 0.672 ± 0.025 | 0.588 ± 0.036 | 0.274 ± 0.023 |
| 200 | 0.841 ± 0.023 | 0.693 ± 0.020 | 0.626 ± 0.037 | 0.299 ± 0.024 |
| all (336 / 420) | 0.859 | 0.716 | 0.635 | 0.329 |
