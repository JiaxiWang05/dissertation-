# Small Data, Large Models: Transfer Learning for Medical Image Classification

BEng Research and Development Project (ENGI 3262, Durham University, 2026-27): glaucoma detection from frozen
RETFound-Green features on the Chákṣu and PAPILA fundus datasets.

| Folder | Contents |
|---|---|
| `pipeline/` | Feature extraction, linear probe, Optuna over the head and over unfrozen layers, fairness and ablations (Oct 2026) |
| `3_Nov/` | Chákṣu, frozen features: split, Optuna for logistic regression, SVM, kNN and MLP, learning curve, model choice |
| `after_3_Nov/` | New 70/15/15 split, MLP overfitting search scored by 5-fold cross-validation, checks |

Each folder has its own README. Only code is here: images, labels, features, predictions and Optuna databases
are not included. The scripts run on Durham's Hamilton cluster (`run_*.slurm`).
